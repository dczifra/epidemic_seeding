#include "utils.h"
#include "ECL-CC.h"
#include "components.h"
#include <map>
#include <algorithm>

/**
===========
  Tricks:
===========
  * Undirected edges are stored, and after dropping out edges (with p prob.) we add
    the other direction
  * (X) Before making the CSR format, set node_{n+1} adjacent to other nodes. That will
    make sure, that all nodes will have >= 1 neighbours. (After the reduce_by_key
    we can leave out node_{n+1}).

**/

Input_graph::Input_graph(int b_, Args* args_): blocks(b_), args(args_),
                                               generator(args->random_seed){
    read_input();
    
    init_GPU_variables();
}

void Input_graph::read_input(){
    std::cin>>nodes>>undir_edges>>centrum_size;
    
    edges = 2*undir_edges;
    
    u = new int[undir_edges];
    v = new int[undir_edges];
    w = new float[undir_edges];
    centrum = new int[centrum_size];
    
    // === Read edges ===
    for(int i=0;i<undir_edges;i++){
        std::cin>>u[i]>>v[i]>>w[i];
    }
    
    // === Read centrum nodes ===
    for(int i=0;i<centrum_size;i++){
        std::cin>>centrum[i];
    }
    
    if(args->verbose) std::cout<<">>> Input read.\n";
}

template<typename T>
void Input_graph::print_arr(T* arr_d, int size, std::string label, bool verbose){
    if(!verbose && !args->verbose) return;
    
    T arr_host[size];
    cudaMemcpy(arr_host, arr_d, size * sizeof(T), cudaMemcpyDeviceToHost);

    std::cout<<label.c_str();
    for(int i=0;i<size;i++){
        std::cout<<*(arr_host+i)<<" ";
    }
    std::cout<<"\n";
}

template<typename T>
struct drop_rand
{
   float threshold;
   drop_rand(float th) : threshold(th){}
   
  __host__ __device__
  bool operator()(const T &x){
    return x > threshold;
  }
};

void Input_graph::drop_out_edges(int random){
    if(random > 0){
        int shift = 0;
        //int shift = (random-1)*undir_edges;
        float p = args->p;
        drop_rand<float> random_gen(p);
        thrust::device_ptr<float> curr_rand=thrust::device_pointer_cast(rand_arr+shift);

        // === Drop U ===
        auto new_nodes_end = thrust::remove_if(temp_u.begin(), temp_u.begin()+undir_edges,
                                               curr_rand, random_gen);
        // === Drop V ===
        thrust::device_ptr<int> nlist_dev_ptr = thrust::device_pointer_cast(nlist_d);
        auto nlist_end = thrust::remove_if(nlist_dev_ptr, nlist_dev_ptr+undir_edges,
                                           curr_rand, random_gen);
        
        edges = 2*(nlist_end-nlist_dev_ptr);
        
        /*
        std::cout<<"temp_u: ";
        for(int i=0;i<edges/2; i++) std::cout<<temp_u[i]<<" ";
        print_arr<int>(nlist_d, edges/2, "\ntemp_v: ", true);
        std::cout<<"Number of (undir) edges: "<<edges/2<<std::endl;*/
    }
    else{
        edges = 2*undir_edges;
    }
}

void Input_graph::get_CSR_format(int random){
    // === INIT source and target nodes of the edges ===
    thrust::copy(thrust::device, u_undir_d, u_undir_d+undir_edges, temp_u.begin());
    thrust::copy(thrust::device, v_undir_d, v_undir_d+undir_edges, nlist_d);

    // === Drop out edges based on p ===
    drop_out_edges(random);
    
    // === Direct the edges: add the other direction ===
    int split = edges/2;
    thrust::copy(thrust::device, nlist_d, nlist_d+split, temp_u.begin()+split);
    thrust::copy(thrust::device, temp_u.begin(), temp_u.begin()+split, nlist_d+split);
    
    
    //(XXX)   Trick 2: Add every node with zero weight (degree must be >= 1)
    //(XXX)   thrust::sequence(thrust::device, temp_u.begin()+edges, temp_u.begin()+edges+nodes, 0);
    
    // === Sort edges [based on source] ===
    //   ==> The target nodes will be nlist_d
    thrust::sort_by_key(thrust::device, temp_u.begin(),
                        temp_u.begin()+edges, nlist_d);
        
    // === Count the DEGREE by reduction [temp_u is sorted] ===
    // TODO: allocate memory previously
    thrust::device_vector<int> temp_keys(edges);
    thrust::device_vector<int> ones(edges, 1);
    
    int* nidx_temp = wl_d;
    thrust::pair<thrust::device_vector<int>::iterator, int*> new_end;
    new_end = thrust::reduce_by_key(thrust::device, temp_u.begin(),
                                    temp_u.begin()+edges, ones.begin(),
                                    temp_keys.begin(), nidx_temp);
    
    // === Insert 0 degrees into nidx_temp ===
    // convert the (key: node_id, value: degree) pairs into array of degrees
    thrust::fill(thrust::device, nidx_d, nidx_d+nodes+1, 0);
    int num_keys = new_end.first  - temp_keys.begin();
    int* temp_keys_raw = thrust::raw_pointer_cast(temp_keys.data());
    
    arr_to_dict<<<blocks, ThreadsPerBlock>>>(num_keys, temp_keys_raw, nidx_temp, nidx_d+1);
    
    print_arr<int>(temp_keys_raw, num_keys, "Keys: ", false);
    print_arr<int>(wl_d, num_keys, "Values: ", false);
    print_arr<int>(nidx_d, nodes+1, "Values (good): ", false);
    
    // === Compute NIDX from degrees ===
    thrust::plus<int> plus_op;
    thrust::inclusive_scan(thrust::device, nidx_d+1,
                           nidx_d+(nodes+1), nidx_d+1, plus_op);
    
    print_arr<int>(nlist_d, edges, "Edge end: ", false);
    print_arr<int>(nidx_d, nodes+1, "Nidx: ", false);
}

void Input_graph::init_GPU_variables(){
    // === INIT target nodes of the edges ===
    CUDA_CALL(cudaMalloc((void **)&u_undir_d, undir_edges*sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&v_undir_d, undir_edges*sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&nlist_d, (edges)*sizeof(int)));
    COPY(cudaMemcpy(u_undir_d, u, (undir_edges)*sizeof(int), cudaMemcpyHostToDevice));
    COPY(cudaMemcpy(v_undir_d, v, (undir_edges)*sizeof(int), cudaMemcpyHostToDevice));
    
    // === INIT centrum nodes ===
    CUDA_CALL(cudaMalloc((void **)&centrum_d, (centrum_size)*sizeof(int)));
    COPY(cudaMemcpy(centrum_d,centrum, (centrum_size)*sizeof(int), cudaMemcpyHostToDevice));
    
    // === INIT other variables ===
    CUDA_CALL(cudaMalloc((void **)&wl_d, (nodes)*sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&nstat_d, (nodes)*sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&nidx_d, (nodes+1)*sizeof(int)));
    CUDA_CALL(cudaMalloc((void **)&dict_d, (nodes)*sizeof(int)));
    
    // === Generate random vectors ===
    int N = undir_edges;
    CUDA_CALL(cudaMalloc((void **)&rand_arr, N*sizeof(float)));
    
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    curandSetPseudoRandomGeneratorSeed(gen, args->random_seed);
    curandGenerateUniform(gen, rand_arr, N);

    // === Init temp parameters ===
    dev_ones.resize(nodes);
    output_keys.resize(nodes);
    output_freqs.resize(nodes);
    comp_size_d.resize(nodes);
    
    temp_u.resize(edges);
    
    if(args->verbose) std::cout<<"GPU variables initialized\n";
}

void Input_graph::compute_components(){
    init<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d);
    compute1<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d, wl_d);
    compute2<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d, wl_d);
    compute3<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d, wl_d);
    flatten<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d);
}

template<typename T>
struct is_center_label
{
   T* dict;
   is_center_label(T* d) : dict(d){}
   
  __host__ __device__
  bool operator()(const T &x){
    return dict[x] > 0;
  }
};
// === compute simple sizes === 
int Input_graph::new_component_sizes(){
    // If we know the component sizes comp_size_d
    
    thrust::fill(thrust::device, dict_d, dict_d+nodes, 0);
    //=== Compute the is_in_center dictionary ===
    int centrum_size = args->s; // Use only s city of the centrum
    is_in_center<<<blocks, ThreadsPerBlock>>>(centrum_size, nstat_d, centrum_d, dict_d);
    
    int sum = thrust::count_if(thrust::device, nstat_d, nstat_d+nodes, is_center_label<int>(dict_d));
    return sum;
}

std::pair<int,int> Input_graph::component_sizes(){
    // ===========================
    //     Get component sizes
    // ===========================
    thrust::fill(dev_ones.begin(), dev_ones.begin()+nodes,1);
    
    thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(nstat_d);
    thrust::copy(thrust::device, dev_ptr, dev_ptr+nodes, comp_size_d.begin());
    
    // TODO: sort takes a lot of time: approx 80%
    thrust::sort(comp_size_d.begin(), comp_size_d.begin()+nodes);
    // TODO reduce by key takes a lot of time: approx 20%
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;
    new_end = thrust::reduce_by_key(comp_size_d.begin(), comp_size_d.begin()+nodes,
                                    dev_ones.begin(), output_keys.begin(),
                                    output_freqs.begin());
    
    int num_keys = new_end.first  - output_keys.begin();
    // === comp_sizes: size of the components ===
    thrust::fill(comp_size_d.begin(), comp_size_d.begin()+nodes, -1);
    thrust::scatter(thrust::device,
                output_freqs.begin(), output_freqs.begin()+num_keys,
                output_keys.begin(), comp_size_d.begin());

    // === Getmaximum component ===
    thrust::device_vector<int>::iterator iter =
        thrust::max_element(comp_size_d.begin(), comp_size_d.begin()+nodes);
    
    // ===========================
    //     Count centrum nodes
    // ===========================
    // === Use only s city of the centrum ===
    int centrum_size = args->s;
    std::cout<<"Centrum size: "<<centrum_size<<std::endl;
    int* centrum_comp_key;
    CUDA_CALL(cudaMalloc((void **)&centrum_comp_key, (centrum_size)*sizeof(int)));
    
    // === Component ids of the centrum nodes ===
    copy_key_to_arr<<<blocks, ThreadsPerBlock>>>(centrum_size, centrum_d, nstat_d, centrum_comp_key);
    //print_arr<int>(centrum_comp_key, centrum_size, "Centrum component IDs0 (not unique): ");
    
    // === Get unique keys ===
    thrust::sort(thrust::device, centrum_comp_key, centrum_comp_key+centrum_size);
    int* new_comp_end = thrust::unique(thrust::device, centrum_comp_key, centrum_comp_key+centrum_size);
    int new_centrum_size = new_comp_end-centrum_comp_key;
    
    // === Replace component ID with component size ===
    //print_arr<int>(centrum_comp_key, new_centrum_size, "Centrum component IDs: ");
    replace_keys_to_values<<<blocks, ThreadsPerBlock>>>(new_centrum_size, centrum_comp_key,
                                                        thrust::raw_pointer_cast(comp_size_d.data()));
    //print_arr<int>(centrum_comp_key, new_centrum_size, "Centrum component sizes: ");
    
    // === Sum the components based on componen size ===
    int result = thrust::reduce(thrust::device, centrum_comp_key,
                           centrum_comp_key+new_centrum_size, 0);
    
    // === Return largest component, and centrum size
    return {(*iter), result};
}

std::vector<int> Input_graph::random_runs(){
    std::vector<int> sims(args->simulation_num,0);
    
    for(int i=0; i< args->simulation_num; i++){
        // Regenerate random variables:
        curandGenerateUniform(gen, rand_arr, undir_edges);
        get_CSR_format(i+1);
        compute_components();
        sims[i] = new_component_sizes();
    }
    std::cout<<"[RES] ";
    for(int i=0; i< args->simulation_num; i++) std::cout<<sims[i]<<" ";
    std::cout<<std::endl;
    
    return sims;
}

void Input_graph::measure_ps_ss(){
    std::map<std::pair<double, int>, std::vector<int>> sim_results;
    //std::map<double, std::vector<int>> num_edges;
    
    for(int i=0; i< args->simulation_num; i++){
        // === Generate new p ===
        curandGenerateUniform(gen, rand_arr, undir_edges); // we can save that
        //print_arr<float>(rand_arr, 10, "rand_arr: ", true);
        
        //std::random_shuffle(centrum, centrum+centrum_size);
        if(args->per){
            std::shuffle(centrum, centrum+centrum_size, generator);
            COPY(cudaMemcpy(centrum_d,centrum, (centrum_size)*sizeof(int), cudaMemcpyHostToDevice));
            //print_arr<int>(centrum_d, centrum_size, "Centrum: ", true);
        }
        
        for(auto p: args->ps){
            curandGenerateUniform(gen, rand_arr, undir_edges); // we could reuse
            args->p = p;
            get_CSR_format(1);
            compute_components();
            //num_edges[p].push_back(edges/2);
            for(auto s: args->ss){
                args->s = s;
                int res = new_component_sizes();

                if(sim_results.find({p,s}) == sim_results.end()){
                    sim_results[{p,s}] = {res};
                }
                else sim_results[{p,s}].push_back(res);
            }
        }
    }
    std::cout<<"[SIMS]\n";
    /*
    for(auto p: args->ps){
        for(auto s: args->ss){
            double mean = 0.0;
            for(int i=0; i< args->simulation_num; i++){
                mean += sim_results[{p,s}][i];
            }
            std::cout<<mean/args->simulation_num<<" ";
        }
        std::cout<<std::endl;
    }*/
    std::cout.precision(8);
    for(auto p: args->ps){
        for(auto s: args->ss){
            std::cout<<p<<","<<s<<":";
            for(int i=0; i< args->simulation_num; i++){
                std::cout<<sim_results[{p,s}][i]<<(i< args->simulation_num-1?" ":"");
            }
            std::cout<<"\n";
        }
    }
    // Number of edges
    /*
    for(auto p: args->ps){
        double mean = 0.0;
        for(int i=0; i< args->simulation_num; i++){
            mean += num_edges[p][i];
        }
    }
    */
}
