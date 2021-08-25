#include "utils.h"
#include "ECL-CC.h"
#include "components.h"

#include <chrono>
#include <iostream>

#define time_point std::chrono::steady_clock::time_point
inline time_point measure_time(time_point &begin_tick, std::string label){
    time_point end_tick = std::chrono::steady_clock::now();
    std::cout<<label<<std::chrono::duration_cast<std::chrono::microseconds>(end_tick - begin_tick).count()<<
        "[ms]"<<std::endl;
    return end_tick;
}

int init_CUDA(){
    // === Try to allocate memory ===
    int* nodestatus = NULL;
    cudaHostAlloc(&nodestatus, 1 * sizeof(int), cudaHostAllocDefault);
    if (nodestatus == NULL) {
        fprintf(stderr, "ERROR: nodestatus - host memory allocation failed\n\n");  exit(-1);
    }
    
    // === Try to allocate device ===
    cudaSetDevice(Device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, Device);
    if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
      fprintf(stderr, "ERROR: there is no CUDA capable device\n\n");  exit(-1);
    }
    const int SMs = deviceProp.multiProcessorCount;
    const int mTSM = deviceProp.maxThreadsPerMultiProcessor;
    printf(">>> GPU: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n",
      deviceProp.name, SMs, mTSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);

    cudaFuncSetCacheConfig(init, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(compute1, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(compute2, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(compute3, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(flatten, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(copy_key_to_arr, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(arr_to_dict, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(is_in_center, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(replace_keys_to_values, cudaFuncCachePreferL1);
    
    int blocks = SMs * mTSM / ThreadsPerBlock;
    return blocks;
}

void test_functions(Input_graph& g){
    time_point read_end = std::chrono::steady_clock::now();
    
    g.get_CSR_format(-1);
    time_point csr_end = measure_time(read_end, ">>> CSR format = ");

    // === Label components ===
    g.compute_components();
    time_point base_algo_end = measure_time(csr_end, ">>> Base algo time = ");
    
    // === Collect components ===
    std::pair<int,int> res = g.component_sizes();
    time_point collect_algo_end = measure_time(base_algo_end, ">>> Collect input time = ");
    printf("Max component size: %d\nCentrum size: %d\n", res.first, res.second);
    
    // === New component sizes ===
    int sum = g.new_component_sizes();
    time_point collect_algo_end2 = measure_time(collect_algo_end, ">>> [New] Collect input time = ");
    printf("Max component size: -1\nCentrum size: %d\n", sum);
}

int main(int argc, char* argv[]){
    Args args(argc, argv);
    
    // ======================
    //       INIT CUDA
    // ======================
    time_point init_begin = std::chrono::steady_clock::now();
    const int blocks = init_CUDA();
    time_point init_end = measure_time(init_begin, ">>> Init time = ");
    
    // ======================
    //       COMPONENTS
    // ======================
    Input_graph g(blocks, &args);
    time_point read_inp = measure_time(init_end, ">>> Reading input time = ");
    
    if(args.mode == Args::MODE::test){
        // === TEST ===
        test_functions(g);
    }
    else if(args.mode == Args::MODE::simulation){
        // === RANDOM RUNS ===
        g.random_runs();
        
        // === Measure ps and ss ===
        g.measure_ps_ss();
    }
    else{
        std::cout<<"Mode not found\n";
    }
}