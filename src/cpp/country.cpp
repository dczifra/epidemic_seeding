#include <iostream>
#include <vector>
#include <numeric>
#include <stdlib.h>
#include <assert.h>

#include "country.h"

void Country::init_agents(std::map<int,int>& inf_cities){
    std::bernoulli_distribution p_worker(args.p_worker);

    int act_city = 0;
    int act_pop = 0;
    int all_agent_num = 0;
    int infect = inf_cities[0];
    while(act_city < graph.cityNum){
        if(act_pop == graph.population[act_city]){
            all_agent_num += act_pop;
            act_pop = 0;
            act_city++;
            infect = inf_cities[act_city];
        }
        else{
            int i = all_agent_num+act_pop;
            agents[location][i] = act_city;
            agents[home][i] = act_city;
            if(infect-- > 0){
                agents[states][i] = I;
                agents[timers][i] = std::min(20,I_time(generator)+1);
                //agents[timers][i] = 10;
            }
            
            if(p_worker(generator)){
                agents[work_location][i] = graph.get_random_neigh(act_city);
            }
            else{
                agents[work_location][i] = -1;
            }
            act_pop++;
        }
    }
    //std::cout<<all_agent_num<<" "<<agents.N<<std::endl;
    assert(all_agent_num == agents.N);
}


// === Read input ===
void Country::read_data(){
    // 1. Read graph: neighbour cities and probs
    for(unsigned int city=0;city<graph.cityNum;city++){
        // Population
        std::cin>>graph.population[city];
        graph.SEIR[city][S] = graph.population[city]; 

        // Neighbours and weights
        unsigned int neighNum;
        std::cin>>neighNum;
        graph.resize_city(city, neighNum+1);
        for(unsigned int j=0;j<neighNum;j++){
            std::cin>>graph.get_neigh_city(city, j);
        }
        graph.get_neigh_city(city, neighNum)=city;

        std::vector<double> neigh_w(neighNum+1);
        int traveller = 0;
        for(unsigned int j=0;j<neighNum;j++){
            std::cin>>neigh_w[j];
            traveller += neigh_w[j];
        }
        neigh_w[neighNum]=graph.population[city]-traveller;
        
        //assert(neigh_w[neighNum]>=0);
        if(neigh_w[neighNum]<0) neigh_w[neighNum]=0;
        
        graph.set_random_generator_for_city(city, neigh_w);
    }

    // 2. Read infected cities
    int inf_city_num;
    std::cin>>inf_city_num;
    std::map<int,int> inf_cities;
    for(int i=0;i<inf_city_num;i++){
        int inf_city, inf_agent_in_city;
        std::cin>>inf_city;
        std::cin>>inf_agent_in_city;

        graph.SEIR[inf_city][S] -= inf_agent_in_city;
        graph.SEIR[inf_city][I] += inf_agent_in_city;
        inf_cities[inf_city] = inf_agent_in_city;
    }

    // === Init data with args ===
    init_agents(inf_cities);
    //std::cout<<"Read cities\n";
}

void Country::go_home(){
    for(unsigned int agent=0; agent<agents.N; agent++){
        if(agents[work_location][agent] != -1 &&
           agents[location][agent] == agents[work_location][agent]){
            int old_loc = agents[location][agent];
            int new_loc = agents[home][agent];
            // === Go home ===
            agents[location][agent] = new_loc;
            // Update SEIR in city
            SEIR state = (SEIR) agents[COLS::states][agent];
            graph.SEIR[old_loc][state] -= 1;
            graph.SEIR[new_loc][state] += 1;
        }
    }
}

void Country::move(){
    std::vector<int> new_loc(agents[location]);
    for(unsigned int agent=0; agent<agents.N; agent++){
        int old_loc = agents[location][agent];
        // 1. Worker Agents
        if(agents[work_location][agent] != -1){
            // If at work ==> go home;
            // If at home ==> go work
            if(agents[location][agent] == agents[work_location][agent]){
                new_loc[agent]=agents[home][agent];
            }
            else if(p_moving(generator)){
                new_loc[agent]=agents[work_location][agent];
            }
        }
        // 2. Traveller Agents
        else if(p_moving(generator)){
            int act_city = agents[location][agent];
            // 2. Travellers
            int move_to = graph.get_random_neigh(act_city);
            new_loc[agent] = move_to;
        }
        // Update SEIR in city
        SEIR state = (SEIR) agents[COLS::states][agent];
        graph.SEIR[old_loc][state] -= 1;
        graph.SEIR[new_loc[agent]][state] += 1;
    }

    for(unsigned int i=0;i<agents.N;i++){
        agents[location][i] = new_loc[i];
    }
}

void Country::handle_S(unsigned int agent, int city){
    long double p = graph.infecttion_prob[city];
    if(p_inf(generator) < p){
        agents[states][agent] = E;
        agents[timers][agent] = E_time(generator);

        graph.SEIR[city][S]-=1;
        graph.SEIR[city][E]+=1;
    }
}

void Country::handle_E(unsigned int agent, int city){
    if(agents[timers][agent] == 0){
        agents[states][agent] = I;
        agents[timers][agent] = std::min(20, I_time(generator)+1);
        //agents[timers][agent] = std::min(10, I_time(generator)+1);

        graph.SEIR[city][E]-=1;
        graph.SEIR[city][I]+=1;
    }
    else{
        agents[timers][agent]-=1;
    }
}

void Country::handle_I(unsigned int agent, int city){
    if(agents[timers][agent] == 0){
        agents[states][agent] = R;
        agents[timers][agent] = 0;

        graph.SEIR[city][I]-=1;
        graph.SEIR[city][R]+=1;
    }
    else{
        agents[timers][agent]-=1;
    }
}

void operator+=(std::array<int,SEIR_SIZE>& base, const std::array<int,SEIR_SIZE>& data){
    for(int i=0;i<SEIR_SIZE;i++){
        base[(SEIR)i] += data[(SEIR) i];
    }
}

void operator<<(std::ostream& os, std::array<int,SEIR_SIZE>& base){
    for(int i=0;i<SEIR_SIZE;i++){
        os<<base[(SEIR)i]<<" ";
    }
    os<<std::endl;
}

int get_city_pop(std::array<int, SEIR::SEIR_SIZE> city_states){
    int sum = 0;
    for(int s: city_states){
        sum += s;
    }
    return sum;
}

void Country::infection(std::array<int, SEIR_SIZE>& stats, bool second_wave){
    // === Compute infection probability ===
    for(unsigned int city=0;city<graph.cityNum;city++){
        long double S_city = graph.SEIR[city][S];
        long double I_city = graph.SEIR[city][I];
        long double I_super = 0; // TODO
        long double pop = graph.population[city];
        int contactNum = 1;
        long double p;
        //std::cout<<"I "<<I_city<<std::endl;

        if(S_city == 0){
            p=0;
        }
        else{
            double beta = second_wave?1.2*args.beta:args.beta;
            long double p_simple = std::pow((1-contactNum*beta/pop), I_city);
            long double p_super = std::pow((1-contactNum*args.beta_super/pop), I_super);
            p = (long double)1.0-p_simple*p_super;
            //p = args.beta*I_city/graph.population[city];
            //std::cout.precision(17);
            //std::cout<<"    "<<p<<" "<<I_city<<std::endl;
        }
        
        graph.infecttion_prob[city] = p;
    }

    // === Infect/heal agents ===
    for(unsigned int agent=0; agent<agents.N; agent++){
        int city = agents[location][agent];
        
        if(agents[states][agent] == S){
            handle_S(agent, city);
        }

        if(agents[states][agent] == E){
            handle_E(agent, city);
        }

        if(agents[states][agent] == I){
            handle_I(agent, city);
        }
    }
    
    for(unsigned int city=0;city<graph.cityNum;city++){
        stats+=graph.SEIR[city];
    }
}

void debug(int iteration_num, AgentData& agents){
    std::cout<<iteration_num<<". iter\n";
    for(unsigned int i=0;i<agents.N;i++){
        std::cout<<agents[states][i]<<"("<<agents[location][i]<<") ";
    }
    std::cout<<std::endl;    
}

void log_cities(const Graph& graph){
    for(unsigned int city=0;city<graph.cityNum;city++){
        std::cout<<graph.SEIR[city][S]<<" "<<graph.SEIR[city][E]<<" "<<
                   graph.SEIR[city][I]<<" "<<graph.SEIR[city][R]<<";";
    }
    std::cout<<std::endl;
}

void init_stats(const Graph& graph, std::array<int,SEIR_SIZE>& base, bool verbose){
    for(unsigned int city=0;city<graph.cityNum;city++){
        base+=graph.SEIR[city];
    }
    std::cout<<base;
    if(verbose) log_cities(graph);
}

void Country::simulate(){
    // === INIT ===
    std::vector<std::array<int, SEIR_SIZE>> stats(args.max_sim+1, {0,0,0,0});
    init_stats(graph, stats[0], args.verbose);
    
    for(int iteration_num=1;iteration_num<=args.max_sim; iteration_num++){
        // 1. Move agents
        move();

        // 2. Infect agents
        bool second_wave = (stats[iteration_num-1][I]+stats[iteration_num-1][R] > agents.N*args.second_launch);
        infection(stats[iteration_num], second_wave);
        
        // 3. Go home, and infect there too: TODO: take care for travaller agents
        //go_home();
        //infection(stats[iteration_num], second_wave);
        
        // === Log ===
        std::cout<<stats[iteration_num];
        if(args.verbose) log_cities(graph);
	    if(stats[iteration_num][I]==0 && stats[iteration_num][E]==0) break;
    }
}
