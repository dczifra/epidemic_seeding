#pragma once
#include "data.h"

#include <map>

class Country{
public:
    Country(const Args& args_, int N, int M): args(args_),
                                              agents(N), graph(M, args_.random_seed),
                                              generator(args_.random_seed),
                                              p_inf(0.0,1.0){
        // TODO init SEIR
        set_random_generators();
        read_data();
    }

    void move();
    void go_home();
    void infection(std::array<int, SEIR_SIZE>& stats, bool second_wave);
    void simulate();

    // INIT
    void init_agents(std::map<int,int>& inf_cities);
    // Read input
    void read_data();
    void read_args(int argc, char* argv[]);
    
    // Helper functions
    void handle_S(unsigned int agent, int city);
    void handle_E(unsigned int agent, int city);
    void handle_I(unsigned int agent, int city);

    void set_random_generators(){
        E_time = std::geometric_distribution<int>(args.sigma);
        I_time = std::geometric_distribution<int>(args.gamma);
        p_moving = std::bernoulli_distribution(args.p_moving);
    }

    Args args;
private:
    AgentData agents;
    Graph graph;

    std::mt19937 generator;
    std::geometric_distribution<int> E_time;
    std::geometric_distribution<int> I_time;
    std::bernoulli_distribution p_moving; // True with p; False with 1-p
    std::uniform_real_distribution<long double> p_inf;

};