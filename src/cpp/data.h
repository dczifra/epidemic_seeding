#pragma once

#include <vector>
#include <random>
#include <array>

struct Args{
    double beta, beta_super;
    double p_worker, p_moving;
    double sigma, gamma;
    double second_launch=1.0;
    int random_seed;
    int max_sim;
    bool verbose = false;
};

enum SEIR{S,E,I,R,SEIR_SIZE};
enum COLS {location, seed, states, timers, home, work_location, SIZE};

class AgentData{
public:
    AgentData(unsigned int N_): N(N_){
        rows[location].resize(N);
        rows[home].resize(N);
        rows[work_location].resize(N, -1);

        rows[seed].resize(N, -1);
        rows[states].resize(N, 0);
        rows[timers].resize(N, 0);
    }
    
    std::vector<int>& operator[](COLS col){
        return rows[col];
    }
    unsigned int N;
private:
    std::vector<int> rows[COLS::SIZE];
};

class Graph{
    friend class Country;
public:
    Graph(int N, int random_seed): cityNum(N),SEIR(N),infecttion_prob(N), generator(random_seed),
                    population(N),
                    neigh_city_indexes(N, std::vector<int>()),
                    neigh_dirtibutions(N, std::discrete_distribution<int>()){
        
    }

    int get_random_neigh(int city){
        int rand = neigh_dirtibutions[city](generator);
        return neigh_city_indexes[city][rand];
    }

    void set_random_generator_for_city(int city, std::vector<double> distrib){
        neigh_dirtibutions[city] = std::discrete_distribution<int>(distrib.begin(), distrib.end());
    }

    int& get_neigh_city(int city, int index){
        return neigh_city_indexes[city][index];
    }

    void resize_city(int city, int neighNum){
        neigh_city_indexes[city].resize(neighNum);
    }

    unsigned int cityNum;
    std::vector<std::array<int,(int)SEIR_SIZE>> SEIR;
    std::vector<long double> infecttion_prob;
private:
    // For each city the neighbouring cities, and the probability, an agent moves to that city
    std::mt19937 generator;
    std::vector<int> population;
    std::vector<std::vector<int>> neigh_city_indexes;
    std::vector<std::discrete_distribution<int>> neigh_dirtibutions;
};