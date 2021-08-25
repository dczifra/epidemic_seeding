#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/scatter.h>
#include <thrust/extrema.h>

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <vector>
#include <random>

#define CUDA_CALL(x) if((x)!=cudaSuccess) { \
        printf("Error at %s:%d\n",__FILE__,__LINE__);exit(1);}
#define COPY(x) if (cudaSuccess != x){ \
        fprintf(stderr, "ERROR: copying to device failed\n\n");  exit(-1);}

/**
==========
Compile:
    * time nvcc -std=c++11 -O3 -lcurand -arch=sm_35 ./cuda/ECL-CC_11.cu -o ecl-cc
==========
**/

class Args;

class Input_graph{
public:
    Input_graph(int b_, Args* args_);

    ~Input_graph(){
        delete u;
        delete v;
        delete w;
        delete centrum;
    }
    
    void read_input();
    void init_GPU_variables();
    void compute_components();
    int new_component_sizes();
    void get_CSR_format(int random);
    std::pair<int,int> component_sizes();
    
    // === Complete measurements ===
    std::vector<int> random_runs();
    void measure_ps_ss();
    
    // === Helper functions ===
    template<typename T>
    void print_arr(T* arr_d, int size, std::string label, bool verbose = false);
    
private:
    void drop_out_edges(int random);
// TODO: allocate only one temp_nodes and temp_edges
private:
    Args* args;
    int undir_edges;
    
    int blocks;
    int nodes, edges, centrum_size;
    int *u,*v;
    float* w;
    int* centrum;
    
    int* centrum_d;
    int* nidx_d;
    int* nlist_d;
    int* nstat_d;
    int* wl_d;
    int *u_undir_d, *v_undir_d;
    
    curandGenerator_t gen;
    std::mt19937 generator;
    
    // === Temp vars for CSR format ===
    //int* edge_index;
    thrust::device_vector<int> temp_u;
    
    // === Temp variables for max component ===
    float* rand_arr;
    thrust::device_vector<int> dev_ones;
    thrust::device_vector<int> output_keys;
    thrust::device_vector<int> output_freqs;
    thrust::device_vector<int> comp_size_d;
    
    // === Variables for getting the centrum comonents ===
    int* dict_d;

};
