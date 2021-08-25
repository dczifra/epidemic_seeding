/*
ECL-CC code: ECL-CC is a connected components graph algorithm. The CUDA
implementation thereof is quite fast. It operates on graphs stored in
binary CSR format.

Copyright (c) 2017-2020, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Jayadharini Jaiganesh and Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/.

Publication: This work is described in detail in the following paper.
Jayadharini Jaiganesh and Martin Burtscher. A High-Performance Connected
Components Implementation for GPUs. Proceedings of the 2018 ACM International
Symposium on High-Performance Parallel and Distributed Computing, pp. 92-104.
June 2018.
*/

#include <set>
#include <map>

#include "ECL-CC.h"
#include "components.h"


/*
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute_comp_size(const int num_keys,
                       thrust::device_vector<int> output_keys,
                       thrust::device_vector<int> output_freqs,
                       int* const __restrict__ comp_size_op){
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for(int v = from; v < num_keys; v += incr) {
      comp_size_op[0] = output_freqs[v];
      //comp_size_op[output_keys[v]] = output_freqs[v];
  }
}
*/

/* link all vertices to sink */
/*
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void largest_comp(const int nodes, const int* const __restrict__ nidx,
                  const int* const __restrict__ nlist,
                  int* const __restrict__ nstat,
                  int* const __restrict__ comp_size){
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;
  
  for(int v = from; v < nodes; v += incr) {
      comp_size[v]=0;
      //comp_size[v]=nstat[v];
      //comp_size[v]=2;
  }
  __syncthreads();

  for(int v = from; v < nodes; v += incr) {
      comp_size[nstat[v]]+=1;
  }
}
*/

struct create_map{
  __host__ __device__
  void operator()(int x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    printf("%d\n", x);
  }
};

struct GPUTimer{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg);  cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg);  cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0);  cudaEventSynchronize(end);  float ms;  cudaEventElapsedTime(&ms, beg, end);  return 0.001 * ms;}
};

/*
static void computeCC(const int nodes, const int edges, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat){
  cudaSetDevice(Device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, Device);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
      fprintf(stderr, "ERROR: there is no CUDA capable device\n\n");  exit(-1);
  }
  const int SMs = deviceProp.multiProcessorCount;
  const int mTSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("gpu: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n",
      deviceProp.name, SMs, mTSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);

  int* nidx_d;
  int* nlist_d;
  int* nstat_d;
  int* comp_size_d;
  int* comp_size_op;
  int* wl_d;

  if (cudaSuccess != cudaMalloc((void **)&nidx_d, (nodes + 1) * sizeof(int))) {
      fprintf(stderr, "ERROR: could not allocate nidx_d\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&nlist_d, edges * sizeof(int))) {
      fprintf(stderr, "ERROR: could not allocate nlist_d\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&nstat_d, nodes * sizeof(int))) {
      fprintf(stderr, "ERROR: could not allocate nstat_d,\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&comp_size_d, nodes * sizeof(int))) {
      fprintf(stderr, "ERROR: could not allocate comp_size_d,\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&comp_size_op, nodes * sizeof(int))) {
      fprintf(stderr, "ERROR: could not allocate comp_size_op,\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&wl_d, nodes * sizeof(int))) {
      fprintf(stderr, "ERROR: could not allocate wl_d,\n\n");  exit(-1);}

  if (cudaSuccess != cudaMemcpy(nidx_d, nidx, (nodes + 1) * sizeof(int), cudaMemcpyHostToDevice)) {
      fprintf(stderr, "ERROR: copying to device failed\n\n");  exit(-1);}
  if (cudaSuccess != cudaMemcpy(nlist_d, nlist, edges * sizeof(int), cudaMemcpyHostToDevice)) {
      fprintf(stderr, "ERROR: copying to device failed\n\n");  exit(-1);}

  cudaFuncSetCacheConfig(init, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(compute1, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(compute2, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(compute3, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(flatten, cudaFuncCachePreferL1);
    

  thrust::device_vector<int> dev_ones(nodes, 1);
  thrust::device_vector<int> output_keys(nodes);
  thrust::device_vector<int> output_freqs(nodes);
  
  const int blocks = SMs * mTSM / ThreadsPerBlock;
  GPUTimer timer;
  timer.start();
  init<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d);
  compute1<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d, wl_d);
  compute2<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d, wl_d);
  compute3<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d, wl_d);
  flatten<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d);
    
  thrust::device_vector<int> dev_keys(nstat_d, nstat_d+nodes);
  thrust::sort(dev_keys.begin(), dev_keys.end());
  thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;
  new_end = thrust::reduce_by_key(dev_keys.begin(), dev_keys.end(), dev_ones.begin(), output_keys.begin(),
                                  output_freqs.begin());
  int num_keys = new_end.first  - output_keys.begin();
  // === dev_keys: size of the components ===
  thrust::fill(dev_keys.begin(), dev_keys.end(), -1);
  thrust::scatter(thrust::device,
                output_freqs.begin(), output_freqs.begin()+num_keys,
                output_keys.begin(), dev_keys.begin());

  double runtime = timer.stop(); 

  printf("compute time: %.4f s\n", runtime);
  printf("throughput: %.3f Mnodes/s\n", nodes * 0.000001 / runtime);
  printf("throughput: %.3f Medges/s\n", edges * 0.000001 / runtime);
    
  if (cudaSuccess != cudaMemcpy(nstat, nstat_d, nodes * sizeof(int), cudaMemcpyDeviceToHost)) {
      fprintf(stderr, "ERROR: copying from device failed 1\n\n");  exit(-1);}
  
  
  thrust::host_vector<int> keys=output_keys;
  thrust::host_vector<int> freqs=output_freqs;
  
  for(int i=0;i<num_keys;i++){
     //printf("Key: %d Freq: %d\n",keys[i], freqs[i]);
  }
  
  //int* max_comp_size = NULL;
  //cudaHostAlloc(&max_comp_size, nodes * sizeof(int), cudaHostAllocDefault);
  //if (cudaSuccess != cudaMemcpy(max_comp_size, comp_size_d, nodes * sizeof(int), cudaMemcpyDeviceToHost)) {
  //    fprintf(stderr, "ERROR: copying from device failed 2\n\n");  exit(-1);}
    
  //printf("Largest comp size %d\n", max_comp_size[0]);
  thrust::host_vector<int> max_comp_size=dev_keys;
  for(int i=0;i<nodes;i++){
      //printf("Com size %d (%d): %d\n",i,nstat[i], max_comp_size[i]);
      
  }
  
  float* rand_arr;
  if (cudaSuccess != cudaMalloc((void **)&rand_arr, (nodes + 1) * sizeof(float))) {
      fprintf(stderr, "ERROR: could not allocate nidx_d\n\n");  exit(-1);}
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
  curandGenerateUniform(gen, rand_arr, edges);

  cudaFree(wl_d);
  cudaFree(nstat_d);
  cudaFree(comp_size_d);
  cudaFree(comp_size_op);
  cudaFree(nlist_d);
  cudaFree(nidx_d);
}
*/

static void verify(const int v, const int id, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat){
  if (nstat[v] >= 0) {
    if (nstat[v] != id) {fprintf(stderr, "ERROR: found incorrect ID value\n\n");  exit(-1);}
    nstat[v] = -1;
    for (int i = nidx[v]; i < nidx[v + 1]; i++) {
      verify(nlist[i], id, nidx, nlist, nstat);
    }
  }
}

