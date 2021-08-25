#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/scatter.h>

#include <cuda.h>
#include <curand.h>

#include <stdlib.h>
#include <stdio.h>
static const int Device = 0;
static const int ThreadsPerBlock = 256;
static const int warpsize = 32;
static __device__ int topL, posL, topH, posH;

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void init(const int nodes,
          const int* const __restrict__ nidx,
          const int* const __restrict__ nlist,
          int* const __restrict__ nstat);

//static inline __device__ int representative(const int idx, int* const __restrict__ nstat);

/* process low-degree vertices at thread granularity and fill worklists */
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute1(const int nodes,
              const int* const __restrict__ nidx,
              const int* const __restrict__ nlist,
              int* const __restrict__ nstat,
              int* const __restrict__ wl);

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute2(const int nodes,
              const int* const __restrict__ nidx,
              const int* const __restrict__ nlist,
              int* const __restrict__ nstat,
              const int* const __restrict__ wl);

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute3(const int nodes,
              const int* const __restrict__ nidx,
              const int* const __restrict__ nlist,
              int* const __restrict__ nstat,
              const int* const __restrict__ wl);

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void flatten(const int nodes,
             const int* const __restrict__ nidx,
             const int* const __restrict__ nlist,
             int* const __restrict__ nstat);

static void computeCC(const int nodes, const int edges, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat);
// =======================
// TODO: send to .cu file
// =======================


/* initialize with first smaller neighbor ID */
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void init(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat){
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    const int beg = nidx[v];
    const int end = nidx[v + 1];
    int m = v;
    int i = beg;
    while ((m == v) && (i < end)) {
      m = min(m, nlist[i]);
      i++;
    }
    nstat[v] = m;
  }

  if (from == 0) {topL = 0; posL = 0; topH = nodes - 1; posH = nodes - 1;}
}

/* intermediate pointer jumping */
static inline __device__ int representative(const int idx, int* const __restrict__ nstat){
  int curr = nstat[idx];
  if (curr != idx) {
    int next, prev = idx;
    while (curr > (next = nstat[curr])) {
      nstat[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

/* process low-degree vertices at thread granularity and fill worklists */
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute1(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat, int* const __restrict__ wl){
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    const int vstat = nstat[v];
    if (v != vstat) {
      const int beg = nidx[v];
      const int end = nidx[v + 1];
      int deg = end - beg;
      if (deg > 16) {
        int idx;
        if (deg <= 352) {
          idx = atomicAdd(&topL, 1);
        } else {
          idx = atomicAdd(&topH, -1);
        }
        wl[idx] = v;
      } else {
        int vstat = representative(v, nstat);
        for (int i = beg; i < end; i++) {
          const int nli = nlist[i];
          if (v > nli) {
            int ostat = representative(nli, nstat);
            bool repeat;
            do {
              repeat = false;
              if (vstat != ostat) {
                int ret;
                if (vstat < ostat) {
                  if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                    ostat = ret;
                    repeat = true;
                  }
                } else {
                  if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                    vstat = ret;
                    repeat = true;
                  }
                }
              }
            } while (repeat);
          }
        }
      }
    }
  }
}

/* process medium-degree vertices at warp granularity */
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute2(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat, const int* const __restrict__ wl){
  const int lane = threadIdx.x % warpsize;

  int idx;
  if (lane == 0) idx = atomicAdd(&posL, 1);
  idx = __shfl_sync(0xffffffff, idx, 0);
  while (idx < topL) {
    const int v = wl[idx];
    int vstat = representative(v, nstat);
    for (int i = nidx[v] + lane; i < nidx[v + 1]; i += warpsize) {
      const int nli = nlist[i];
      if (v > nli) {
        int ostat = representative(nli, nstat);
        bool repeat;
        do {
          repeat = false;
          if (vstat != ostat) {
            int ret;
            if (vstat < ostat) {
              if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                ostat = ret;
                repeat = true;
              }
            } else {
              if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                vstat = ret;
                repeat = true;
              }
            }
          }
        } while (repeat);
      }
    }
    if (lane == 0) idx = atomicAdd(&posL, 1);
    idx = __shfl_sync(0xffffffff, idx, 0);
  }
}

/* process high-degree vertices at block granularity */
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute3(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat, const int* const __restrict__ wl){
  __shared__ int vB;
  if (threadIdx.x == 0) {
    const int idx = atomicAdd(&posH, -1);
    vB = (idx > topH) ? wl[idx] : -1;
  }
  __syncthreads();
  while (vB >= 0) {
    const int v = vB;
    __syncthreads();
    int vstat = representative(v, nstat);
    for (int i = nidx[v] + threadIdx.x; i < nidx[v + 1]; i += ThreadsPerBlock) {
      const int nli = nlist[i];
      if (v > nli) {
        int ostat = representative(nli, nstat);
        bool repeat;
        do {
          repeat = false;
          if (vstat != ostat) {
            int ret;
            if (vstat < ostat) {
              if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                ostat = ret;
                repeat = true;
              }
            } else {
              if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                vstat = ret;
                repeat = true;
              }
            }
          }
        } while (repeat);
      }
    }
    if (threadIdx.x == 0) {
      const int idx = atomicAdd(&posH, -1);
      vB = (idx > topH) ? wl[idx] : -1;
    }
    __syncthreads();
  }
}

/* link all vertices to sink */
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void flatten(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat){
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    int next, vstat = nstat[v];
    const int old = vstat;
    while (vstat > (next = nstat[vstat])) {
      vstat = next;
    }
    if (old != vstat) nstat[v] = vstat;
  }
}

/* link all vertices to sink */
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void arr_to_dict(const int n,
                 const int* const __restrict__ id,
                 const int* const __restrict__ keys,
                 int* const __restrict__ target){
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < n; v += incr) {
    target[id[v]] = keys[v];
  }
}

/* link all vertices to sink */
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void copy_key_to_arr(const int n,
                     const int* const __restrict__ id,
                     const int* const __restrict__ keys,
                     int* const __restrict__ target){
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < n; v += incr) {
    target[v] = keys[id[v]];
  }
}

/* link all vertices to sink */
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void replace_keys_to_values(const int n,
                     int* const __restrict__ keys,
                     const int* const __restrict__ values){
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < n; v += incr) {
    keys[v] = values[keys[v]];
  }
}


/* Compute the given node belongs to centrum component */
static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void is_in_center(const int n,
                  const int* const __restrict__ nstat_d,
                  int* const __restrict__ centrum_d,
                  int* const __restrict__ dict){
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < n; v += incr) {
      int curr_centrum_label = nstat_d[centrum_d[v]];
      dict[curr_centrum_label] = 1;
  }
}