# epidemic_seeding
Epidemic modelling with metapopulation and percolation models, based on the https://arxiv.org/abs/2106.16070 paper.

The repo is organized as follows:

* src/cuda : source code for the percolation model
* src/cpp : source for the metapopulation model
* src/utils : source code for preprocessing
* paper : launch scripts for the figures of the paper

# Initialization

## Required packages
Python
```
numpy powerlaw matplotlib scipy networkx subprocess pickle multiprcessing pandas json
```

C++
```
>=g++7.5
```

CUDA
```
>= nvcc 10.0
```

## Build
```
# GPU code for percolation
./src/bin/build_ecl
# C++ code for metapop model
./src/bin/build_cpp
```

