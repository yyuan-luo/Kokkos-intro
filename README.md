# Kokkos-intro

Project Structure:
```
├── cuda
│   ├── main.cu
│   ├── Makefile
│   ├── README.md
│   └── run.sh
├── kokkos
│   ├── main.cpp
│   ├── Makefile
│   ├── openmp_tranlate.cpp
│   ├── README.md
│   └── run.sh
├── openmp
│   ├── main.cpp
│   ├── Makefile
│   ├── README.md
│   └── run.sh
├── plot
│   ├── compile.py
│   ├── cuda_op.py
│   ├── cuda.py
│   ├── kokkos.py
│   ├── openmp_op.py
│   └── openmp.py
└── README.md
```

The project aims to compare the performance of `Kokkos`, `OpenMP`, and `CUDA` implementations for matrix multiplication across different scales. The project directory consists of separate folders for each implementation, namely `kokkos`, `openmp`, and `cuda`. Each implementation contains source code files, a Makefile for compilation, a `README.md` file providing relevant information, and a `run.sh` script for executing the program.

The `plot` folder contains Python scripts for data visualization and analysis. These scripts include `compile.py` for compare the compile duration of different programming models, `kokkos.py` for ploting the performance of Kokkos programming running on different backend programming models, `openmp.py`, and `cuda.py` for plotting the performance non-optimized comparison between the implementations, and `openmp_op.py` and `cuda_op.py` for plotting the performance of optimized versions.