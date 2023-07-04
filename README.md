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
│   ├── README.md
│   └── run.sh
├── openmp
│   ├── main.cpp
│   ├── Makefile
│   ├── README.md
│   └── run.sh
├── plot
│   ├── compile.py
│   ├── cuda_flops.py
│   ├── cuda_time.py
│   ├── openmp_flops.py
│   └── openmp_time.py
├── README.md
├── Kokkos_Paper.pdf
└── Kokkos_Presentation.pdf
```

The project aims to compare the performance of `Kokkos`, `OpenMP`, and `CUDA` implementations for matrix multiplication across different scales. The project directory consists of separate folders for each implementation, namely `kokkos`, `openmp`, and `cuda`. Each implementation contains source code files, a Makefile for compilation, a `README.md` file providing relevant information, and a `run.sh` script to calculate the average value of the data by executing the program multiple times.

The `plot` folder contains Python scripts for data visualization and analysis. These scripts include `compile.py` for compare the compile duration of different programming models, `openmp_flops.py`, and `cuda_flops.py` for plotting the performance comparison between the coarse- and fine-grained implementations, and `openmp_time.py` and `cuda_time.py` for plotting the execution of the programs.