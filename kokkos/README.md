## The Kokkos implementation

To be able to run the Kokkos implementation, you need to first git clone the [Kokkos project](https://github.com/kokkos/kokkos) and then replace the `KOKKOS_PATH = ${HOME}/kokkos` in the `Makefile` with your own path.

```Makefile
KOKKOS_PATH = /Your own path of Kokkos project/
```

- `Make build` to build the program
- `Make test KOKKOS_DEVICE=XX` to build and run the program with default parameters for matrix size. Replace `XX` either with `Cuda`, `OpenMP`, or `Serial`
- `bash run.sh Device M N K` to run the program 10 times and calculate the average kernel FLOPS performan and average execution time. Again replace the `Device` with either `OpenMP`, `Cuda`, or `Serial`. Assign `M`, `N`, `K` with some numbers you like or the program runs with default values.
- `openmp_translate.cpp` is a possible implementation of the optimized Kokkos program running in OpenMP.