# Kokkos-intro

This project is to compare `Kokkos` with `OpenMP` and `Cuda` against large scale matrix multiplication. The implemantations are stored sepearately in different folders.

In each folder, there are `main.cpp`, `Makefile`, and `run.sh`. 

For `OpenMP` and `Cuda`, you can simple run the script, for example, `bash run.js 100 100 100`.

For `Kokkos`, you have to specify the devices as well, for example, `bash run.js OpenMP/Cuda 100 100 100`.

The script will calculate the average FLOPS and the average run time of the whole program.