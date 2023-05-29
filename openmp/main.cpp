#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>
#include <chrono>
#include <iostream>

// #define M 1250
// #define N 1250

int main(int argc, char const* argv[]) {
    if (argc < 3) {
        printf("./main.openmp (M=1250) (N=1250)\n");
    }

    unsigned long M = 1250;
    unsigned long N = 1250;
    if (argc == 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
    }
    unsigned long long flops_count = 2 * N * N * N;
    struct timeval start, end;
    gettimeofday(&start, nullptr);
    double** matrix1 = new double*[M];
    for (int i = 0; i < M; i++) {
        matrix1[i] = new double[N];
    }

    double** matrix2 = new double*[N];
    for (int i = 0; i < N; i++) {
        matrix2[i] = new double[M];
    }

    double** result = new double*[M];
    for (int i = 0; i < M; i++) {
        result[i] = new double[M];
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix1[i][j] = 1.0;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            matrix2[i][j] = 2.0;
        }
    }

    auto kernel_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double temp = 0.0;
            // #pragma omp parallel for reduction(+: temp)
            for (int k = 0; k < N; ++k) {
                temp += matrix1[i][k] * matrix2[k][j];
            }
            result[i][j] = temp;
        }
    }
    auto kernel_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);

    double flops = flops_count / (duration.count() * 1e-6);
    std::cout << "Kernel: " << (duration.count() * 1e-6) << "s" << std::endl;
    std::cout << "FLOPS: " << flops << std::endl;

    for (int i = 0; i < M; i++) {
        delete[] matrix1[i];
    }
    delete[] matrix1;

    for (int i = 0; i < N; i++) {
        delete[] matrix2[i];
    }
    delete[] matrix2;

    for (int i = 0; i < M; i++) {
        delete[] result[i];
    }
    delete[] result;
    gettimeofday(&end, nullptr);
    double elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    std::cout << "Time: " << elapsedTime << "s" << std::endl;
    return 0;
}
