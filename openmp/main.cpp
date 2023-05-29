#include <omp.h>
#include <sys/time.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>

int main(int argc, char const* argv[]) {
    if (argc < 4 && argc > 1) {
        printf("./main.openmp (M=1250) (N=1250) (K=1250)\n");
    }

    unsigned long M = 1250;
    unsigned long N = 1250;
    unsigned long K = 1250;
    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    unsigned long long flops_count = 2 * M * N * K;
    struct timeval start, end;
    gettimeofday(&start, nullptr);
    double** matrix1 = new double*[M];
    for (int i = 0; i < M; i++) {
        matrix1[i] = new double[N];
    }

    double** matrix2 = new double*[K];
    for (int i = 0; i < K; i++) {
        matrix2[i] = new double[N];
    }

    double** matrix3 = new double*[N];
    for (int i = 0; i < N; i++) {
        matrix3[i] = new double[K];
    }

    double** result = new double*[M];
    for (int i = 0; i < M; i++) {
        result[i] = new double[K];
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix1[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            matrix2[i][j] = static_cast<double>(rand()) / RAND_MAX;
            matrix3[j][i] = matrix2[i][j];
        }
    }
    auto kernel_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            double temp = 0.0;
// #pragma omp simd reduction(+ : temp)
            for (int k = 0; k < N; ++k) {
                temp += matrix1[i][k] * matrix2[j][k];
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
