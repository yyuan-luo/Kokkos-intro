#include <omp.h>
#include <sys/time.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>

void sequentialMatrixMultiplication(double** matrix1, double** matrix2, double** result, unsigned long M, unsigned long N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double temp = 0.0;
            for (int k = 0; k < N; ++k) {
                temp += matrix1[i][k] * matrix2[k][j];
            }
            result[i][j] = temp;
        }
    }
}

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

    double** matrix2 = new double*[M];
    for (int i = 0; i < M; i++) {
        matrix2[i] = new double[N];
    }

    double** matrix3 = new double*[N];
    for (int i = 0; i < N; i++) {
        matrix3[i] = new double[M];
    }

    double** result = new double*[M];
    for (int i = 0; i < M; i++) {
        result[i] = new double[M];
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix1[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix2[i][j] = static_cast<double>(rand()) / RAND_MAX;
            matrix3[j][i] = matrix2[i][j];
        }
    }
    auto kernel_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            double temp = 0.0;
        #pragma omp simd reduction(+ : temp)
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

    double** sequentialResult = new double*[M];
    for (int i = 0; i < M; i++) {
        sequentialResult[i] = new double[M];
    }

    sequentialMatrixMultiplication(matrix1, matrix3, sequentialResult, M, N);

    bool isValid = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            if (result[i][j] != sequentialResult[i][j]) {
                isValid = false;
                break;
            }
        }
        if (!isValid) {
            break;
        }
    }

    std::cout << "Validation: " << (isValid ? "Passed" : "Failed") << std::endl;

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
