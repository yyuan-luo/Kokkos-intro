#include <omp.h>
#include <sys/time.h>
#include <iostream>

#define M 12500
#define N 12500

int main() {
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

    struct timeval start, end;
    gettimeofday(&start, nullptr);

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

    gettimeofday(&end, nullptr);
    double elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    std::cout << "Time: " << elapsedTime << "s" << std::endl;

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

    return 0;
}
