#include <sys/time.h>
#include <iostream>

#define M 1250
#define N 1250

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

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            result[i][j] = 0.0;
            for (int k = 0; k < N; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
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
