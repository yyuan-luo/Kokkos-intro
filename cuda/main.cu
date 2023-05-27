#include <sys/time.h>
#include <iostream>

#define M 12500
#define N 12500
#define BLOCK_SIZE 256 // Define the block size for CUDA kernel

__global__ void matrixMultiplication(double* matrix1, double* matrix2, double* result, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += matrix1[row * n + k] * matrix2[k + col * n];
        }
        result[row * n + col] = sum;
    }
}

int main() {
    double* matrix1 = new double[M * N];
    double* matrix2 = new double[N * M];
    double* result = new double[M * M];

    for (int i = 0; i < M * N; i++) {
        matrix1[i] = 1.0;
    }

    for (int i = 0; i < N * M; i++) {
        matrix2[i] = 2.0;
    }

    double* d_matrix1;
    double* d_matrix2;
    double* d_result;

    cudaMalloc((void**)&d_matrix1, M * N * sizeof(double));
    cudaMalloc((void**)&d_matrix2, N * M * sizeof(double));
    cudaMalloc((void**)&d_result, M * M * sizeof(double));

    cudaMemcpy(d_matrix1, matrix1, M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, N * M * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    struct timeval start, end;
    gettimeofday(&start, nullptr);

    matrixMultiplication<<<gridDim, blockDim>>>(d_matrix1, d_matrix2, d_result, M);

    gettimeofday(&end, nullptr);
    double elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    std::cout << "Time: " << elapsedTime << "s" << std::endl;

    cudaMemcpy(result, d_result, M * M * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result);

    delete[] matrix1;
    delete[] matrix2;
    delete[] result;

    return 0;
}
