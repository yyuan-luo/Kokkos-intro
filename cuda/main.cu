#include <sys/time.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>

#define BLOCK_SIZE 32  // Define the block size for CUDA kernel

__global__ void matrixMultiplication(double* matrix1, double* matrix2, double* result, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < K) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            sum += matrix1[row * N + i] * matrix2[i * K + col];
        }
        result[row * K + col] = sum;
    }
}

int main(int argc, char const* argv[]) {
    if (argc < 4 && argc > 1) {
        printf("./main.cuda (M=1250) (N=1250) (K=1250)\n");
    }
    unsigned long int M = 1250;
    unsigned long int N = 1250;
    unsigned long int K = 1250;
    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    unsigned long long flops_count = 2 * M * N * K;
    struct timeval start, end;
    gettimeofday(&start, nullptr);

    double* matrix1 = new double[M * N];
    double* matrix2 = new double[N * K];
    double* result = new double[M * K];

    for (int i = 0; i < M * N; i++) {
        matrix1[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < N * K; i++) {
        matrix2[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    double* d_matrix1;
    double* d_matrix2;
    double* d_result;

    cudaMalloc((void**)&d_matrix1, M * N * sizeof(double));
    cudaMalloc((void**)&d_matrix2, N * K * sizeof(double));
    cudaMalloc((void**)&d_result, M * K * sizeof(double));

    cudaMemcpy(d_matrix1, matrix1, M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, N * K * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (K + blockDim.y - 1) / blockDim.y);

    auto kernel_start = std::chrono::high_resolution_clock::now();

    matrixMultiplication<<<gridDim, blockDim>>>(d_matrix1, d_matrix2, d_result, N, K);

    auto kernel_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);

    double flops = flops_count / (duration.count() * 1e-6);
    std::cout << "Kernel: " << (duration.count() * 1e-6) << "s" << std::endl;
    std::cout << "FLOPS: " << flops << std::endl;

    cudaMemcpy(result, d_result, M * K * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result);

    delete[] matrix1;
    delete[] matrix2;
    delete[] result;

    gettimeofday(&end, nullptr);
    double elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    std::cout << "Time: " << elapsedTime << "s" << std::endl;

    return 0;
}
