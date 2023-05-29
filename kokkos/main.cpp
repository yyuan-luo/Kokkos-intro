#include <iostream>
#include <sys/time.h>

#include <Kokkos_Core.hpp>

// #define M 12500
// #define N 12500

int main(int argc, char** argv) {
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
    Kokkos::initialize(argc, argv);
    {
        typedef Kokkos::View<double**> ViewMatrix;

        ViewMatrix x("x", M, N);
        ViewMatrix y("y", N, M);
        ViewMatrix z("z", M, M);

        ViewMatrix::HostMirror h_x = Kokkos::create_mirror_view(x);
        ViewMatrix::HostMirror h_y = Kokkos::create_mirror_view(y);
        ViewMatrix::HostMirror h_z = Kokkos::create_mirror_view(z);

        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++) {
                h_x(j, i) = 1.0;
            }
        }
        for (int j = 0; j < M; j++) {
            for (int i = 0; i < N; i++) {
                h_y(j, i) = 2.0;
            }
        }
        for (int j = 0; j < M; j++) {
            for (int i = 0; i < M; i++) {
                h_z(j, i) = 0.0;
            }
        }
        Kokkos::deep_copy(y, h_y);
        Kokkos::deep_copy(x, h_x);
        Kokkos::deep_copy(z, h_z);
        Kokkos::Timer timer;
        Kokkos::parallel_for(
            M, KOKKOS_LAMBDA(const size_t j) {
                for (int i = 0; i < M; ++i) {
                    for (int k = 0; k < N; k++) {
                        z(j, i) += x(j, k) * y(k, j);
                    }
                }
            });
        double time = timer.seconds();
        double flops = flops_count / time;
        std::cout << "Kernel: " << time << "s" << std::endl;
        std::cout << "FLOPS: " << flops << std::endl;
    }
    Kokkos::finalize();
    gettimeofday(&end, nullptr);
    double elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    std::cout << "Time: " << elapsedTime << "s" << std::endl;
    return 0;
}