#include <iostream>

#include <Kokkos_Core.hpp>

#define M 10000
#define N 10000

int main(int argc, char** argv) {
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
                for (int i = 0; i < N; ++i) {
                    z(j, j) += x(j, i) * y(i, j);
                }
            });
        double time = timer.seconds();
        Kokkos::deep_copy(h_z, z);     // require for CUDA, why?
        for (int j = 0; j < M; j++) {
            double temp = 0.0;
            for (int i = 0; i < N; i++) {
                temp += h_x(j, i) * h_y(i, j);
            }
            if (temp != h_z(j, j)) {
                std::cout << "Error" << std::endl;
                Kokkos::finalize();
                return 0;
            }
        }
        std::cout << "Success: " << time << "s" << std::endl;
    }
    Kokkos::finalize();
    return 0;
}