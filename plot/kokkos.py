import numpy as np
import matplotlib.pyplot as plt

width = 0.2

kokkos_openmp = [6108404000, 8006242000, 2492092000]
kokkos_openmp_op = [1089478000, 2922572000, 3046510000]

kokkos_cuda = [125562800000, 61745480000000, 29902220000000000]
kokkos_cuda_op = [367766800000, 150187000000000, 71678620000000000]

fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(np.arange(len(kokkos_openmp)), kokkos_openmp, width=width, label='Kokkos OpenMP FLOPS', color='#01BAEF')
ax.bar(np.arange(len(kokkos_openmp_op)) + width, kokkos_openmp_op, width=width, label='Optimized Kokkos OpenMP FLOPS',
       color='#118AB2')
ax.bar(np.arange(len(kokkos_cuda)) + 2 * width, kokkos_cuda, width=width, label='Kokkos CUDA FLOPS', color='tab:orange')
ax.bar(np.arange(len(kokkos_cuda_op)) + 3 * width, kokkos_cuda_op, width=width, label='Optimized Kokkos CUDA FLOPS',
       color='#DDA15E')

ax.set_xticks(np.arange(len(kokkos_openmp)) + 2 * width)
ax.set_xticklabels(['125*125', '1250*1250', '12500*12500'])
ax.set_ylabel('FLOPS')
ax.set_yscale('log')

ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Calculate performance change compared to CUDA FLOPS and CUDA time
flops_change = [(kokkos_openmp_op[i] / kokkos_openmp[i] - 1) * 100 for i in range(len(kokkos_openmp))]
time_change = [(kokkos_cuda_op[i] / kokkos_cuda[i] - 1) * 100 for i in range(len(kokkos_cuda))]

# Add annotations for FLOPS change
for i, change in enumerate(flops_change):
    if change >= 0:
        ax.annotate(f'+{change:.1f}%', xy=(i + width, kokkos_openmp_op[i]), xytext=(0, 5),
                    textcoords="offset points", ha='center', va='bottom')
    else:
        ax.annotate(f'{change:.1f}%', xy=(i + width, kokkos_openmp_op[i]), xytext=(0, 5),
                    textcoords="offset points", ha='center', va='bottom')

# Add annotations for time change
for i, change in enumerate(time_change):
    if change >= 0:
        ax.annotate(f'+{change:.1f}%', xy=(i + 3 * width, kokkos_cuda_op[i]), xytext=(0, 5),
                    textcoords="offset points", ha='center', va='bottom')
    else:
        ax.annotate(f'{change:.1f}%', xy=(i + 3 * width, kokkos_cuda_op[i]), xytext=(0, 5),
                    textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("kokkos.png")
plt.show()
