import numpy as np
import matplotlib.pyplot as plt

cuda_flops = [1432288000000, 1139321000000000, 706380300000000000]
kokkos_flops = [125562800000, 61745480000000, 29902220000000000]
cuda_flops_op = [1822912000000, 1757808000000000, 1009114000000000000]
kokkos_flops_op = [367766800000, 150187000000000, 71678620000000000]

width = 0.2

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.bar(np.arange(len(cuda_flops)), cuda_flops, width=width, label='CUDA FLOPS', color='#01BAEF')
ax1.bar(np.arange(len(kokkos_flops)) + width, kokkos_flops, width=width, label='Kokkos FLOPS', color='tab:orange')
ax1.bar(np.arange(len(cuda_flops_op)) + 2 * width, cuda_flops_op, width=width, label='Optimized CUDA FLOPS', color='#118AB2')
ax1.bar(np.arange(len(kokkos_flops_op)) + 3 * width, kokkos_flops_op, width=width, label='Optimized Kokkos FLOPS', color='#DDA15E')

ax1.set_xticks(np.arange(len(cuda_flops)) + 2.5 * width)
ax1.set_xticklabels(['125*125', '1250*1250', '12500*12500'])
ax1.set_ylabel('FLOPS')
ax1.set_yscale('log')

ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Calculate performance change compared to CUDA FLOPS
flops_change = [(kokkos_flops[i] / cuda_flops[i] - 1) * 100 for i in range(len(cuda_flops))]
flops_change_op = [(kokkos_flops_op[i] / cuda_flops_op[i] - 1) * 100 for i in range(len(cuda_flops_op))]

# Add annotations for FLOPS change
for i, change in enumerate(flops_change):
    if change >= 0:
        ax1.annotate(f'+{change:.1f}%', xy=(i + width, kokkos_flops[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')
    else:
        ax1.annotate(f'{change:.1f}%', xy=(i + width, kokkos_flops[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')

for i, change in enumerate(flops_change_op):
    if change >= 0:
        ax1.annotate(f'+{change:.1f}%', xy=(i + 3 * width, kokkos_flops_op[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')
    else:
        ax1.annotate(f'{change:.1f}%', xy=(i + 3 * width, kokkos_flops_op[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')

plt.ylim(1e8, 1e19)

plt.tight_layout()
plt.savefig("cuda_flops.png")
plt.show()
