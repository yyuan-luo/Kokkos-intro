import numpy as np
import matplotlib.pyplot as plt

width = 0.2

cuda_flops_op = [1822912000000, 1757808000000000, 1009114000000000000]
cuda_time_op = [1.1913000000, 1.3025800000, 6.4871400000]

kokkos_flops_op = [367766800000, 150187000000000, 71678620000000000]
kokkos_time_op = [1.2334400000, 1.4904900000, 1401.6400000000]

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.bar(np.arange(len(cuda_flops_op)), cuda_flops_op, width=width, label='CUDA FLOPS', color='#01BAEF')
ax1.bar(np.arange(len(kokkos_flops_op)) + width, kokkos_flops_op, width=width, label='Kokkos FLOPS', color='tab:orange')

ax2 = ax1.twinx()
ax2.bar(np.arange(len(cuda_time_op)) + 2 * width, cuda_time_op, width=width, label='CUDA Time Consumption', color='#118AB2')
ax2.bar(np.arange(len(kokkos_time_op)) + 3 * width, kokkos_time_op, width=width, label='Kokkos Time Consumption', color='#DDA15E')

ax1.set_xticks(np.arange(len(cuda_flops_op)) + 2 * width)
ax1.set_xticklabels(['125*125', '1250*1250', '12500*12500'])
ax1.set_ylabel('FLOPS')
ax1.set_yscale('log')

ax2.set_ylabel('Time Consumption(s)')

ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.85))

# Calculate performance change compared to CUDA FLOPS and CUDA time
flops_change = [(kokkos_flops_op[i] / cuda_flops_op[i] - 1) * 100 for i in range(len(cuda_flops_op))]
time_change = [(kokkos_time_op[i] / cuda_time_op[i] - 1) * 100 for i in range(len(cuda_time_op))]

# Add annotations for FLOPS change
for i, change in enumerate(flops_change):
    if change >= 0:
        ax1.annotate(f'+{change:.1f}%', xy=(i + width, kokkos_flops_op[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')
    else:
        ax1.annotate(f'{change:.1f}%', xy=(i + width, kokkos_flops_op[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')

# Add annotations for time change
for i, change in enumerate(time_change):
    if change >= 0:
        ax2.annotate(f'+{change:.1f}%', xy=(i + 3 * width, kokkos_time_op[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')
    else:
        ax2.annotate(f'{change:.1f}%', xy=(i + 3 * width, kokkos_time_op[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')

plt.title('CUDA and Kokkos Performance Comparison')

plt.tight_layout()
plt.savefig("cuda_optimized.png")
plt.show()
