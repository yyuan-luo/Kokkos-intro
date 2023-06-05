import numpy as np
import matplotlib.pyplot as plt

cuda_flops = [1432288000000, 1139321000000000, 706380300000000000]
kokkos_flops = [125562800000, 61745480000000, 29902220000000000]
cuda_time = [1.3790100000, 1.4661600000, 10.6963000000]
kokkos_time = [1.2334000000, 2.0857300000, 188.2920000]

width = 0.2

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.bar(np.arange(len(cuda_flops)), cuda_flops, width=width, label='CUDA FLOPS', color='#01BAEF')
ax1.bar(np.arange(len(kokkos_flops)) + width, kokkos_flops, width=width, label='Kokkos FLOPS', color='tab:orange')

ax2 = ax1.twinx()
ax2.bar(np.arange(len(cuda_time)) + 2 * width, cuda_time, width=width, label='CUDA Time Consumption', color='#118AB2')
ax2.bar(np.arange(len(kokkos_time)) + 3 * width, kokkos_time, width=width, label='Kokkos Time Consumption', color='#DDA15E')

ax1.set_xticks(np.arange(len(cuda_flops)) + 2 * width)
ax1.set_xticklabels(['125*125', '1250*1250', '12500*12500'])
ax1.set_ylabel('FLOPS')
ax1.set_yscale('log')

ax2.set_ylabel('Time Consumption(s)')

ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.85))

# Calculate performance change compared to CUDA FLOPS and CUDA time
flops_change = [(kokkos_flops[i] / cuda_flops[i] - 1) * 100 for i in range(len(cuda_flops))]
time_change = [(kokkos_time[i] / cuda_time[i] - 1) * 100 for i in range(len(cuda_time))]

# Add annotations for FLOPS change
for i, change in enumerate(flops_change):
    if change >= 0:
        ax1.annotate(f'+{change:.1f}%', xy=(i + width, kokkos_flops[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')
    else:
        ax1.annotate(f'{change:.1f}%', xy=(i + width, kokkos_flops[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')

# Add annotations for time change
for i, change in enumerate(time_change):
    if change >= 0:
        ax2.annotate(f'+{change:.1f}%', xy=(i + 3 * width, kokkos_time[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')
    else:
        ax2.annotate(f'{change:.1f}%', xy=(i + 3 * width, kokkos_time[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')

plt.title('CUDA and Kokkos Performance Comparison')

plt.tight_layout()
plt.savefig("cuda.png")
plt.show()
