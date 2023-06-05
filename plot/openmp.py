import numpy as np
import matplotlib.pyplot as plt

openmp_flops = [532579800, 4384916000, 1837514000]
kokkos_flops = [6108404000, 8006242000, 2492092000]
openmp_time = [0.0083719000, 0.9245270000, 2128.6000000]
kokkos_time = [0.0758677000, 0.5222410000, 1571.316000]

width = 0.2
colors = []

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.bar(np.arange(len(openmp_flops)), openmp_flops, width=width, label='OpenMP FLOPS', color='#01BAEF')
ax1.bar(np.arange(len(kokkos_flops)) + width, kokkos_flops, width=width, label='Kokkos FLOPS', color='tab:orange')

ax2 = ax1.twinx()
ax2.bar(np.arange(len(openmp_time)) + 2 * width, openmp_time, width=width, label='OpenMP Time Consumption', color='#118AB2')
ax2.bar(np.arange(len(kokkos_time)) + 3 * width, kokkos_time, width=width, label='Kokkos Time Consumption', color='#DDA15E')

ax1.set_xticks(np.arange(len(openmp_flops)) + 2 * width)
ax1.set_xticklabels(['125*125', '1250*1250', '12500*12500'])
ax1.set_ylabel('FLOPS')
ax1.set_yscale('log')

ax2.set_ylabel('Time Consumption(s)')

ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize='small')
ax2.legend(loc='upper right', fontsize='small')

# Calculate performance change compared to CUDA FLOPS and CUDA time
flops_change = [(kokkos_flops[i] / openmp_flops[i] - 1) * 100 for i in range(len(openmp_flops))]
time_change = [(kokkos_time[i] / openmp_time[i] - 1) * 100 for i in range(len(openmp_time))]

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

plt.title('OpenMP and Kokkos Performance Comparison')

plt.tight_layout()
plt.savefig("openmp.png")
plt.show()
