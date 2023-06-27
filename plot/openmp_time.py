import numpy as np
import matplotlib.pyplot as plt

openmp_time = [0.0083719000, 0.9245270000, 2128.6000000]
kokkos_time = [0.0758677000, 0.5222410000, 1571.316000]
openmp_time_op = [0.0094181000, 0.6738300000, 569.177]
kokkos_time_op = [0.0170888000, 2.3096800000, 1387.12]

width = 0.2

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.bar(np.arange(len(openmp_time)), openmp_time, width=width, label='OpenMP (Coarse-grained)',
        color='#01BAEF')
ax1.bar(np.arange(len(kokkos_time)) + width, kokkos_time, width=width, label='Kokkos (Coarse-grained)',
        color='#118AB2')
ax1.bar(np.arange(len(openmp_time_op)) + 2 * width, openmp_time_op, width=width,
        label='OpenMP (Fine-grained)', color='tab:orange')
ax1.bar(np.arange(len(kokkos_time_op)) + 3 * width, kokkos_time_op, width=width,
        label='Kokkos (Fine-grained)', color='#DDA15E')

ax1.set_xticks(np.arange(len(openmp_time)) + 1.5 * width)
ax1.set_xticklabels(['125*125', '1250*1250', '12500*12500'])
ax1.set_ylabel('Total Execution Time (s)')
ax1.set_yscale('log')

ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Calculate performance change compared to OpenMP Time Consumption
time_change = [(kokkos_time[i] / openmp_time[i] - 1) * 100 for i in range(len(openmp_time))]
time_change_op = [(kokkos_time_op[i] / openmp_time_op[i] - 1) * 100 for i in range(len(openmp_time_op))]

# Add annotations for Time Consumption change
for i, change in enumerate(time_change):
    if change >= 0:
        ax1.annotate(f'+{change:.1f}%', xy=(i + width, kokkos_time[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')
    else:
        ax1.annotate(f'{change:.1f}%', xy=(i + width, kokkos_time[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')

for i, change in enumerate(time_change_op):
    if change >= 0:
        ax1.annotate(f'+{change:.1f}%', xy=(i + 3 * width, kokkos_time_op[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')
    else:
        ax1.annotate(f'{change:.1f}%', xy=(i + 3 * width, kokkos_time_op[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')

plt.ylim(1e-3, 1e4)  # Set the y-axis limits to show non-zero values only

plt.tight_layout()
plt.savefig("openmp_time.png")
plt.show()
