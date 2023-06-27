import numpy as np
import matplotlib.pyplot as plt

cuda_time = [1.3790100000, 1.4661600000, 10.6963000000]
kokkos_time = [1.2334000000, 2.0857300000, 188.2920000]
cuda_time_op = [1.1913000000, 1.3025800000, 6.4871400000]
kokkos_time_op = [1.2334400000, 1.4904900000, 1401.6400000000]

width = 0.2

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.bar(np.arange(len(cuda_time)), cuda_time, width=width, label='CUDA (Coarse-grained)', color='#01BAEF')
ax1.bar(np.arange(len(kokkos_time)) + width, kokkos_time, width=width, label='Kokkos (Coarse-grained)', color='#118AB2')
ax1.bar(np.arange(len(cuda_time_op)) + 2 * width, cuda_time_op, width=width, label='CUDA (Fine-grained)',
        color='tab:orange')
ax1.bar(np.arange(len(kokkos_time_op)) + 3 * width, kokkos_time_op, width=width, label='Kokkos (Fine-grained)',
        color='#DDA15E')

ax1.set_xticks(np.arange(len(cuda_time)) + 1.5 * width)
ax1.set_xticklabels(['125*125', '1250*1250', '12500*12500'])
ax1.set_ylabel('Total Execution Time (s)')

ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
ax1.set_yscale('log')

# Calculate time change compared to CUDA Time Consumption
time_change = [(kokkos_time[i] / cuda_time[i] - 1) * 100 for i in range(len(cuda_time))]
time_change_op = [(kokkos_time_op[i] / cuda_time_op[i] - 1) * 100 for i in range(len(cuda_time_op))]

# Add annotations for time change
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

plt.ylim(1e-3, 1e4)

plt.tight_layout()
plt.savefig("cuda_time.png")
plt.show()
