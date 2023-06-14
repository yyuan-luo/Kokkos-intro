import numpy as np
import matplotlib.pyplot as plt

openmp_flops = [532579800, 4384916000, 1837514000]
kokkos_flops = [6108404000, 8006242000, 2492092000]
openmp_flops_op = [747344700, 7280303000, 7019946000]
kokkos_flops_op = [1089478000, 2922572000, 3046510000]

width = 0.2

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.bar(np.arange(len(openmp_flops)), openmp_flops, width=width, label='OpenMP FLOPS', color='#01BAEF')
ax1.bar(np.arange(len(kokkos_flops)) + width, kokkos_flops, width=width, label='Kokkos FLOPS', color='tab:orange')
ax1.bar(np.arange(len(openmp_flops_op)) + 2 * width, openmp_flops_op, width=width, label='Optimized OpenMP FLOPS', color='#118AB2')
ax1.bar(np.arange(len(kokkos_flops_op)) + 3 * width, kokkos_flops_op, width=width, label='Optimized Kokkos FLOPS', color='#DDA15E')

ax1.set_xticks(np.arange(len(openmp_flops)) + 2.5 * width)
ax1.set_xticklabels(['125*125', '1250*1250', '12500*12500'])
ax1.set_ylabel('FLOPS')
ax1.set_yscale('log')

ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Calculate performance change compared to OpenMP FLOPS
flops_change = [(kokkos_flops[i] / openmp_flops[i] - 1) * 100 for i in range(len(openmp_flops))]
flops_change_op = [(kokkos_flops_op[i] / openmp_flops_op[i] - 1) * 100 for i in range(len(openmp_flops_op))]

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

plt.ylim(1e8, 1e11)

plt.tight_layout()
plt.savefig("openmp_flops.png")
plt.show()
