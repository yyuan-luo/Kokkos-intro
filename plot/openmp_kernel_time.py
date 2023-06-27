import numpy as np
import matplotlib.pyplot as plt

# openmp_kernel_time = [532579800, 4384916000, 1837514000]
# kokkos_kernel_time = [6108404000, 8006242000, 2492092000]
# openmp_kernel_time_fine = [487384600, 6279978000, 6920900000]
# kokkos_kernel_time_fine = [1089478000, 2922572000, 3046510000]

openmp_kernel_time = [0.0036672, 0.44541, 1062.917]
kokkos_kernel_time = [0.00031974, 0.24395, 783.7290]
openmp_kernel_time_fine = [0.0089644000, 0.6336140000, 282.206]
kokkos_kernel_time_fine = [0.0043974800, 1.4115600000, 641.102]

width = 0.2

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.bar(np.arange(len(openmp_kernel_time)), openmp_kernel_time, width=width,
        label='OpenMP (Coarse-grained)', color='#01BAEF')
ax1.bar(np.arange(len(kokkos_kernel_time)) + width, kokkos_kernel_time, width=width,
        label='Kokkos (Coarse-grained)',
        color='#118AB2')
ax1.bar(np.arange(len(openmp_kernel_time_fine)) + 2 * width, openmp_kernel_time_fine, width=width,
        label='OpenMP (Fine-grained)',
        color='tab:orange')
ax1.bar(np.arange(len(kokkos_kernel_time_fine)) + 3 * width, kokkos_kernel_time_fine, width=width,
        label='Kokkos (Fine-grained)',
        color='#DDA15E')

ax1.set_xticks(np.arange(len(openmp_kernel_time)) + 1.5 * width)
ax1.set_xticklabels(['125*125', '1250*1250', '12500*12500'])
ax1.set_ylabel('Kernel Execution Time (s)')
ax1.set_yscale('log')

ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Calculate performance change compared to OpenMP FLOPS
flops_change = [(kokkos_kernel_time[i] / openmp_kernel_time[i] - 1) * 100 for i in range(len(openmp_kernel_time))]
flops_change_op = [(kokkos_kernel_time_fine[i] / openmp_kernel_time_fine[i] - 1) * 100 for i in
                   range(len(openmp_kernel_time_fine))]

# Add annotations for FLOPS change
for i, change in enumerate(flops_change):
    if change >= 0:
        ax1.annotate(f'+{change:.1f}%', xy=(i + width, kokkos_kernel_time[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')
    else:
        ax1.annotate(f'{change:.1f}%', xy=(i + width, kokkos_kernel_time[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')

for i, change in enumerate(flops_change_op):
    if change >= 0:
        ax1.annotate(f'+{change:.1f}%', xy=(i + 3 * width, kokkos_kernel_time_fine[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')
    else:
        ax1.annotate(f'{change:.1f}%', xy=(i + 3 * width, kokkos_kernel_time_fine[i]), xytext=(0, 5),
                     textcoords="offset points", ha='center', va='bottom')

plt.ylim(1e-4, 1e4)

plt.tight_layout()
plt.savefig("openmp_kernel_time.png")
plt.show()
