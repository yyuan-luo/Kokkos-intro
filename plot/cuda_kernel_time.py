import numpy as np
import matplotlib.pyplot as plt

cuda_kernel_time = [0.00000136363985, 0.0000017142886, 0.0000027649766]
kokkos_kernel_time = [0.0000155549653, 0.0000316318701, 0.0000653170567]
cuda_kernel_time_fine = [0.000002400, 0.0000026000, 0.0000047000]
kokkos_kernel_time_fine = [0.0000113774, 0.0000250690, 0.0000272483622]

width = 0.2

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.bar(np.arange(len(cuda_kernel_time)), cuda_kernel_time, width=width,
        label='CUDA (Coarse-grained)', color='#01BAEF')
ax1.bar(np.arange(len(kokkos_kernel_time)) + width, kokkos_kernel_time, width=width,
        label='Kokkos (Coarse-grained)',
        color='#118AB2')
ax1.bar(np.arange(len(cuda_kernel_time_fine)) + 2 * width, cuda_kernel_time_fine, width=width,
        label='CUDA (Fine-grained)',
        color='tab:orange')
ax1.bar(np.arange(len(kokkos_kernel_time_fine)) + 3 * width, kokkos_kernel_time_fine, width=width,
        label='Kokkos (Fine-grained)',
        color='#DDA15E')

ax1.set_xticks(np.arange(len(cuda_kernel_time)) + 1.5 * width)
ax1.set_xticklabels(['125*125', '1250*1250', '12500*12500'])
ax1.set_ylabel('Kernel Execution Time')
ax1.set_yscale('log')

ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Calculate performance change compared to CUDA FLOPS
flops_change = [(kokkos_kernel_time[i] / cuda_kernel_time[i] - 1) * 100 for i in range(len(cuda_kernel_time))]
flops_change_op = [(kokkos_kernel_time_fine[i] / cuda_kernel_time_fine[i] - 1) * 100 for i in
                   range(len(cuda_kernel_time_fine))]

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

plt.ylim(1e-06, 1e-4)

plt.tight_layout()
plt.savefig("cuda_kernel_time.png")
plt.show()
