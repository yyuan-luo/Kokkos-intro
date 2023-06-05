import numpy as np
import matplotlib.pyplot as plt

labels = ['OpenMP', 'Kokkos OpenMP', 'CUDA', 'Kokkos CUDA']
duration = [0.521, 49.931, 2.385, 140.886]

fig, ax = plt.subplots(figsize=(8, 5))

bar_width = 0.35
opacity = 0.8

index = np.arange(len(labels))

rects1 = ax.bar(index[0:2], duration[0:2], bar_width,
                alpha=opacity,
                color='b',
                label='OpenMP')

rects2 = ax.bar(index[2:], duration[2:], bar_width,
                alpha=opacity,
                color='r',
                label='CUDA')

ax.set_ylabel('Compilation Duration (s)')
ax.set_title('Comparison on Compilation Durations')
ax.set_xticks(index)
ax.set_xticklabels(labels)

# Add comparison percentages
percentage1 = int((duration[1] - duration[0]) / duration[0] * 100)
ax.text(rects1[1].get_x() + rects1[1].get_width() / 2, rects1[1].get_height(), f'+{percentage1}%', ha='center', va='bottom')

percentage2 = int((duration[3] - duration[2]) / duration[2] * 100)
ax.text(rects2[1].get_x() + rects2[1].get_width() / 2, rects2[1].get_height(), f'+{percentage2}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("compile.png")
plt.show()
