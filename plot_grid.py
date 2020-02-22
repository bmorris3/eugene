import numpy as np
import matplotlib.pyplot as plt

R0_grid = np.arange(0.8, 3, 0.2)
k_grid = np.logspace(-2, 2, 10)

kx = np.arange(0, len(k_grid), 2)
accepted_grid = np.load('results.npy')

extent = [kx.min(), kx.max(), R0_grid.min(), R0_grid.max()]
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(accepted_grid, extent=extent, origin='lower', aspect=3,
          cmap=plt.cm.Reds)
ax.set_xticks(kx)
ax.set_xticklabels([f"{k_grid[x]:.2f}" for x in kx])
ax.set(xlabel='k', ylabel='$\mathcal{R}_0$')
plt.savefig('plots/grid.pdf', bbox_inches='tight')
plt.show()