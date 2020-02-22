import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from glob import glob

R0_grid = np.arange(0.8, 5, 0.2)
k_grid = np.logspace(-2, 1, 10)

# R0_grid = np.arange(2, 5, 2)
# k_grid = np.logspace(-2, 1, 2)
red_plot = True
samples_plot = True

if red_plot:
    kx = np.arange(0, len(k_grid), 2)
    # accepted_grid = np.load('results.npy')
    samples = np.vstack([np.load(p) for p in glob('samples*.npy')])

    hist2d = np.histogram2d(samples[:, 0], samples[:, 1])[0]

    extent = [kx.min(), kx.max(), R0_grid.min(), R0_grid.max()]
    fig, ax = plt.subplots(figsize=(5, 4))
    # cax = ax.imshow(accepted_grid, extent=extent, origin='lower', aspect=2,
    #                 cmap=plt.cm.Reds)
    cax = ax.imshow(hist2d, extent=extent, origin='lower', aspect=2,
                    cmap=plt.cm.Reds)
    cbar = plt.colorbar(cax, label='Acceptance')
    ax.set_xticks(kx)
    ax.set_xticklabels([f"{k_grid[x]:.2f}" for x in kx])
    ax.set(xlabel='k', ylabel='$\mathcal{R}_0$')
    plt.savefig('plots/grid.pdf', bbox_inches='tight')
    plt.show()

if samples_plot:

    samples = np.vstack([np.load(p) for p in glob('samples*.npy')])
    print(samples.shape)
    corner(samples, labels=['R0', 'k', 'sigmas', 'seed'], smooth=True)
    #plt.scatter(samples[:, 0], samples[:, 1], marker='.')
    #plt.xlabel('sigmas')
    #plt.ylabel('seed')
    plt.show()
