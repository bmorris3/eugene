import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from glob import glob

R0_grid = np.linspace(0.7, 8, 15)
k_grid = np.logspace(-2, 1, 15)

# R0_grid = np.arange(2, 5, 2)
# k_grid = np.logspace(-2, 1, 2)
red_plot = True
samples_plot = True

samples = np.vstack([np.load(p) for p in glob('samples/samples*.npy')])

if red_plot:
    kx = np.arange(0, len(k_grid), 4)
    # accepted_grid = np.load('results.npy')

    hist2d = np.histogram2d(samples[:, 0], np.log10(samples[:, 1]),
                            bins=[10, 15])[0]

    extent = [kx.min(), kx.max(), R0_grid.min(), R0_grid.max()]
    fig, ax = plt.subplots(figsize=(5, 4))
    # cax = ax.imshow(accepted_grid, extent=extent, origin='lower', aspect=2,
    #                 cmap=plt.cm.Reds)
    cax = ax.imshow(hist2d, extent=extent, origin='lower', aspect=3.5,
                    cmap=plt.cm.Reds)
    cbar = plt.colorbar(cax, label='Posterior density')
    ax.set_xticks(kx)
    ax.set_xticklabels([f"{k_grid[x]:.2g}" for x in kx])
    ax.set(xlabel='$k$', ylabel='$\mathcal{R}_0$')
    plt.savefig('plots/grid.pdf', bbox_inches='tight')
    plt.show()

if samples_plot:

    samples[:, 1] = np.log10(samples[:, 1])

    hist_kwargs = dict(plot_contours=False,
                       no_fill_contours=False, bins=6)

    corner(samples, labels=['$\mathcal{R}_0$', '$\log k$', '$D$', '$n$'],
                            #'$\sigma$', 'seed'],
           smooth=True, contour=False, **hist_kwargs)
    #plt.scatter(samples[:, 0], samples[:, 1], marker='.')
    #plt.xlabel('sigmas')
    #plt.ylabel('seed')
    plt.savefig('plots/corner.pdf', bbox_inches='tight')
    plt.show()
