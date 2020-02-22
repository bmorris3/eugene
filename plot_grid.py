import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from glob import glob

R0_grid = np.logspace(np.log10(0.7), np.log10(20), 15)
k_grid = np.logspace(-2, 1, 15)

red_plot = True
samples_plot = True

samples = np.vstack([np.load(p) for p in glob('samples/samples*.npy')])

if red_plot:
    kx = np.arange(0, len(k_grid), 2)
    ry = np.arange(0, len(R0_grid), 2)

    hist2d = np.histogram2d(np.log10(samples[:, 0]), np.log10(samples[:, 1]),
                            bins=[13, 15])[0]

    extent = [kx.min(), kx.max(), ry.min(), ry.max()]
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.imshow(hist2d, extent=extent, origin='lower', aspect=1.,
                    cmap=plt.cm.Reds)
    cbar = plt.colorbar(cax, label='Posterior density')
    ax.set_xticks(kx)
    ax.set_xticklabels([f"{k_grid[x]:.2g}" for x in kx])

    ax.set_yticks(ry)
    ax.set_yticklabels([f"{R0_grid[y]:.2g}" for y in ry])

    ax.set(xlabel='$k$', ylabel='$\mathcal{R}_0$')
    fig.savefig('plots/grid.pdf', bbox_inches='tight')
    plt.show()

if samples_plot:

    samples[:, 0] = np.log10(samples[:, 0])
    samples[:, 1] = np.log10(samples[:, 1])

    hist_kwargs = dict(plot_contours=False, plot_datapoints=False,
                       no_fill_contours=False, bins=5)

    corner(samples, labels=['$\log \mathcal{R}_0$', '$\log k$', '$D$', '$n$'],
           smooth=True, contour=False, **hist_kwargs)
    plt.savefig('plots/corner.pdf', bbox_inches='tight')
    plt.show()
