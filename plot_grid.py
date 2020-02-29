import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from glob import glob

from grid_parallel import params

R0_grid = params['R0_grid']
k_grid = params['k_grid']
trials = params['trials']

red_plot = True
samples_plot = False

samples = np.vstack([np.load(p) for p in glob('samples/samples*.npy')])

lo, mid, hi = np.percentile(samples[:, 0], [16, 50, 84])
print(f'R0 = {mid:.2f}_{{-{mid-lo:.2f}}}^{{+{hi-mid:.2f}}}')

if red_plot:
    hist2d, xedges, yedges = np.histogram2d(np.log10(samples[:, 0]),
                                            np.log10(samples[:, 1]),
                                            bins=[R0_grid.shape[0],
                                                  k_grid.shape[0]])

    fig, ax = plt.subplots(figsize=(5, 4))

    X, Y = np.meshgrid(R0_grid, k_grid)

    im = ax.pcolor(Y, X, hist2d.T / trials, cmap=plt.cm.Reds)

    ax.set_xscale('log')
    ax.set_yscale('log')

    cbar = plt.colorbar(im, label='Acceptance fraction')

    ax.set(xlabel='$k$', ylabel='$\mathcal{R}_0$')
    fig.savefig('plots/grid.pdf', bbox_inches='tight')
    plt.show()

key_text = """Key:

$\log \mathcal{R}_0$: Reproduction number
$\log k$: Dispersion factor
$D$: Generation time interval [days]
$n$: Number of index cases
$\Delta t$: Time since index case [days]
$\\alpha$: Gamma function shape parameter"""

std_bin_size = 25
bins = [std_bin_size, 10, std_bin_size, 20, std_bin_size,
        std_bin_size]

if samples_plot:

    samples[:, 0] = np.log10(samples[:, 0])
    samples[:, 1] = np.log10(samples[:, 1])

    hist_kwargs = dict(plot_contours=False, plot_datapoints=False,
                       no_fill_contours=False, bins=bins)

    corner(samples, labels=['$\log \mathcal{R}_0$', '$\log k$', '$D$', '$n$',
                            '$\Delta t$', '$\\alpha$'],
           smooth=True, contour=False, **hist_kwargs)

    plt.annotate(key_text, xy=(0.55, 0.8), fontsize=18,
                 ha='left', va='bottom', xycoords='figure fraction')

    plt.savefig('plots/corner.pdf', bbox_inches='tight')
    plt.show()
