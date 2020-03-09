import numpy as np
import matplotlib.pyplot as plt
from corner import corner
from glob import glob

from eugene.core import grouper
from grid_parallel import params

f_home_grid = params['f_home_grid']
max_community_spread_grid = params['max_community_spread_grid']
trials = params['trials']

blue_plot = True
red_plot = False
samples_plot = False

samples = np.vstack([np.load(p) for p in sorted(glob('samples/samples*.npy'))])

if blue_plot:
    #print(np.array(list(grouper(params['f_home_grid'], params['n_grid_points_per_process'])))[:, 0])
    fig, ax = plt.subplots(figsize=(5, 4))

    X, Y = np.meshgrid(f_home_grid, max_community_spread_grid)

    im = ax.pcolor(Y, X, np.median(samples, axis=-1).T,
                   cmap=plt.cm.Blues)
    cbar = plt.colorbar(im, label='Final size')

    ax.set(xlabel='$N_{\\rm max}$', ylabel='$f_{\\rm home}$')
    fig.savefig('plots/containment.pdf', bbox_inches='tight')
    plt.show()

if red_plot:
    hist2d, xedges, yedges = np.histogram2d(samples[:, 0],
                                            samples[:, 1],
                                            bins=[f_home_grid.shape[0],
                                                  max_community_spread_grid.shape[0]])

    fig, ax = plt.subplots(figsize=(5, 4))

    X, Y = np.meshgrid(f_home_grid, max_community_spread_grid)

    im = ax.pcolor(Y, X, hist2d.T / trials, cmap=plt.cm.Blues, vmin=0, vmax=1)

    # ax.set_xscale('log')
    # ax.set_yscale('log')

    cbar = plt.colorbar(im, label='Extinction fraction')

    ax.set(xlabel='$N_{\\rm max}$', ylabel='$f_{\\rm home}$')
    fig.savefig('plots/containment.pdf', bbox_inches='tight')
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

    # samples[:, 0] = np.log10(samples[:, 0])
    # samples[:, 1] = np.log10(samples[:, 1])

    hist_kwargs = dict(plot_contours=False, plot_datapoints=False,
                       no_fill_contours=False, bins=bins)

    corner(samples, labels=['$\log \mathcal{R}_0$', '$\log k$', '$D$', '$n$',
                            '$\Delta t$', '$\\alpha$'],
           smooth=True, contour=False, **hist_kwargs)

    plt.annotate(key_text, xy=(0.55, 0.8), fontsize=18,
                 ha='left', va='bottom', xycoords='figure fraction')

    plt.savefig('plots/corner.pdf', bbox_inches='tight')
    plt.show()
