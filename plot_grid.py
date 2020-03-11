import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from grid_parallel import params

f_home_grid = params['f_home_grid']
k_grid = params['k_grid']
trials = params['trials']

blue_plot = False
k_contour_plot = True
samples_plot = False

samples = np.vstack([np.load(p) for p in sorted(glob('samples/samples*.npy'))])

if blue_plot:
    fig, ax = plt.subplots(figsize=(5, 4))

    X, Y = np.meshgrid(f_home_grid, k_grid)

    im = ax.pcolor(Y, X, np.median(samples, axis=-1).T,
                   cmap=plt.cm.Blues)

    ax.set_xscale('log')

    cbar = plt.colorbar(im, label='Final size')

    ax.set(xlabel='$k$', ylabel='$f_{\\rm home}$')
    fig.savefig('plots/containment.pdf', bbox_inches='tight')
    plt.show()

if k_contour_plot:
    fig, ax = plt.subplots(figsize=(5, 4))

    X, Y = np.meshgrid(f_home_grid, k_grid)
    Z = np.mean(samples, axis=-1).T

    levels = [0.1, 0.2, 0.35, 0.58, 1.18]
    CS = ax.contour(X, Z, Y, levels=levels, 
                    colors=plt.cm.viridis((np.log10(levels) + 1) / 1.2))
    ax.clabel(CS, inline=1, fontsize=10, fmt='$k = %.2f$', use_clabeltext=True)
    ax.set_yscale('log')
    ax.set_yticks([1e-3, 1e-2, 1e-1, 0])
    ax.set_yticklabels(['0.001', '0.01', '0.1', '1'])
    ax.set_xticks(np.arange(0.5, 1.0, 0.05), minor=True)
    ax.set_xticks(np.arange(0.5, 1.0, 0.1))
    ax.set_ylim([3e-4, 1])

    # cbar = plt.colorbar(im, label='Final size')

    ax.set_yticks([0.001, 0.01, 0.1, 1.0])
    ax.set_yticklabels(['0.001', '0.01', '0.1', '1.0'])

    ax.set_ylim([5e-4, 1])

    ax.set(ylabel='Final size', xlabel='$f_{\\rm home}$')
    fig.savefig('plots/contour.pdf', bbox_inches='tight')
    plt.show()

