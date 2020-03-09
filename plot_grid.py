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
    Z = np.median(samples, axis=-1).T

    # im = ax.pcolor(Y, X, k_grid,
    #                cmap=plt.cm.Reds)
    CS = ax.contour(X, Z, Y, levels=[0.1, 0.3, 0.5, 1.0, 3.0, 5.0])
    ax.clabel(CS, inline=1, fontsize=10, fmt='$k = %.2f$')
    ax.set_yscale('log')
    ax.set_ylim([1e-4, 1])
    # cbar = plt.colorbar(im, label='Final size')

    ax.set(ylabel='Final size', xlabel='$f_{\\rm home}$')
    fig.savefig('plots/containment.pdf', bbox_inches='tight')
    plt.show()

