*********************************
Reproducing Riou & Althaus (2020)
*********************************

Approximate Bayesian Computation
++++++++++++++++++++++++++++++++

In this short tutorial, we'll show you how to reproduce the results of
`Riou & Althaus (2020)
<https://doi.org/10.2807/1560-7917.ES.2020.25.4.2000058>`_, which showed that
the early COVID-19 in Wuhan, China has :math:`\mathcal{R}_0 \sim 2` using
Approximate Bayesian Computation.

First we'll import some of the required packages and ``eugene``::

    import numpy as np
    import matplotlib.pyplot as plt
    from glob import glob

    from eugene import abc

Next we'll set the parameters of the run, which we explain below::

    params = dict(
        # Grid of R0 and k parameters to iterate over
        R0_grid = np.logspace(np.log10(0.7), np.log10(10), 50),
        k_grid = np.logspace(-2, 1, 10),

        # Maximum number of cases to run the simulation through (should be
        # greater than ``max_number_cases``)
        max_cases = 1e4,

        # Maximum number of days someone might transmit the disease
        max_time = 90,   # days

        # Number of stochastic trials to run at each grid-point
        trials = 1000,

        # Days elapsed since zoonotic transmission
        days_elapsed_min = [46-7, 52-7],  # days
        days_elapsed_max = [46+7, 52+7],  # days

        # Number of cases after ``days_elapsed``
        min_number_cases = [190, 1000],  # cases
        max_number_cases = [5590, 9700],  # cases

        # Initial number of index cases n (day-zero cases)
        n_min = 1,   # cases
        n_max = 100,  # cases

        # Generation interval/Gamma function shape parameter
        gamma_shape_min = 1,
        gamma_shape_max = 5,

        # Generation time interval D
        D_min = 7,   # days
        D_max = 60,  # days

        # Computer parameters
        n_processes = 16,
        n_grid_points_per_process = 2,

        # Formatting string for naming simulation outputs
        samples_path = 'samples/samples{0}.npy'
    )

The number of processes ``n_processes`` should be equivalent to the number of
processes you can run simultaneously on your machine.
``n_grid_points_per_process`` determines the number of :math:`\mathcal{R}_0`
grid points distributed to each process. The ``samples_path`` argument will
determine where to put the chains from the rejection sampler -- you'll need to
create a ``samples/`` directory for this example to work.

Finally, we can run the simulation with the following::

    total_trials = (params['trials'] * params['R0_grid'].shape[0] *
                    params['k_grid'].shape[0])
    print(f'Total number of simulations triggered: {total_trials}')

    abc(**params)

The Approximate Bayesian Computation ``abc`` function will run simple parallel
processes that each save the accepted chains from a rejection sampler algorithm.

Visualizing the results
+++++++++++++++++++++++

We can view the acceptance rate of the chains as a function of
:math:`\mathcal{R}_0` and :math:`k` with the following commands::

    R0_grid = params['R0_grid']
    k_grid = params['k_grid']
    trials = params['trials']

    samples = np.vstack([np.load(p) for p in glob('samples/samples*.npy')])

    lo, mid, hi = np.percentile(samples[:, 0], [16, 50, 84])
    print(f'R0 = {mid:.2f}_{{-{mid-lo:.2f}}}^{{+{hi-mid:.2f}}}')


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

.. image:: plots/grid.pdf
  :width: 800
  :alt: Acceptance rates for R0 and k

The plot above shows the acceptance rate of the ABC rejection sampler as a
function of :math:`\mathcal{R}_0` and :math:`k`, darker red represents higher
acceptance rates, meaning a better match between the simulated cumulative
incidence curves and the observations. The median :math:`\mathcal{R}_0 \sim 2`,
meaning for every case of COVID-19 there are approximately two new cases
generated, and :math:`k \gtrsim 0.1`.

Parameter degeneracies
++++++++++++++++++++++

Since we sampled for a range of :math:`\mathcal{R}_0, k, D, n`, and
``gamma_shape`` parameters which we will call :math:`\alpha`, we can plot the
fraction of accepted rejection sampler iterations as a function of each
combination of these parameters to examine how the uncertainty on one parameter
propagates into uncertainties on the others.

We can generate a *corner plot* with our results like so::

    from corner import corner

    key_text = """Key:

    $\log \mathcal{R}_0$: Reproduction number
    $\log k$: Dispersion factor
    $D$: Generation time interval [days]
    $n$: Number of index cases
    $\Delta t$: Time since index case [days]
    $\\alpha$: Gamma function shape parameter"""

    std_bin_size = 25
    bins = [std_bin_size, std_bin_size - 15, std_bin_size, std_bin_size - 5,
            std_bin_size, std_bin_size]

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

.. image:: plots/corner.pdf
  :width: 800
  :alt: Corner plot for R0, k, n, D and alpha

We investigate the larger uncertainties and long tail towards large
:math:`\mathcal{R}_0` with the "corner plot" above. The diagonal elements in the
matrix of plots (histograms) represent the posterior distributions for each
parameter (see label for each column in the bottom row). The off-diagonal
elements represent joint posterior distributions for each pair of model
parameters, and darker pixels represent a higher density of posterior samples.
Note for example that the 2D histogram in the second row, first column is the
same as the figure above (with its axes swapped). The corner plot is useful
for examining degeneracies between parameters, which are visible as correlations
between model parameters.

There are degeneracies between four pairs of model parameters. First, simulated
epidemics can reproduce the observed cumulative incidence on 18 Jan 2020 equally
well with small :math:`\mathcal{R}_0` and small :math:`D`, or with larger
:math:`\mathcal{R}_0` and larger :math:`D`. There is degeneracy between
:math:`\mathcal{R}_0` and the :math:`\Gamma`-function shape parameter
:math:`\alpha` the observed cumulative incidence is reproduced equally well
with :math:`\log_{10}\mathcal{R}_0 = 0.2` and :math:`\alpha=5`, or with
:math:`\log_{10}\mathcal{R}_0 = 1` and :math:`\alpha=1`. There are also
degeneracies between :math:`\mathcal{R}_0` and :math:`n`, and :math:`\alpha`
and :math:`D`.
