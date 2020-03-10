from copy import copy
from itertools import zip_longest
from concurrent import futures as cf

import numpy as np
from numba import njit

__all__ = ['batch', 'compute', 'simulate_outbreak_structured']


@njit
def sample_nbinom(n, p, size):
    """
    Draw random samples from the negative binomial distribution.

    Parameters
    ----------
    n : float
        Number of trials
    p : float
        Probability of success on (0, 1)
    size : int
        Number of samples to draw from negative binomial distribution

    Returns
    -------
    nb : `~numpy.ndarray`
        Draws from the negative binomial distribution
    """
    nb = np.zeros(size)
    for i in range(size):
        nb[i] = np.random.poisson(np.random.gamma(n, (1 - p) / p))
    return nb

@njit
def min_along_axis(a, b):
    """
    Take the minimum along the first axis of each array ``a`` and ``b``.
    """
    mins = []
    for i, j in zip(a, b):
        mins.append(min([i, j]))
    return np.array(mins)


@njit
def max_along_axis(a, b):
    """
    Take the maximum along the first axis of each array ``a`` and ``b``.
    """
    maxes = []
    for i, j in zip(a, b):
        maxes.append(max([i, j]))
    return np.array(maxes)


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks.

    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"

    Source: https://docs.python.org/3/library/itertools.html#recipes
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def batch(n_processes, f_home_grid, n_grid_points_per_process, **parameters):
    """
    Run the `~eugene.compute` function in parallel.

    Parameters
    ----------
    n_processes : int
        Number of parallel processes to initialize
    f_home_grid : `~numpy.ndarray`
        Grid of ``f_home`` values to iterate over
    n_grid_points_per_process : int
        Number of ``f_home`` grid points to compute for each process
    parameters : dict
        Remaining parameters to pass to the `~eugene.compute` function
    """
    # https://stackoverflow.com/a/15143994
    executor = cf.ProcessPoolExecutor(max_workers=n_processes)
    futures = [executor.submit(compute, group, **parameters)
               for group in grouper(f_home_grid, n_grid_points_per_process)]
    cf.wait(futures)


def simulate_outbreak_slow(R0, k, n, D, gamma_shape, max_time, days_elapsed_max,
                           max_cases):
    """
    Simulate an outbreak.

    Parameters
    ----------
    R0 : float
    k : float
    n : float
    D : float
    gamma_shape : float
    max_time : float
    days_elapsed_max : float
    max_cases : float

    Returns
    -------
    times : `~numpy.ndarray`
        Times of incidence measurements
    cumulative_incidence : `~numpy.ndarray`
        Cumulative incidence (total cases) at each time
    """
    from scipy.stats import gamma, nbinom

    times = n * [0]
    cumulative_incidence = copy(n)
    t = np.array(times)
    cases = copy(n)
    incidence = [n]
    t_mins = [0]

    while (cases > 0) and (t.min() < days_elapsed_max) and (
            cumulative_incidence < max_cases):
        secondary = nbinom.rvs(n=k, p=k / (k + R0), size=cases)

        # Vectorized approach (optimized for speed in Python)
        inds = np.arange(0, secondary.max())
        gamma_size = (secondary.shape[0], secondary.max())
        t_new = np.ma.array(t[:, None] + gamma.rvs(D / gamma_shape,
                                                   size=gamma_size),
                            mask=secondary[:, None] <= inds)

        times_in_bounds = ((t_new.data < max_time) &
                           np.logical_not(t_new.mask))
        cases = np.count_nonzero(times_in_bounds)
        cumulative_incidence += cases
        t = t_new[times_in_bounds].copy()
        if cases > 0:
            t_mins.append(t.min())
            incidence.append(cases)

    incidence = np.array(incidence)
    epidemic_curve = incidence.cumsum()
    t_mins = np.array(t_mins)
    return t_mins, epidemic_curve


def compute(f_home_grid, max_community_spread,
            R0, k_grid, trials, D_min, D_max, n_min, n_max, max_cases, max_time,
            samples_path, people_per_household, population, **kwargs):
    """
    Compute the final size of a grid of epidemics.

    The grid is defined by ``f_home_grid`` and ``k_grid``. This function is
    designed to be called by the `~eugene.abc` function.
    """
    final_size = []

    f_home_grid = np.array(f_home_grid)

    for i, f_home in enumerate(f_home_grid):
        final_size_i = []
        for j, k in enumerate(k_grid):
            final_size_j = []
            for n in range(trials):
                D = D_min + (D_max - D_min) * np.random.rand()
                n = np.random.randint(n_min, n_max)

                ell = k * (f_home**2 + (1 - f_home)**2)

                args = (R0, ell, n, D, max_time,
                        max_cases, f_home, people_per_household,
                        max_community_spread, population)
                t_mins, cum_inc, t, p = simulate_outbreak_structured(*args)

                final_size_j.append(cum_inc[-1]/population)
            final_size_i.append(final_size_j)
        final_size.append(final_size_i)
    np.save(samples_path.format(f_home_grid[0]), final_size)


@njit
def simulate_outbreak_structured(R0, k, n, D, max_time,
                                 max_cases, f_home,
                                 people_per_household, max_community_spread,
                                 population, seed=None):
    """
    Simulate an outbreak.

    Parameters
    ----------
    R0 : float
        Reproduction number
    k : float
        Overdispersion factor
    n : float
        Number of index cases
    D : float
        Generation time interval (Gamma distribution shape parameter)
    max_time : float
        Maximum infectious time
    max_cases : float
        Maximum number of cases (usually equal to ``population``)
    f_home : float
        Fraction of cases that occur at home
    people_per_household : float
        Number of people in each household
    max_community_spread : int
        Maximum number of secondary cases from a single spreading event
    population : int
        Total population size
    seed : int (optional)
        Random seed

    Returns
    -------
    times : `~numpy.ndarray`
        Times of incidence measurements
    cumulative_incidence : `~numpy.ndarray`
        Cumulative incidence (total cases) at each time
    """
    if seed is not None:
        np.random.seed(seed)

    # `population_vector` is a binary vector representing the infectious state
    # of every person in the population (0=healthy, 1=infected)
    population_vector = np.zeros(int(population))

    # `time_vector` tracks how long a person is infections, and only allows them
    # to spread the virus if the time elapsed since their infection is less than
    # `max_time`
    time_vector = np.zeros(int(population))

    # Seed population with `n` index cases
    population_vector[:n] = 1
    time_vector[:n] = 0.01

    cumulative_incidence = int(n)
    cases = int(n)
    incidence = [n]
    t_mins = [0]
    steps = 0

    while (cases > 0) and (cumulative_incidence < max_cases):
        n_cases_home = int(cases * f_home) + 1
        n_cases_comm = cases - n_cases_home

        secondary_comm = sample_nbinom(n=k, p=k/(k + R0),
                                       size=n_cases_comm)

        # impose maximum on number of secondary cases from single primary:
        secondary_comm_min = min_along_axis(secondary_comm, np.ones(cases) *
                                            int(max_community_spread))

        secondary_home = sample_nbinom(n=k, p=k/(k + R0),
                                       size=n_cases_home)

        # Draw household size from max(Poisson(lambda), 1):
        poisson_home = max_along_axis(np.random.poisson(people_per_household,
                                                        size=n_cases_home),
                                      np.ones(cases))

        secondary_home_min = min_along_axis(poisson_home, secondary_home)

        secondary = np.sum(secondary_comm_min) + np.sum(secondary_home_min)

        # Infect new cases by randomly drawing from indices from population
        new_infect_inds = np.random.choice(int(population), int(secondary),
                                           replace=False)

        # If already infected and generation time < `max_time`, add to
        # `still_infectious` index array
        still_infections_inds = ((population_vector[new_infect_inds] == 1) &
                                 (time_vector[new_infect_inds] < max_time))
        still_infectious = new_infect_inds[still_infections_inds]

        # If newly infected, add to `new_infections` index array
        new_infections_inds = (population_vector[new_infect_inds] == 0)
        new_infections = new_infect_inds[new_infections_inds]

        # Compute generation time interval for the new infections and the
        # already infected
        g1 = np.random.standard_gamma(D, size=len(new_infections))
        g2 = np.random.standard_gamma(D, size=len(still_infectious))

        # Infect the newly infected
        population_vector[new_infections] = 1

        # Increment time interval for new cases
        time_vector[new_infections] += g1

        # Increment time interval for existing cases
        time_min_args = (time_vector[still_infectious] + g2,
                         max_time * np.ones(len(still_infectious)))
        time_vector[still_infectious] = min_along_axis(*time_min_args)

        # Cumulative incidence is the number of ones in `population_vector`
        cumulative_incidence = np.count_nonzero(population_vector)

        if cases > 0:
            t_mins.append(steps)
            incidence.append(cumulative_incidence)
            steps += 1
            cases = len(new_infections)

    incidence = np.array(incidence)
    return np.arange(steps + 1), incidence, time_vector, population_vector
