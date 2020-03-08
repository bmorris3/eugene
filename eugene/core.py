from copy import copy
from itertools import zip_longest
from concurrent import futures as cf

import numpy as np
from scipy.stats import gamma, nbinom
from numba import njit

__all__ = ['abc', 'compute', 'simulate_outbreak',
           'simulate_outbreak_structured']


@njit
def sample_nbinom(n, p, size):
    nb = np.zeros(size)
    for i in range(size):
        nb[i] = np.random.poisson(np.random.gamma(n, (1 - p) / p))
    return nb

@njit
def min_along_axis(a, b):
    mins = []
    for i, j in zip(a, b):
        mins.append(min([i, j]))
    return np.array(mins)


@njit
def max_along_axis(a, b):
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


def abc(n_processes, f_home_grid, n_grid_points_per_process, **parameters):
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


@njit
def simulate_outbreak(R0, k, n, D, gamma_shape, max_time,
                      days_elapsed_max,
                      max_cases, seed=None):
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

    seed : int

    Returns
    -------
    times : `~numpy.ndarray`
        Times of incidence measurements
    cumulative_incidence : `~numpy.ndarray`
        Cumulative incidence (total cases) at each time
    """
    if seed is not None:
        np.random.seed(seed)
    cumulative_incidence = int(n)
    t = np.zeros(n)
    cases = int(n)
    incidence = [n]
    t_mins = [0]

    while (cases > 0) and (t.min() < days_elapsed_max) and (
            cumulative_incidence < max_cases):
        secondary = sample_nbinom(n=k, p=k/(k+R0), size=cases)

        inds = np.arange(0, secondary.max())
        gamma_size = (secondary.shape[0], secondary.max())

        g = np.random.standard_gamma(D / gamma_shape, size=gamma_size)
        t_new = np.expand_dims(t, 1) + g
        mask = np.expand_dims(secondary, 1) <= inds
        times_in_bounds = ((t_new < max_time) &
                           np.logical_not(mask))
        cases = np.count_nonzero(times_in_bounds)
        cumulative_incidence += cases

        t = t_new.ravel()[times_in_bounds.ravel()].copy()
        if cases > 0:
            t_mins.append(t.min())
            incidence.append(cases)

    incidence = np.array(incidence)
    epidemic_curve = incidence.cumsum()
    t_mins = np.array(t_mins)
    return t_mins, epidemic_curve


def compute(f_home_grid, max_community_spread_grid,
            R0, k, trials, D_min, D_max, n_min, n_max, max_cases,
            gamma_shape_min, gamma_shape_max, max_time, days_elapsed_min,
            days_elapsed_max, min_number_cases, max_number_cases,
            samples_path, people_per_household, population):

    accepted_grid = []

    D_chain = []
    n_chain = []
    f_home_chain = []
    max_community_spread_chain = []
    # R0_chain = []
    # k_chain = []

    days_elapsed_chain = []
    gamma_shape_chain = []

    f_home_grid = np.array(f_home_grid)

    for i, f_home in enumerate(f_home_grid):
        accept_k = []
        for j, max_community_spread in enumerate(max_community_spread_grid):
            accepted = []
            for n in range(trials):
                D = D_min + (D_max - D_min) * np.random.rand()
                n = np.random.randint(n_min, n_max)
                gamma_shape = (gamma_shape_min + (gamma_shape_max -
                                                  gamma_shape_min) *
                               np.random.rand())
                days_elapsed = (max(days_elapsed_min) +
                                (max(days_elapsed_max) - max(days_elapsed_min)
                                 ) * np.random.rand())

                ell = k * (f_home**2 + (1 - f_home)**2)

                t_mins, cum_inc = simulate_outbreak_structured(R0, ell, n, D,
                                                               gamma_shape,
                                                               max_time,
                                                               days_elapsed,
                                                               max_cases,
                                                               f_home,
                                                               people_per_household,
                                                               max_community_spread,
                                                               population)


                if t_mins.max() > max(days_elapsed_max):
                    # Outbreak is still ongoing
                    accept = False
                elif cum_inc.max() > max_cases:
                    # Outbreak has tons of cases
                    accept = False
                else:
                    # Outbreak has terminated:
                    accept = True

                if accept:
                    D_chain.append(D)
                    n_chain.append(n)
                    f_home_chain.append(f_home)
                    max_community_spread_chain.append(max_community_spread)
                    days_elapsed_chain.append(days_elapsed)
                    gamma_shape_chain.append(gamma_shape)

            if len(accepted) > 0:
                accepted_fraction = np.count_nonzero(accepted) / len(accepted)
            else:
                accepted_fraction = 0

            accept_k.append(accepted_fraction)

        accepted_grid.append(accept_k)

    samples = np.vstack([f_home_chain, max_community_spread_chain, D_chain,
                         n_chain, days_elapsed_chain, gamma_shape_chain]).T
    np.save(samples_path.format(f_home_grid[0]), samples)


@njit
def simulate_outbreak_structured(R0, k, n, D, gamma_shape, max_time,
                                 days_elapsed_max, max_cases, f_home,
                                 people_per_household, max_community_spread,
                                 population, seed=None):
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

    f_home : float
        Fraction of cases that occur at home

    people_per_household : float
        Number of people in each household

    max_community_spread : int
        Maximum number of secondary cases from a single spreading event

    seed : int

    population : int
        Total population size

    Returns
    -------
    times : `~numpy.ndarray`
        Times of incidence measurements
    cumulative_incidence : `~numpy.ndarray`
        Cumulative incidence (total cases) at each time
    """
    if seed is not None:
        np.random.seed(seed)

    population_vector = np.zeros(int(population))
    time_vector = np.zeros(int(population))
    population_vector[:n] = 1
    time_vector[:n] = 0.01

    cumulative_incidence = int(n)
    cases = int(n)
    incidence = [n]
    t_mins = [0]
    steps = 0

    while (cases > 0) and (cumulative_incidence < max_cases):
        infected_inds = np.nonzero(time_vector)[0]
        n_cases_home = int(cases * f_home) + 1
        n_cases_comm = cases - n_cases_home

        secondary_comm = sample_nbinom(n=k, p=k/(k + R0),
                                       size=n_cases_comm)

        # impose maximum on number of secondary cases from single primary:
        secondary_comm_min = min_along_axis(secondary_comm, np.ones(cases) *
                                            int(max_community_spread))

        secondary_home = sample_nbinom(n=k, p=k/(k + R0),
                                       size=n_cases_home)

        # Draw household size from max(Poisson(3.1), 1):
        poisson_home = max_along_axis(np.random.poisson(people_per_household,
                                                        size=n_cases_home),
                                      np.ones(cases))

        secondary_home_min = min_along_axis(poisson_home, secondary_home)

        secondary = np.sum(np.concatenate((secondary_comm_min, secondary_home_min)))

        # Infect new cases
        new_infect_inds = np.random.choice(population, int(secondary),
                                           replace=False)

        # Increment time interval for existing cases

        still_infectious = new_infect_inds[(population_vector[new_infect_inds] == 1) &
                                           (time_vector[new_infect_inds] < max_time)]
        new_infections = new_infect_inds[(population_vector[new_infect_inds] == 0)]

        g1 = np.random.standard_gamma(D / gamma_shape, size=len(new_infections))

        # Increment time interval for new cases
        g2 = np.random.standard_gamma(D / gamma_shape, size=len(still_infectious))

        population_vector[new_infections] = 1

        time_vector[new_infections] += g1
        time_vector[still_infectious] = min_along_axis(time_vector[still_infectious] + g2,
                                                       max_time * np.ones(len(still_infectious)))

        if np.count_nonzero(population_vector) < population:
            cumulative_incidence = np.count_nonzero(population_vector)

            if cases > 0:
                t_mins.append(steps)
                incidence.append(cumulative_incidence)
                steps += 1
                cases = len(new_infections)
        else:
            break

    incidence = np.array(incidence)
    return np.arange(steps + 1), incidence, time_vector, population_vector
