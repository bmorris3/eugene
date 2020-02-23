from copy import copy
from itertools import zip_longest
from concurrent import futures as cf
import numpy as np
from scipy.stats import gamma, nbinom

__all__ = ['abc']


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks.

    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"

    Source: https://docs.python.org/3/library/itertools.html#recipes
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def abc(n_processes, R0_grid, n_grid_points_per_process, **parameters):
    # https://stackoverflow.com/a/15143994
    executor = cf.ProcessPoolExecutor(max_workers=n_processes)
    futures = [executor.submit(compute, group, **parameters)
               for group in grouper(R0_grid, n_grid_points_per_process)]
    cf.wait(futures)


def compute(R0_grid, k_grid, trials, D_min, D_max, n_min, n_max, max_cases,
            gamma_shape_min, gamma_shape_max, max_time, days_elapsed_min,
            days_elapsed_max, min_number_cases, max_number_cases,
            samples_path):

    accepted_grid = []

    D_chain = []
    n_chain = []
    R0_chain = []
    k_chain = []
    days_elapsed_chain = []
    gamma_shape_chain = []

    R0_grid = np.array(R0_grid)

    for i, R0 in enumerate(R0_grid):
        accept_k = []
        for j, k in enumerate(k_grid):
            accepted = []
            for n in range(trials):
                D = D_min + (D_max - D_min) * np.random.rand()
                n = np.random.randint(n_min, n_max)
                gamma_shape = (gamma_shape_min + (gamma_shape_max -
                                                  gamma_shape_min) *
                               np.random.rand())
                days_elapsed = (days_elapsed_min + (days_elapsed_max -
                                                    days_elapsed_min) *
                                np.random.rand())
                times = [0]
                times_counter = 1
                t = copy(times)
                cases = copy(n)
                incidence = [1]
                t_maxes = [0]

                while (cases > 0) and (times_counter < max_cases):
                    secondary = nbinom.rvs(n=k, p=k / (k + R0), size=cases)

                    # Vectorized approach (optimized for speed in Python)
                    inds = np.arange(0, secondary.max())
                    gamma_size = (secondary.max(), secondary.shape[0])
                    t_new = np.ma.array(t + gamma.rvs(D / gamma_shape,
                                                      size=gamma_size),
                                        mask=secondary[:, None] <= inds)
                    times_in_bounds = ((t_new.data < max_time) &
                                       np.logical_not(t_new.mask))
                    # times.extend(t_new[times_in_bounds].tolist())
                    cases = np.count_nonzero(times_in_bounds)
                    times_counter += cases
                    t = t_new[times_in_bounds].copy()
                    incidence.append(cases)
                    t_maxes.append(t_new.mean())

                # times = np.array(times)
                # total_incidence = len(times)
                incidence = np.array(incidence)
                cum_inc = incidence.cumsum()
                t_maxes = np.array(t_maxes)

                if t_maxes.max() >= days_elapsed:
                    terminal_cum_inc = 10**np.interp(days_elapsed, t_maxes,
                                                     np.log10(cum_inc))

                    accept = (min_number_cases < terminal_cum_inc <
                              max_number_cases)
                    accepted.append(accept)

                    if accept:
                        D_chain.append(D)
                        n_chain.append(n)
                        R0_chain.append(R0)
                        k_chain.append(k)
                        days_elapsed_chain.append(days_elapsed)
                        gamma_shape_chain.append(gamma_shape)

            if len(accepted) > 0:
                accepted_fraction = np.count_nonzero(accepted) / len(accepted)
            else:
                accepted_fraction = 0

            accept_k.append(accepted_fraction)

        accepted_grid.append(accept_k)

    samples = np.vstack([R0_chain, k_chain, D_chain, n_chain,
                         days_elapsed_chain, gamma_shape_chain]).T
    np.save(samples_path.format(R0_grid[0]), samples)