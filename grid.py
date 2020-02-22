import numpy as np
from copy import copy
from scipy.stats import gamma, nbinom
from tqdm import tqdm

gamma_shape = 2
# R0_grid = np.arange(2, 5, 2)
# k_grid = np.logspace(-2, 1, 2)

R0_grid = np.arange(0.8, 5, 0.2)
k_grid = np.logspace(-2, 1, 10)
# seed = 1
max_cases = 1e4
max_time = 90
n_trials = 50
accepted_grid = []

sigma_chain = []
seed_chain = []
R0_chain = []
k_chain = []
accepts = []

for i, R0 in tqdm(enumerate(R0_grid), total=R0_grid.shape[0]):
    accept_k = []
    for j, k in enumerate(k_grid):
        accepted = []
        for n in range(n_trials):
            it_sigma = 7 + 7 * np.random.rand()
            seed = np.random.randint(1, 51)
            times = [0]
            t = copy(times)
            cases = copy(seed)
            incidence = [1]
            t_maxes = [0]

            while (cases > 0) and (len(times) < max_cases):
                secondary = nbinom.rvs(n=k, p=k / (k + R0), size=cases)

                # Vectorized approach (optimized for speed in Python)
                inds = np.arange(0, secondary.max())
                gamma_size = (secondary.max(), secondary.shape[0])
                t_new = np.ma.array(t + gamma.rvs(it_sigma / gamma_shape,
                                                  size=gamma_size),
                                    mask=secondary[:, None] <= inds)
                times_in_bounds = ((t_new.data < max_time) &
                                   np.logical_not(t_new.mask))
                times.extend(t_new[times_in_bounds].tolist())
                cases = np.count_nonzero(times_in_bounds)
                t = t_new[times_in_bounds].copy()
                incidence.append(cases)
                t_maxes.append(t_new.mean())

            times = np.array(times)
            incidence = np.array(incidence)
            total_incidence = len(times)
            cum_inc = incidence.cumsum()
            t_maxes = np.array(t_maxes)

            if t_maxes.max() >= 52:
                terminal_cum_inc = 10**np.interp(52, t_maxes, np.log10(cum_inc))

                accept = 1000 < terminal_cum_inc < 9700
                accepted.append(accept)

                if accept:
                    sigma_chain.append(it_sigma)
                    seed_chain.append(seed)
                    R0_chain.append(R0)
                    k_chain.append(k)
        if len(accepted) > 0:
            accepted_fraction = np.count_nonzero(accepted) / len(accepted)
            # print('acceptance fraction:', accepted_fraction)
        else:
            accepted_fraction = 0
        accept_k.append(accepted_fraction)
    accepted_grid.append(accept_k)

np.save('results.npy', accepted_grid)
samples = np.vstack([R0_chain, k_chain, sigma_chain, seed_chain]).T
np.save('samples.npy', samples)