import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from scipy.stats import gamma, nbinom
from time import time as systime

start = systime()
gamma_shape = 2
it_sigma = 10  # 7->14
R0 = 2
k = 1
seed = 1
max_cases = 5e4

max_time = 90

n_trials = 100
for n in range(n_trials):
    times = [0]
    t = copy(times)
    cases = np.copy(seed)
    incidence = []

    while (cases > 0) and (len(times) < max_cases):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
        secondary = nbinom.rvs(n=k, p=k / (k + R0), size=cases)

        # Vectorized approach (optimized for speed in Python)
        inds = np.arange(0, secondary.max())
        t_new = np.ma.array(t + gamma.rvs(it_sigma / gamma_shape,
                                          size=(
                                          secondary.max(), secondary.shape[0])),
                            mask=secondary[:, None] <= inds)
        times_in_bounds = (t_new.data < max_time) & np.logical_not(t_new.mask)
        times.extend(t_new[times_in_bounds].tolist())
        cases = np.count_nonzero(times_in_bounds)
        t = copy(t_new[times_in_bounds])
        incidence.append(cases)

    times = np.array(times)
    incidence = np.array(incidence)
    total_incidence = len(times)

    plt.semilogy(incidence.cumsum(), color='k', alpha=0.5)
#     plt.plot(incidence.cumsum())

end = systime()

plt.xlabel('transmission branch depth')
plt.ylabel('cumulative incidence')
plt.savefig('plots/test.pdf', bbox_inches='tight')
plt.show()