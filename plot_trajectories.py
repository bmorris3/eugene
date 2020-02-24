import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from eugene import simulate_outbreak

np.random.seed(2019)

fig, ax = plt.subplots(figsize=(4, 3))

R0_grid = np.linspace(0.7, 3, 10)

cmap = lambda x: plt.cm.viridis((x - R0_grid.min())/R0_grid.ptp())

for j in range(R0_grid.shape[0]):
    for i in range(10):
        parameters = dict(
            R0=R0_grid[j],
            k=1,
            n=1,
            D=10,
            gamma_shape=2,
            max_time=90,
            days_elapsed_max=52+7,
            max_cases=1e4
        )

        times, cumulative_incidence = simulate_outbreak(**parameters)

        inc_at_t = 10**np.interp([52-7, 52+7], times,
                                 np.log10(cumulative_incidence))

        cax = ax.semilogy(times, cumulative_incidence, '.-',
                          color=cmap(R0_grid[j]),
                          alpha=1.0 if (1000 < inc_at_t).any() &
                                       (inc_at_t < 9700).any() else 0.3)

ax.errorbar(52, 5000, xerr=14, yerr=[[4000], [4700]], fmt='s', color='k')

norm = Normalize(vmin=R0_grid.min(), vmax=R0_grid.max())
cbar = plt.colorbar(mappable=ScalarMappable(norm=norm, cmap=plt.cm.viridis),
                    label='$\mathcal{R}_0$')

ax.set_xticks(np.arange(0, 60, 1), minor=True)

for sp in ['right', 'top']:
    ax.spines[sp].set_visible(False)

ax.set_xlabel('Time [days]')
ax.set_ylabel('Cumulative Incidence')
fig.tight_layout()
fig.savefig('plots/trajectories.pdf', bbox_inches='tight')
plt.show()