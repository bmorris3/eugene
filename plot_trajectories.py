import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from eugene import simulate_outbreak_structured

from grid_parallel import params

np.random.seed(2019)

fig, ax = plt.subplots(figsize=(4, 3))

R0_grid = np.linspace(0.7, 5, 10)

cmap = lambda x: plt.cm.viridis((x - R0_grid.min())/R0_grid.ptp())

for j in range(R0_grid.shape[0]):
    for i in range(20):
        parameters = dict(
            R0=R0_grid[j],
            k=1,
            n=1,
            D=10,
            gamma_shape=2,
            max_time=90,
            days_elapsed_max=52+7,
            max_cases=1e4,
            f_home = 0.8,
            people_per_household = 3.1,
            max_community_spread = 1000,
        )

        times, cumulative_incidence = simulate_outbreak_structured(**parameters)

        days_elapsed_min = params['days_elapsed_min']
        days_elapsed = parameters['days_elapsed_max']
        min_number_cases = params['min_number_cases']
        max_number_cases = params['max_number_cases']

        delta_t = (np.array(days_elapsed_min) -
                   max(days_elapsed_min))
        cases_at_measurement_times = np.interp(days_elapsed +
                                               delta_t,
                                               times, cumulative_incidence)

        accept = ((np.asarray(min_number_cases) <
                   cases_at_measurement_times) &
                  (cases_at_measurement_times <
                   np.asarray(max_number_cases))).all()
        cax = ax.semilogy(times, cumulative_incidence, '.-',
                          color=cmap(R0_grid[j]),
                          alpha=1.0 if accept else 0.25,
                          zorder=1 if not accept else 5)

plot_kwargs = dict(fmt='s', color='k', zorder=10, ecolor='k')
ax.errorbar(48, 2890, xerr=14, yerr=[[2700], [2700]], **plot_kwargs)
ax.errorbar(52, 5000, xerr=14, yerr=[[4000], [4700]], **plot_kwargs)

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