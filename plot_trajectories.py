import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from eugene import simulate_outbreak

from grid_parallel import params

np.random.seed(2019)

fig, ax = plt.subplots(figsize=(4, 3))

R0_grid = np.linspace(0.5, 3, 10)

cmap = lambda x: plt.cm.viridis((x - R0_grid.min())/R0_grid.ptp())

for j in range(R0_grid.shape[0]):
    for i in range(100):
        parameters = dict(
            R0=R0_grid[j],
            k=100,
            n=1,
            D=7,
            gamma_shape=2,
            max_time=10,
            days_elapsed_max=max(params['days_elapsed_max']),
            max_cases=200
        )

        times, cumulative_incidence = simulate_outbreak(**parameters)

        days_elapsed_min = params['days_elapsed_min']
        days_elapsed_max = params['days_elapsed_max']
        min_number_cases = params['min_number_cases']
        max_number_cases = params['max_number_cases']

        delta_t = 0#(np.mean([days_elapsed_min, days_elapsed_max], axis=0) -
                  # max(days_elapsed_max))

        cases_at_measurement_times = 10**np.interp(days_elapsed_max,
                                                   times,
                                                   np.log10(cumulative_incidence))

        accept = ((np.asarray(min_number_cases) <
                   cases_at_measurement_times) &
                  (cases_at_measurement_times <
                   np.asarray(max_number_cases))).all()

        cax = ax.semilogy(times, cumulative_incidence, '.-',
                          color=cmap(R0_grid[j]),
                          alpha=1.0 if accept else 0.25,
                          zorder=1 if not accept else 5)

cases_mean = np.mean([params['min_number_cases'],
                     params['max_number_cases']], axis=0)
cases_max = np.asarray(params['max_number_cases'])
cases_min = np.asarray(params['max_number_cases'])
plot_kwargs = dict(fmt='s', color='k', zorder=10, ecolor='k')
ax.errorbar(np.mean([params['days_elapsed_min'],
                     params['days_elapsed_max']], axis=0),
            cases_mean,
            xerr=1, yerr=[cases_mean-cases_min,
                          cases_max-cases_mean], **plot_kwargs)

norm = Normalize(vmin=R0_grid.min(), vmax=R0_grid.max())
cbar = plt.colorbar(mappable=ScalarMappable(norm=norm, cmap=plt.cm.viridis),
                    label='$\mathcal{R}_0$')

ax.set_xticks(np.arange(0, 10, 1), minor=True)

for sp in ['right', 'top']:
    ax.spines[sp].set_visible(False)

ax.set_xlabel('Time [days]')
ax.set_ylabel('Cumulative Incidence')
fig.tight_layout()
fig.savefig('plots/trajectories.pdf', bbox_inches='tight')
plt.show()