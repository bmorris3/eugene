import numpy as np
import matplotlib.pyplot as plt

from eugene import simulate_outbreak

parameters = dict(
    R0 = 2,
    k = 1,
    n = 1,
    D = 10,
    gamma_shape = 2,
    max_time = 90,
    days_elapsed = 52,
    max_cases = 1e4
)

np.random.seed(2020)

fig, ax = plt.subplots(figsize=(4, 3))

for i in range(100):
    times, cumulative_incidence = simulate_outbreak(**parameters)
    ax.semilogy(times, cumulative_incidence, '.-', color='k', alpha=0.2)

ax.set_xlabel('Time [days]')
ax.set_ylabel('Cumulative Incidence')
fig.tight_layout()
plt.show()