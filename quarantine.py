import os
import numpy as np

from eugene import abc, quarantine_fraction

params = dict(
    # Grid of R0 and k parameters to iterate over
    R0_grid = np.logspace(np.log10(0.7), np.log10(10), 50),
    k_grid = np.logspace(-2, 1, 10),

    # Maximum number of cases to run the simulation through (should be greater
    # than ``max_number_cases``)
    max_cases = 1e4,

    # Maximum number of days someone might transmit the disease
    max_time = 365,   # days

    # Number of stochastic trials to run at each grid-point
    trials = 10,

    # Days elapsed since zoonotic transmission
    days_elapsed_min = [365],  # days
    days_elapsed_max = [365],  # days

    # Number of cases after ``days_elapsed``
    min_number_cases = [1000],  # cases
    max_number_cases = [1000],  # cases

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
)

# Fraction of cases perfectly quarantined
quarantine_fracs = [0.1, 0.3, 0.5, 0.7, 0.9]

if __name__ == '__main__':
    total_trials = (params['trials'] * params['R0_grid'].shape[0] *
                    params['k_grid'].shape[0] * len(quarantine_fracs))
    print(f'Total number of simulations triggered: {total_trials}')

    for f_Q in quarantine_fracs:
        print(f'Running f_Q={f_Q}...')
        newdir = 'quarantine_{0:d}'.format(int(f_Q * 10))
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        samples_path = os.path.join(newdir, 'samples{0}.npy')
        abc(quarantine_fraction, f_Q=f_Q, samples_path=samples_path, **params)
