import numpy as np
from eugene import abc

params = dict(
    # Grid of R0 and k parameters to iterate over
    R0_grid = np.logspace(np.log10(0.7), np.log10(20), 15),
    k_grid = np.logspace(-2, 1, 15),

    # Maximum number of cases to run the simulation through (should be greater
    # than ``max_number_cases``)
    max_cases = 1e4,

    # Maximum number of days someone might transmit the disease
    max_time = 90,   # days

    # Number of stochastic trials to run at each grid-point
    n_trials = 1000,

    # Number of cases on January 18, 2020
    min_number_cases = 1000,  # cases
    max_number_cases = 9700,  # cases

    # Initial number of index cases n (day-zero cases)
    n_min = 1,   # cases
    n_max = 51,  # cases

    # Days elapsed since zoonotic transmission
    days_elapsed = 52,  # days

    # Generation interval/Gamma function shape parameter
    gamma_shape = 2,

    # Generation time interval D
    D_min = 7,   # days
    D_max = 60,  # days

    # Computer parameters
    n_processes = 8,
    n_grid_points_per_process = 2,

    # Formatting string for naming simulation outputs
    samples_path = 'samples/samples{0}.npy'
)

total_trials = (params['n_trials'] * params['R0_grid'].shape[0] *
                params['k_grid'].shape[0])
print(f'Total number of simulations triggered: {total_trials}')

abc(**params)
