import numpy as np
from eugene import abc

params = dict(
    # Grid of R0 and k parameters to iterate over
    R0_grid = np.logspace(np.log10(0.7), np.log10(10), 50),
    k_grid = np.logspace(-2, 1, 30),

    # Maximum number of cases to run the simulation through (should be greater
    # than ``max_number_cases``)
    max_cases = 1e4,

    # Maximum number of days someone might transmit the disease
    max_time = 90,   # days

    # Number of stochastic trials to run at each grid-point
    trials = 1000,

    # Days elapsed since zoonotic transmission
    days_elapsed_min = [46-7, 52-7],  # days
    days_elapsed_max = [46+7, 52+7],  # days

    # Number of cases after ``days_elapsed``
    min_number_cases = [190, 1000],  # cases
    max_number_cases = [5590, 9700],  # cases

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
    n_processes = 8,
    n_grid_points_per_process = 2,

    # Formatting string for naming simulation outputs
    samples_path = 'samples/samples{0}.npy'
)

if __name__ == '__main__':
    total_trials = (params['trials'] * params['R0_grid'].shape[0] *
                    params['k_grid'].shape[0])
    print(f'Total number of simulations triggered: {total_trials}')

    abc(**params)
