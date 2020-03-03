import numpy as np
from eugene import abc, compute

params = dict(
    # Grid of R0 and k parameters to iterate over
    R0_grid = np.logspace(np.log10(0.7), np.log10(20), 100),
    k_grid = np.logspace(-2, 3, 50),

    # Maximum number of cases to run the simulation through (should be greater
    # than ``max_number_cases``)
    max_cases = 100,

    # Maximum number of days someone might transmit the disease
    max_time = 10,   # days

    # Number of stochastic trials to run at each grid-point
    trials = 50000,

    # Days elapsed since zoonotic transmission
    days_elapsed_min = [0, 2-1, 4-1, 5-1, 6-1],  # days
    days_elapsed_max = [0, 2+1, 4+1, 5+1, 6+1],  # days

    # Number of cases after ``days_elapsed``
    min_number_cases = [0, 8-3, 18-5, 27-10, 42-20],  # cases
    max_number_cases = [2, 8+3, 18+5, 27+10, 42+20],  # cases

    # Initial number of index cases n (day-zero cases)
    n_min = 1,   # cases
    n_max = 5,  # cases

    # Generation interval/Gamma function shape parameter
    gamma_shape_min = 1,
    gamma_shape_max = 4,

    # Generation time interval D
    D_min = 4,   # days
    D_max = 11,  # days

    # Computer parameters
    n_processes = 16,
    n_grid_points_per_process = 1,

    # Formatting string for naming simulation outputs
    samples_path = 'samples/samples{0}.npy'
)

if __name__ == '__main__':
    total_trials = (params['trials'] * params['R0_grid'].shape[0] *
                    params['k_grid'].shape[0])
    print(f'Total number of simulations triggered: {total_trials}')

    abc(**params)
    # compute(**params)