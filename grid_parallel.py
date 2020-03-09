import numpy as np
from eugene import abc, compute

params = dict(
    R0 = 2,
    k_grid = np.logspace(-1, 1, 20),

    # Maximum number of cases to run the simulation through (should be greater
    # than ``max_number_cases``)
    max_cases = 1e5,

    # Maximum number of days someone might transmit the disease
    max_time = 365,   # days

    # Number of stochastic trials to run at each grid-point
    trials = 100,

    # Initial number of index cases n (day-zero cases)
    n_min = 10,   # cases
    n_max = 50,  # cases

    # Generation interval/Gamma function shape parameter
    gamma_shape_min = 1.5,
    gamma_shape_max = 3,

    # Generation time interval D
    D_min = 3,   # days
    D_max = 11,  # days

    # Fraction of transmissions that occur at home, f_home:
    f_home_grid = np.linspace(0.5, 0.95, 20),

    # Average number of people per household
    people_per_household = 2.2,

    # Maximum super-spreading event size
    max_community_spread = 50,

    # Population size
    population = 1e5,

    # Computer parameters
    n_processes = 8,
    n_grid_points_per_process = 1,

    # Formatting string for naming simulation outputs
    samples_path = 'samples/samples{0}.npy'
)

if __name__ == '__main__':
    total_trials = (params['trials'] * params['f_home_grid'].shape[0] *
                    params['k_grid'].shape[0])
    print(f'Total number of simulations triggered: {total_trials}')

    abc(**params)
    # compute(**params)