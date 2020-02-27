import os
import numpy as np

from ..core import simulate_outbreak

control_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            os.pardir, 'data',
                                            'control.txt'))

seed = 2020


def test_simulate_outbreak():
    control_times, control_inc = np.loadtxt(control_path, unpack=True)

    times, inc = simulate_outbreak(R0=1.5, k=1, n=1, D=10, gamma_shape=2,
                                   max_time=90, days_elapsed_max=50,
                                   max_cases=1e4, seed=seed)

    np.testing.assert_allclose(times, control_times)
    np.testing.assert_allclose(control_inc, inc)


def print_result():
    """
    python -c "from eugene.tests.test_core import print_result as p; p()"
    """
    times, inc = simulate_outbreak(R0=1.5, k=1, n=1, D=10, gamma_shape=2,
                                   max_time=90, days_elapsed_max=50,
                                   max_cases=1e4, seed=seed)

    # np.savetxt(control_path, np.vstack([times, inc]).T)

    print(times, inc)