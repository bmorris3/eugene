import os
import numpy as np

from ..core import simulate_outbreak


def test_simulate_outbreak():
    control_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, 'data',
                                                'control.txt'))
    control_times, control_inc = np.loadtxt(control_path, unpack=True)

    np.random.seed(2020)

    times, inc = simulate_outbreak(R0=1.5, k=1, n=1, D=10, gamma_shape=2,
                                   max_time=90, days_elapsed_max=50,
                                   max_cases=1e4)

    np.testing.assert_allclose(times, control_times)
    np.testing.assert_allclose(control_inc, inc)
