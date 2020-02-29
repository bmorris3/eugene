import os
import numpy as np
import pytest

from ..core import sample_nbinom

control_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            os.pardir, 'data',
                                            'control.txt'))


@pytest.mark.parametrize("n, p", ((10, 0.5), (15, 0.1), (20, 0.01)))
def test_nbinom(n, p):
    np.random.seed(1)
    nb_np = np.random.negative_binomial(n, p, size=10000)
    nb_eu = sample_nbinom(n, p, size=10000)

    assert (abs(nb_np.mean() - nb_eu.mean()) < 0.01 * n * (1 - p) / p)
    assert (abs(nb_np.std() - nb_eu.std()) < 0.01 * n * (1 - p) / p)
