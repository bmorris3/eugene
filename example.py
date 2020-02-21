import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gamma, nbinom

gamma_shape = 2
it_sigma = 10  # 7->14
R0 = 2.0
k = 1
cases = 100

bin = nbinom.rvs(n=k, p=k/(k + R0), size=100)

# plt.hist(gamma.rvs(gamma_shape, size=1000))
plt.hist(bin)
plt.show()