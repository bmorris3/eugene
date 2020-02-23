***************
Getting started
***************

``eugene`` is a generalized simulator for outbreaks based on the work of
`Riou & Althaus (2020) <https://github.com/jriou/wcov>`_. The primary goal is to
probe two of the parameters that describe the epidemic of COVID-19, the basic
reproduction number :math:`\mathcal{R}_0` and the overdispersion factor
:math:`k`. We take an Approximate Bayesian Computation approach to estimating
:math:`\mathcal{R}_0` and :math:`k` by running thousands of stochastic outbreak
simulations and exploring which values of :math:`\mathcal{R}_0`, :math:`k`, and
other parameters accurately reproduce the incidence of COVID-19 on January 18,
2020.

At the core of ``eugene`` lies a simple outbreak model, which starts with
number of index cases :math:`n`. The user must also specify
:math:`\mathcal{R}_0`, :math:`k`, the generation time between incidences
:math:`D`, the shape of the Gamma distribution parameterized by parameter
``gamma_shape``, maximum number of days to simulate ``days_elapsed`` and the
maximum number of cases beyond which to stop simulating ``max_cases``. We can
specify those parameters in code like so:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(2020)

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

Now we can simulate 100 outbreaks with these initial parameters:

.. code-block:: python

    from eugene import simulate_outbreak

    fig, ax = plt.subplots(figsize=(4, 3))

    for i in range(100):
        times, cumulative_incidence = simulate_outbreak(**parameters)
        ax.semilogy(times, cumulative_incidence, '.-', color='k', alpha=0.2)

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Cumulative Incidence')
    fig.tight_layout()
    plt.show()

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    from eugene import simulate_outbreak

    parameters = dict(
        R0 = 2,
        k = 0.1,
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
