Documentation
=============

``eugene`` is a generalized simulator for outbreaks based on the work of
`Riou & Althaus (2020) <https://github.com/jriou/wcov>`_. The primary goal is to
probe two of the parameters that describe the epidemic of COVID-19, the basic
reproduction number :math:`\mathcal{R}_0` and the overdispersion factor
:math:`k`. We take an Approximate Bayesian Computation approach to estimating
:math:`\mathcal{R}_0` and :math:`k` by running thousands of stochastic outbreak
simulations and exploring which values of :math:`\mathcal{R}_0`, :math:`k`, and
other parameters accurately reproduce the incidence of COVID-19 on January 18,
2020.

.. toctree::
  :maxdepth: 2

  eugene/installation.rst
  eugene/gettingstarted.rst
  eugene/index.rst
