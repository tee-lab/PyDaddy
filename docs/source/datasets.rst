Sample Datasets
===============

PyDaddy comes pre-packaged with several sample datasets. These can be loaded easily using the :code:`load_sample_dataset()` function.

::

    pydaddy.load_sample_dataset(<dataset-name>)

The following sample datasets are available:

    - :code:`fish-data-etroplus`: Group polarization data from a fish schooling experiment [1]_.
    - :code:`cell-data-cellhopping`: Dataset from a confined cell migration experiment [2]_
    - :code:`model-data-scalar-pairwise` and :code:`model-data-scalar-ternary`: Scalar (1-D) simulated datasets generated from a stochastic Gillespie simulation, with pairwise and ternary interaction models respectively [1]_ [3]_.
    - :code:`model-data-vector-pairwise` and :code:`model-data-vector-ternary`: Vector (2-D) simulated datasets generated from a stochastic Gillespie simulation, with pairwise and ternary interaction models respectively [1]_ [3]_.

The fish schooling dataset contains the time series of the group polarization vector $\mathbf m$ (2-dimensional), for a group of 15 fish (\emph{Etroplus suratensis}). The polarization time series is available at a uniform interval of 0.12 second. The dataset contains many missing data points [1]_.

The simulated datasets were generated using a continuous-time stochastic simulation algorithm, with pairwise and ternary interaction models respectively. Each simulated time series was resampled to a suitable uniform sampling interval [1]_ [2]_. Simulated datasets are provided for both 1-D and 2-D.


.. [1] Jhawar, J., Morris, R. G., Amith-Kumar, U. R., Danny Raj, M., Rogers, T., Rajendran, H., & Guttal, V. (2020). Noise-induced schooling of fish. Nature Physics, 16(4), 488-493 (`doi <https://doi.org/10.1038/s41567-020-0787-y>`_).
.. [2] Brückner, D. B., Fink, A., Schreiber, C. et al. Stochastic nonlinear dynamics of confined cell migration in two-state systems. Nat. Phys. 15, 595–601 (2019) (`doi <https://doi.org/10.1038/s41567-019-0445-4>`_).
.. [3] Jhawar, J., & Guttal, V. (2020). Noise-induced effects in collective dynamics and inferring local interactions from data. Philosophical Transactions of the Royal Society B, 375(1807), 20190381. (`doi <http://dx.doi.org/10.1098/rstb.2019.0381>`_)
