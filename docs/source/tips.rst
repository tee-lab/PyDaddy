Usage Tips
==========

* While working with 1-D data, if your data is a 1-D array, remember to wrap the array in a list while passing to PyDaddy, like so: :code:`pydaddy.Characterize([x], ...`. Otherwise, PyDaddy will throw an error.

* PyDaddy expects uniformly sampled time-series. If your dataset is sampled with irregular time intervals, resample the time-series to a uniform sampling interval before using PyDaddy (a library like `traces <http://traces.readthedocs.io/>`_ will be useful for this).

* There is a :code:`simulate()` function (:meth:`pydaddy.daddy.Daddy.simulate`) provided, that can generate simulated time series using the SDE estimated by PyDaddy. If you need to do advanced testing or diagnostics using simulated data, use this function.

* When necessary, you can 'hack' the :code:`simulate()` (:meth:`pydaddy.daddy.Daddy.simulate`) function to use custom drift and diffusion functions, not necessarily the results of the fits. To do this, assign appropriate functions to :code:`ddsde.F` and :code:`ddsde.G` (:code:`ddsde.F1`, :code:`ddsde.F2`, :code:`ddsde.G11`, :code:`ddsde.G12`, :code:`ddsde.G22` for vector).

* The :code:`fit()` (:meth:`pydaddy.daddy.Daddy.fit`) function has an :code:`alpha` parameter, which is a ridge regularization parameter. This is useful when the data is noisy or has outliers. If you think :code:`fit()` is tending to overfit the data, non-zero for :code:`alpha` and see if the fits improve (very high values, as high as 10e6 or 10e7 may be often required to see noticable effects). Be aware of the fact that large values of alpha has a side-effect of shrinking the estimated parameters.

* If :code:`noise_diagnostics()` (:meth:`pydaddy.daddy.Daddy.noise_diagnostics`)suggests that the noise autocorrelation is too high, a straightforward way around this problem is to subsample the data until the noise-correlation goes away. PyDaddy provides an easy way to do this: initialize :code:`Characterize()` (:class:`pydaddy.characterize.Characterize`) with parameters :code:`Dt=T, dt=T` where :code:`T` is the autocorrelation time (in number of time-steps) rounded to the nearest integer. However, note that using larger values of :code:`T` can distort the estimated drift and diffusion functions. Specifically, when the sampling time is too high, the estimated drift will be linear and the estimated diffusion will be quadratic (regardless the shape of the actual drift and diffusion functions).

* If the estimated drift is linear and the estimated diffusion is quadratic, the analysis results may not be reliable, and additional checks may be required: as mentioned above, these results can appear spuriously when the sampling interval is high [1]_.

.. [1] Riera, R., & Anteneodo, C. (2010). Validation of drift and diffusion coefficients from experimental data. Journal of Statistical Mechanics: Theory and Experiment, 2010(04), P04020.