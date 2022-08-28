Usage Tips
==========

* While working with 1-D data, if your data is a 1-D array, remember to wrap the array in a list while passing to PyDaddy, like so: :code:`pydaddy.Characterize([x], ...`. Otherwise, PyDaddy will throw an error.

* PyDaddy expects uniformly sampled time-series. If your dataset is sampled with irregular time intervals, resample the time-series to a uniform sampling interval before using PyDaddy (a library like `traces <http://traces.readthedocs.io/>`_ will be useful for this).

* There is a :code:`simulate()` function (:meth:`pydaddy.daddy.Daddy.simulate`) provided, that can generate simulated time series using the SDE estimated by PyDaddy. If you need to do advanced testing or diagnostics using simulated data, use this function.

* When necessary, you can 'hack' the :code:`simulate()` (:meth:`pydaddy.daddy.Daddy.simulate`) function to use custom drift and diffusion functions, not necessarily the results of the fits. To do this, assign appropriate functions to :code:`ddsde.F` and :code:`ddsde.G` (:code:`ddsde.F1`, :code:`ddsde.F2`, :code:`ddsde.G11`, :code:`ddsde.G12`, :code:`ddsde.G22` for vector).

* If :code:`noise_diagnostics()` (:meth:`pydaddy.daddy.Daddy.noise_diagnoistcs`)suggests that the noise autocorrelation is too high, a straightforward way around this problem is to subsample the data until the noise-correlation goes away. PyDaddy provides an easy way to do this: initialize :code:`Characterize()` (:class:`pydaddy.daddy.Characterize`) with parameters :code:`Dt=T, dt=T` where :code:`T` is the autocorrelation time (in number of time-steps) rounded to the nearest integer. However, note that using larger values of :code:`T` can distort the estimated drift and diffusion functions.