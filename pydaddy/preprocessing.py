from collections.abc import Iterable

import numpy as np
from pydaddy.analysis import GaussianTest


class Preprocessing(GaussianTest):
	"""
	pass

    :meta private:
	"""
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
		GaussianTest.__init__(self)

	def _r2_vs_order(self, op1, op2, avgDrift, avgDiff, max_order):
		"""
		Get R2 for different order
		"""
		r2_drift = []
		r2_diff = []
		for i in range(max_order):
			p_drift, _ = self._fit_poly(x=op1, y=avgDrift, deg=i)
			p_diff, _ = self._fit_poly(x=op2, y=avgDiff, deg=i)
			r2_drift.append(
				self._R2(data=avgDrift, op=op1, poly=p_drift, k=i, adj=True))
			r2_diff.append(
				self._R2(data=avgDiff, op=op2, poly=p_diff, k=i, adj=True))
		return r2_drift, r2_diff

	def _remove_nan(self, x, y, sample_size=10):
		nan_idx = np.argwhere(np.isnan(y))
		
		x, y = np.delete(x, nan_idx), np.delete(y, nan_idx)
		if len(x) < sample_size:
			sample_size = len(x)
		idx = np.linspace(0, len(x)-1, sample_size, dtype=np.int)
		return x[idx], y[idx]

	def _remove_outliers(self, xs, y, quantile=0.01):
		""" Remove points corresponding to outliers in y. xs is a list of one or more arrays, indices corresponding
		to outliers in y will be removed from each array in xs as well. """

		lb = np.nanquantile(y, quantile)
		ub = np.nanquantile(y, 1 - quantile)

		cond = (lb <= y) & (y <= ub)
		y = y[cond]
		xs = [x[cond] for x in xs]

		return xs, y

	def _r2_vs_order_multi_dt(self,
							  X,
							  M_square,
							  t_int,
							  dt=1,
							  max_order=10,
							  inc=0.01):
		"""
		Get R2 vs order for different Dt
		"""
		r2_drift_m_dt = []
		r2_diff_m_dt = []
		max_dt = self._act(M_square, t_lag=self.t_lag)
		N = 8
		time_scale_list = sorted(set(map(int, np.linspace(1, max_dt, N))).union(set([self.Dt])))
		for time_scale in time_scale_list:
			drift, diff, avgDiff, avgDrift, op, drift_ebar, diff_ebar, _, _, _, _ = self._drift_and_diffusion(
				X, t_int, Dt=time_scale, dt=time_scale, inc=inc)
			op1, avgDrift = self._remove_nan(op, avgDrift)
			op2, avgDiff = self._remove_nan(op, avgDiff)
			if len(avgDrift) == 0 or len(avgDiff) == 0:
				continue
			r2_drift, r2_diff = self._r2_vs_order(op1, op2, avgDrift, avgDiff,max_order)
			r2_drift_m_dt.append(r2_drift)
			r2_diff_m_dt.append(r2_diff)

		r2_drift_m_dt.append(time_scale_list)
		r2_diff_m_dt.append(time_scale_list)

		return r2_drift_m_dt, r2_diff_m_dt

	def _rms_variation(self, x):
		"""
		Get rms variation of array
		"""
		y = []
		for i in range(len(x)):
			y.append(self._rms(x[i:]))
		return np.array(y)

	def _o1(self, x, i=0):
		"""
		All possible values of order
		"""
		if x[i] < x.mean():
			return i
		return self._o1(x, i=i+1)

	def _o2(self, x):
		"""
		Least likely values of order
		"""
		return np.ceil(self._fit_exp(range(len(x)), x)[0][1])


	def _get_o1_o2(self, x):
		"""
		Get o1 and o2 values for r2_adjusted multiple Dt
		"""
		o1 = []
		o2 = []
		for i in x:
			o1.append(self._o1(self._rms_variation(i)))
			o2 .append(self._o2(i))
		return np.array(o1), o2

	def _find_order(self, x):
		"""
		Get expected order by elemination least likely to be values from all possble values.
		Then decides the order by looking at the R2 values.
		"""
		x = np.array(x)
		if len(x.shape) !=2:
			x = np.array([x])
		return self._o1(self._rms_variation(x[0]))
		
		o1, o2 = self._get_o1_o2(x)
		if np.all(o1==o1[0]):
			return o1[0]
		for d in set(o2):
			i = np.where(o1 == d)
			o1 = np.delete(o1, i)
		p = sorted(set(o1))
		for i in range(len(p) - 1):
			if np.any(x[:,p[i]] > x[:,p[i+1]]):
				return p[i]
		return p[-1]

		"""
		o = []
		for i in range(len(x)):
			rms_x = self._rms_variation(x[i])
			d = np.abs(np.diff(rms_x)[:-1] - np.diff(rms_x, 2))
			o.append(np.where(d <= 0.25*d.mean())[0][0])
		return mode(o).mode[0]
		"""


	def _order(self,
			   X,
			   M_square,
			   t_int,
			   Dt='auto',
			   dt=1,
			   max_order=10,
			   inc=0.01):
		"""
		Find the order of drift and diffusion, and timescale based on drift order.

		Notes
		-----
			Time scale = autocorrelation time if drift order is 1, else its auto correaltion time.
		"""

		#R2_adj multiple Dt
		self._r2_drift_m_dt, self._r2_diff_m_dt = self._r2_vs_order_multi_dt(X, M_square, t_int=t_int ,inc=inc, dt=dt, max_order=max_order)

		if self.drift_order is None:
			self.drift_order = self._find_order(self._r2_drift_m_dt[:-1])
		if self.diff_order is None:
			self.diff_order = self._find_order(self._r2_diff_m_dt[:-1])

		autocorr_time = self._get_autocorr_time(M_square, t_lag=self.t_lag)
		optimum_dt = autocorr_time - 1 if self.drift_order == 1 else autocorr_time / 10

		k = np.abs(np.array(self._r2_drift_m_dt[-1]) - optimum_dt).argmin()
		self._r2_drift = np.array(self._r2_drift_m_dt[k])
		self._r2_diff = np.array(self._r2_diff_m_dt[0])

		return self.drift_order, self._r2_drift, np.ceil(optimum_dt)


	def _optimium_timescale(self,
							X,
							M_square,
							t_int,
							Dt='auto',
							max_order=10,
							t_lag=1000,
							inc=0.01):
		"""
		Get timescale based on observed order of drift
		"""
		order, r2, optimum_dt = self._order(X,
						M_square,
						t_int,
						Dt=Dt,
						max_order=max_order,
						inc=inc)
		if Dt != 'auto':
			return Dt
		return int(optimum_dt)

	def _preprocess(self):
		self._validate_inputs()
		# inc = self.inc_x if self.vector else self.inc
		# self._r2_drift_m_dt, self._r2_diff_m_dt = self._r2_vs_order_multi_dt(self._X, self._M_square, t_int=self.t_int ,inc=inc, dt=self.dt, max_order=self.max_order)
		# k = self._r2_drift_m_dt[-1].index(self.Dt)
		# self._r2_drift = np.array(self._r2_drift_m_dt[k])
		# self._r2_diff = np.array(self._r2_diff_m_dt[0])
		return None

	def _timestep(self, t):
		return (t[-1]-t[0]) / (len(t)-1)

	def _validate_inputs(self):
		"""
		Initailize and validate all inputs.
		"""

		if not isinstance(self._data, Iterable):
			raise InputError('Characterize(data=[Mx,My],...)',
							 'data input must be a list of length 1 or 2!')
		for d in self._data:
			if np.isinf(d).any():
				raise ValueError('TimeSeries data must not contain inf')
		if len(self._data) == 1:
			self._X = self._data[0].flatten()
			self._M_square = self._X
			self.vector = False
		elif len(self._data) == 2:
			self._Mx, self._My = np.array(self._data[0]).flatten(), np.array(self._data[1]).flatten()
			self._M_square = self._Mx**2 + self._My**2
			self._X = self._Mx.copy()
			self.vector = True
		else:
			raise InputError('Characterize(data=[Mx,My],...)',
							 'data input must be a list of length 1 or 2!')

		if hasattr(self._t, "__len__"):
			self.t_int = self._timestep(self._t)
			if len(self._t) != len(self._M_square):
				raise InputError("len(t) = len(Mx) = len(My)","TimeSeries and time-stamps must be of same length")
		else:
			self.t_int = self._t
			if not isinstance(self._t, (float, int)):
				raise InputError("t <float> or <array>","Time increment must either array or float type")

		if self.t_lag >= len(self._X):
			print('Warning : t_lag ({}) is greater that the length of data; setting t_lag as {}\n'.format(
				self.t_lag,
				len(self._data[0]) - 1))
			self.t_lag = len(self._X) - 1
		self.autocorrelation_time = self._get_autocorr_time(self._M_square)

		if self.vector:
			self._act_mx = self._act(self._Mx)
			self._act_my = self._act(self._My)

		if not self._isValidRange(self.op_range):
			if self.op_range is None:
				self.op_range = (np.nanmin(self._X), np.nanmax(self._X))
			else:
				print("Warning : given order parameter range is not in valid (typle or list of length 2) format\nUsing range of data")
				self.op_range = (np.nanmin(self._X), np.nanmax(self._X))

		if self.vector:
			if not self._isValidRange(self.op_x_range):
				if self.op_x_range is None:
					self.op_x_range = (np.nanmin(self._Mx), np.nanmax(self._Mx))
				else:
					print("Warning : given order parameter range is not in valid (typle or list of length 2) format\nUsing range of data")
					self.op_x_range = (np.nanmin(self._Mx), np.nanmax(self._Mx))

			if not self._isValidRange(self.op_y_range):
				if self.op_y_range is None:
					self.op_y_range = (np.nanmin(self._My), np.nanmax(self._My))
				else:
					print("Warning : given order parameter range is not in valid (typle or list of length 2) format\nUsing range of data")
					self.op_y_range = (np.nanmin(self._My), np.nanmax(self._My))


		if self.bins:
			if self.vector:
				#r_mx = (min(self._Mx), max(self._Mx))
				r_mx = self.op_x_range
				r_my = self.op_y_range
				#r_my = (min(self._My), max(self._My))
				self.inc_x = (r_mx[-1] - r_mx[0])/self.bins
				self.inc_y = (r_my[-1] - r_my[0])/self.bins
				self.inc = self.inc_x/10 
			#r = (min(self._X), max(self._X))
			r = self.op_range
			self.inc = (r[-1] - r[0])/self.bins
		else:
			try:
				assert self.inc > 0
				assert self.inc_x > 0
				assert self.inc_y > 0
			except AssertionError:
				raise InputError("inc, inc_x, inc_y must be > 0", " inc, inc_x, inc_y must be > 0")

		try:
			assert isinstance(self.dt, int)
			assert self.dt >= 1
			if self.Dt is None:
				self.Dt = int(np.ceil(self.autocorrelation_time/10))
			assert isinstance(self.Dt, int) and self.Dt >= 1
		except AssertionError:
			raise InputError("dt and Dt must be int and >= 1","dt and Dt must be int and >= 1")

		#if not self._isValidSliderRange(self.slider_range):
		#	self.slider_range = 'default'
		#	print("\n[Warning] : Entered slider range is not in valid format. Using default range.\nValid format <(slider_start, slider_stop, n_steps)>\nAll values must be >= 1\n")

		if not self._is_valid_slider_timescale_list(self.slider_timescales) and self.slider_timescales is not None:
			print("\n[Warning] : Given slider timescale list is not valid, or contains invalid timescales")

			return None		

class Error(Exception):
	"""
	Base class for exceptions in this module.
	
    :meta private:
	"""
	pass


class InputError(Error):
	"""Exception raised for errors in the input.

	Attributes:
		expression -- input expression in which the error occurred
		message -- explanation of the error

    :meta private:
	"""
	def __init__(self, expression, message):
		self.expression = expression
		self.message = message
