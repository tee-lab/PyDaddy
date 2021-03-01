import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyddsde.analysis import underlying_noise
from pyddsde.analysis import AutoCorrelation
from pyddsde.analysis import gaussian_test
from pyddsde.metrics import metrics
from pyddsde.sde import SDE


class preprocessing(gaussian_test):
	"""
	pass

    :meta private:
	"""
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
		gaussian_test.__init__(self)

	def _r2_vs_order(self, op, avgDrift, avgDiff, max_order):
		"""
		Get R2 for different order
		"""
		adj = False if self.order_metric == "R2" else True
		r2_drift = []
		r2_diff = []
		for i in range(max_order):
			p_drift, _ = self._fit_poly(x=op, y=avgDrift, deg=i)
			p_diff, _ = self._fit_poly(x=op, y=avgDiff, deg=i)
			r2_drift.append(
				self._R2(data=avgDrift, op=op, poly=p_drift, k=i, adj=adj))
			r2_diff.append(
				self._R2(data=avgDiff, op=op, poly=p_diff, k=i, adj=adj))
		return r2_drift, r2_diff

	def _r2_vs_order_multi_dt(self,
							  X,
							  M_square,
							  t_int,
							  delta_t=1,
							  max_order=10,
							  inc=0.01):
		"""
		Get R2 vs order for different dt
		"""
		r2_drift_m_dt = []
		r2_diff_m_dt = []
		max_dt = self._get_autocorr_time(M_square, t_lag=self.t_lag)
		N = 8
		time_scale_list = sorted(set(map(int, np.linspace(1, max_dt, N))))
		for time_scale in time_scale_list:
			drift, diff, avgDiff, avgDrift, op = self._drift_and_diffusion(
				X, t_int, dt=time_scale, delta_t=time_scale, inc=inc)
			r2_drift, r2_diff = self._r2_vs_order(op, avgDrift, avgDiff,max_order)
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
		Get o1 and o2 values for r2_adjusted multiple dt
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


	def _order(self,
			   X,
			   M_square,
			   t_int,
			   dt='auto',
			   delta_t=1,
			   max_order=10,
			   inc=0.01):
		"""
		Find the order of drift and diffusion, and timescale based on drift order.

		Notes
		-----
			Time scale = autocorrelation time if drift order is 1, else its auto correaltion time.
		"""

		#R2_adj multiple dt
		self._r2_drift_m_dt, self._r2_diff_m_dt = self._r2_vs_order_multi_dt(X, M_square, t_int=t_int ,inc=inc, delta_t=delta_t, max_order=max_order)

		self.drift_order = self._find_order(self._r2_drift_m_dt[:-1])
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
							dt='auto',
							max_order=10,
							t_lag=1000,
							inc=0.01):
		"""
		Get timescale based on observed order of drift
		"""
		order, r2, optimum_dt = self._order(X,
						M_square,
						t_int,
						dt=dt,
						max_order=max_order,
						inc=inc)
		if dt != 'auto':
			return dt
		return int(optimum_dt)
