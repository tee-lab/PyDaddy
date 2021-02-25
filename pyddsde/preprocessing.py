import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyddsde.analysis import underlying_noise
from pyddsde.analysis import AutoCorrelation
from pyddsde.analysis import gaussian_test
from pyddsde.metrics import metrics
from pyddsde.sde import SDE


class preprocessing(gaussian_test):
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
		gaussian_test.__init__(self)

	def _get_dt(self, X):
		return int(self._get_autocorr_time(X, t_lag=self.t_lag) / 10)

	def _r2_vs_order(self, op, avgDrift, avgDiff, max_order):
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

	def _order(self,
			   X,
			   M_square,
			   t_int,
			   dt='auto',
			   delta_t=1,
			   max_order=10,
			   inc=0.01):
		
		"""
		dt = self._get_dt(X) + 5 if dt == 'auto' else dt
		_, _, avgDiff, avgDrift, op = self._drift_and_diffusion(
			X, t_int, dt=dt, delta_t=delta_t, inc=inc)
		self._r2_drift, self._r2_diff = self._r2_vs_order(
			op, avgDrift, avgDiff, max_order)
		"""

		#R2_adj multiple dt
		self._r2_drift_m_dt, self._r2_diff_m_dt = self._r2_vs_order_multi_dt(X, M_square, t_int=t_int ,inc=inc, delta_t=delta_t, max_order=max_order)

		try:
			drift_degree_list = []
			diff_degree_list = []
			for i in range(len(self._r2_drift_m_dt)-1):
				r2_drift = np.array(self._r2_drift_m_dt[i])
				r2_diff = np.array(self._r2_diff_m_dt[i])

				d_drift1 = np.diff(r2_drift, 1)
				d_diff1 = np.diff(r2_diff, 1)

				d_drift2 = np.diff(r2_drift, 2)
				d_diff2 = np.diff(r2_diff, 2)

				r_drift = np.where(np.logical_and(d_drift1[:-1] >= 0, d_drift2 <= 0))[0]
				r_diff = np.where(np.logical_and(d_diff1[:-1] >= 0, d_diff2 <= 0))[0]

				drift_degree_list.append(r_drift[np.nonzero(r_drift)][0] + 1) 
				diff_degree_list.append(r_diff[np.nonzero(r_diff)][0] + 1) 
			if self.drift_order is None:
				drift_degrees, drift_count = np.unique(drift_degree_list, return_counts=True)
				self.drift_order = drift_degrees[drift_count.argmax()]
			if self.diff_order is None:
				diff_degrees, diff_count = np.unique(diff_degree_list, return_counts=True)
				self.diff_order = diff_degrees[diff_count.argmax()]

			self._r2_drift = self._r2_drift_m_dt[np.where(np.array(drift_degree_list) == self.drift_order)[0][0]]
			self._r2_diff = self._r2_diff_m_dt[np.where(np.array(diff_degree_list) == self.diff_order)[0][0]]
		except:
			self._r2_drift = self._r2_drift_m_dt[0]
			self._r2_diff = self._r2_diff_m_dt[0]

			if self.drift_order is None:
				self.drift_order = np.where(np.isclose(self._r2_drift, max(self._r2_drift), atol=0.1))[0][0]
			
			if self.diff_order is None:
				self.diff_order = np.where(np.isclose(self._r2_diff, max(self._r2_diff), atol=0.1))[0][0]


		return self.drift_order, np.array(self._r2_drift)

	def _r2_vs_order_multi_dt(self,
							  X,
							  M_square,
							  t_int,
							  delta_t=1,
							  max_order=10,
							  inc=0.01):
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

	def _opt_dt_estimate(self,
						 X,
						 M_square,
						 t_int,
						 dt='auto',
						 max_order=10,
						 inc=0.01,
						 t_lag=1000):
		order, r2 = self._order(X,
								M_square,
								t_int,
								dt=dt,
								max_order=max_order,
								inc=inc)
		autocorr_time = self._get_autocorr_time(M_square, t_lag=self.t_lag)
		optimum_dt = autocorr_time - 1 if order == 1 else autocorr_time / 10
		return int(np.ceil(optimum_dt))

	def _optimium_timescale(self,
							X,
							M_square,
							t_int,
							dt='auto',
							max_order=10,
							t_lag=1000,
							inc=0.01):
		if dt != 'auto':
			_ = self._order(X, M_square, t_int)
			return dt
		return self._opt_dt_estimate(X,
									 M_square,
									 t_int,
									 dt=dt,
									 max_order=max_order,
									 t_lag=t_lag,
									 inc=inc)


"""
	def _detailed_estimate(self, X, M_square, t_int, dt='auto', delta_t=1, max_order=10, inc=0.01, t_lag=1000):
		self._kl_min = []
		self._kl_max = []
		self._kl_min_index = []
		autocorr_time = self._get_autocorr_time(M_square, t_lag=t_lag)
		order,_ = self._order(X, M_square, t_int, dt='auto', max_order=max_order, inc=inc)
		for i in tqdm(range(1,autocorr_time)):
			_,_,_, avgDrift, op = self._drift_and_diffusion(X, t_int, dt=i, delta_t=delta_t ,inc=inc)
			q_poly, op = self._fit_poly(x=op, y=avgDrift, deg=order)
			q = q_poly(op)
			kl = []
			for _dt in range(1,autocorr_time):
				_,_,_, avgDrift, op = self._drift_and_diffusion(X, t_int, dt=_dt, delta_t=delta_t ,inc=inc)
				p_poly, op = self._fit_poly(x=op, y=avgDrift, deg=order)
				p = p_poly(op)
				kl.append(self._kl_divergence(p,q))
			kl = np.array(kl)
			self._kl_min.append(kl.min())
			self._kl_max.append(kl.max())
			self._kl_min_index.append(kl.argmin())
		print("Optimium dt found : {}".format(np.abs(self._kl_min).argmin()))
		return np.abs(self._kl_min).argmin()
"""
