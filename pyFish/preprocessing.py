import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyFish.analysis import underlying_noise
from pyFish.analysis import AutoCorrelation
from pyFish.analysis import gaussian_test
from pyFish.metrics import metrics
from pyFish.sde import SDE

class preprocessing(AutoCorrelation, SDE, metrics):
	def __init__(self,**kwargs):
		self.__dict__.update(kwargs)
		AutoCorrelation.__init__(self)
		SDE.__init__(self)
		metrics.__init__(self)

	def _get_dt(self, X):
		return int(self.get_autocorr_time(X, t_lag=1000)/10)

	def order(self, X, t_int, dt='auto', delta_t=1, max_order=10, inc=0.01):
		adj = False if self.order_metric=="R2" else True
		self._r2_drift = []
		self._r2_diff = []
		dt = self._get_dt(X)+5 if dt == 'auto' else dt
		#t_int = t[-1]/len(t)
		_,_,avgDiff, avgDrift, op = self.drift_and_diffusion(X, t_int, dt=dt, delta_t=delta_t, inc=inc)
		for i in range(max_order):
			p_drift, _ = self.fit_poly(x=op, y=avgDrift, deg=i)
			p_diff, _ = self.fit_poly(x=op, y=avgDiff, deg=i)
			self._r2_drift.append(self.R2(data=avgDrift,op=op, poly=p_drift, k=i, adj=adj))
			self._r2_diff.append(self.R2(data=avgDiff, op=op, poly=p_diff, k=i, adj=adj))
		#self.drift_order = np.diff(r2_drift).argmax() + 1
		self.drift_order = np.where(np.isclose(self._r2_drift, max(self._r2_drift), atol=0.1))[0][0]
		#self.diff_order = np.diff(self._r2_diff).argmax() + 1
		self.diff_order = np.where(np.isclose(self._r2_diff, max(self._r2_diff), atol=0.1))[0][0]
		#plt.plot(range(max_order), self._r2_drift)
		#plt.show()
		return self.drift_order , np.array(self._r2_drift)

	def simple_estimate(self, X,t_int, dt='auto',max_order=10, inc=0.01, t_lag=1000):
		order, r2 = self.order(X, t_int, dt=dt, max_order=max_order, inc=inc)
		#print('_____________order')
		#print(order)
		autocorr_time = self.get_autocorr_time(X, t_lag=t_lag)
		optimum_dt = autocorr_time - 1 if order==1 else autocorr_time/10
		#print(autocorr_time)
		return int(np.ceil(optimum_dt))

	"""
	def detail_estimate(self, X, t, dt='auto', delta_t=1 ,max_order=10, t_lag=1000, inc=0.01):
		rms = []
		autocorr_time = AutoCorrelation.get_autocorr_time(X, t_lag=t_lag)
		order = self.order(X, t, dt='auto', max_order=max_order, inc=inc)
		for i in range(1,autocorr_time):
			t_int = t[-1]/len(t)
			_,_,_, avgDrift, op = SDE.drift_and_diffusion(X, t_int, dt=i, delta_t=delat_t ,inc=inc)
			if i == 1:
				p_, op_ = metrics.fit_poly(x=op, y=avgDrift, deg=order)
				continue
			p, op = metrics.fit_poly(x=op, y=avgDrift, deg=order)
			rms.append(metrics.rms(p_(op_) - p(op)))
			p_, op_ = p, op
		rms = np.array(rms)
		r = np.array(list(range(1,autocorr_time))[1:])
		coeff1, coeff2 = AutoCorrelation.fit_exp(r, rms)
		return coeff1[-1]
	"""

	def detailed_estimate(self, X, t_int, dt='auto', delta_t=1, max_order=10, inc=0.01, t_lag=1000):
		self._kl_min = []
		self._kl_max = []
		self._kl_min_index = []
		autocorr_time = self.get_autocorr_time(X, t_lag=t_lag)
		order,_ = self.order(X, t_int, dt='auto', max_order=max_order, inc=inc)
		for i in tqdm(range(1,autocorr_time)):
			_,_,_, avgDrift, op = self.drift_and_diffusion(X, t_int, dt=i, delta_t=delta_t ,inc=inc)
			q_poly, op = self.fit_poly(x=op, y=avgDrift, deg=order)
			q = q_poly(op)
			kl = []
			for _dt in range(1,autocorr_time):
				_,_,_, avgDrift, op = self.drift_and_diffusion(X, t_int, dt=_dt, delta_t=delta_t ,inc=inc)
				p_poly, op = self.fit_poly(x=op, y=avgDrift, deg=order)
				p = p_poly(op)
				kl.append(self.kl_divergence(p,q))
			kl = np.array(kl)
			self._kl_min.append(kl.min())
			self._kl_max.append(kl.max())
			self._kl_min_index.append(kl.argmin())
		print("Optimium dt found : {}".format(np.abs(self._kl_min).argmin()))
		return np.abs(self._kl_min).argmin()



	def optimium_timescale(self, X, t_int, simple_method=True, dt='auto', max_order=10, t_lag=1000, inc=0.01):
		if dt != 'auto':
			_ = self.order(X,t_int)
			return dt
		if simple_method:
			return self.simple_estimate(X,t_int,dt=dt,max_order=max_order,t_lag=t_lag, inc=inc)
		else:
			return self.detailed_estimate(X,t_int, dt=dt, max_order=max_order, t_lag=t_lag, inc=inc)




