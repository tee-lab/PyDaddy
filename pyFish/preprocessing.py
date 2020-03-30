import numpy as np 
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

	def order(self, X, t, dt='auto', delta_t=1, max_order=10, inc=0.01):
		self._r2_drift = []
		self._r2_diff = []
		dt = self._get_dt(X) if dt == 'auto' else dt
		t_int = t[-1]/len(t)
		_,_,avgDiff, avgDrift, op = self.drift_and_diffusion(X, t_int, dt=dt, delta_t=delta_t, inc=inc)
		for i in range(max_order):
			p_drift, _ = self.fit_poly(x=op, y=avgDrift, deg=i)
			p_diff, _ = self.fit_poly(x=op, y=avgDiff, deg=i)
			self._r2_drift.append(self.R2(data=avgDrift,op=op, poly=p_drift))
			self._r2_diff.append(self.R2(data=avgDiff, op=op, poly=p_diff))
		#self.drift_order = np.diff(r2_drift).argmax() + 1
		self.drift_order = np.where(np.isclose(self._r2_drift, max(self._r2_drift), atol=0.1))[0][0]
		self.diff_order = np.diff(self._r2_diff).argmax() + 1
		return self.drift_order , np.array(self._r2_drift)

	def simple_estimate(self, X,t, dt='auto',max_order=10, inc=0.01, t_lag=1000):
		order, r2 = self.order(X, t, dt=dt, max_order=max_order, inc=inc)
		#print(order)
		autocorr_time = self.get_autocorr_time(X, t_lag=t_lag)
		optimum_dt = autocorr_time - 1 if order==1 else autocorr_time/10
		#print(optimum_dt)
		return int(np.ceil(optimum_dt))

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

	def optimium_timescale(self, X, t, simple_method=True, dt='auto', max_order=10, t_lag=1000, inc=0.01):
		if dt != 'auto':
			_ = self.order(X,t)
			return dt
		if simple_method:
			return self.simple_estimate(X,t,dt=dt,max_order=max_order,t_lag=t_lag, inc=inc)
		else:
			return self.detail_estimate(X, t, dt=dt, max_order=max_order, t_lag=t_lag, inc=0.01)




