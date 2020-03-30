import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from pyFish.sde import SDE
from pyFish.analysis import underlying_noise
from pyFish.analysis import AutoCorrelation
from pyFish.analysis import gaussian_test
from pyFish.preprocessing import preprocessing
from pyFish.metrics import metrics
from pyFish.output import output

class Characterize(preprocessing):
	def __init__(self, **kwargs):
		"""
		input parms:
		t_lag : maxmium lag for autocorelation relation analysis
		simple_method : rule to use to find Dt
		inc : bin increments
		delta_t : time step to calculate diffusion
		"""
		self.t_lag = 1000
		self.simple_method = True
		self.max_order = 10
		self.inc = 0.01
		self.dt_ = 'auto'
		self.delta_t = 1
		#self.optimium_timescale = optimium_timescale()
		self.__dict__.update(kwargs)
		preprocessing.__init__(self)

	def _timestep(self, t):
		return t[-1]/len(t)

	def visualize(self, X, t):
		drift, drff, avgdrift, avgdiff, op = self.__call__(X,t)
		#Time series
		fig1 = fig = plt.figure(dpi=150)
		plt.plot(t,X)
		plt.title('Figure 1')
		#PDF
		fig2 = fig = plt.figure(dpi=150, figsize=(5,5))
		sns.distplot(X)
		plt.title('Figure 2')
		plt.xlim([-1,1])
		plt.ylabel('PDF')
		plt.xlabel('Order Parameter')
		#Drift
		fig3 = fig = plt.figure(dpi=150,figsize=(5,5))
		p_drift, _ = self.fit_poly(op, avgdrift, self.drift_order)
		plt.scatter(op, avgdrift, marker='.')
		plt.scatter(op, p_drift(op), marker='.')
		plt.title('Figure 3')
		plt.xlabel('Order Parameter')
		plt.ylabel("Deterministic")
		plt.xlim([-1,1])
		#Diffusion
		fig4 = fig = plt.figure(dpi=150,figsize=(5,5))
		p_diff, _ = self.fit_poly(op, avgdiff, self.diff_order)
		plt.scatter(op, avgdiff, marker='.')
		plt.scatter(op, p_diff(op), marker='.')
		plt.title('Figure 4')
		plt.xlim([-1,1])
		plt.xlabel("Order Parameter")
		plt.ylabel('Stochastic')
		plt.show()

	def parameters(self):
		params = dict()
		for keys in self.__dict__.keys():
			if str(keys)[0] != '_':
				params[keys] = self.__dict__[keys]
		return params

	def __call__(self, X, t, inc=0.01, t_lag=1000, max_order=10, simple_method=True, dt='auto', **kwargs):
		self.t_int = self._timestep(t)
		self.__dict__.update(kwargs)
		self.dt = self.optimium_timescale(X, t, simple_method=self.simple_method, dt=dt, max_order=self.max_order, t_lag=self.t_lag, inc=self.inc)
		diff, drift, avgdiff, avgdrift, op = self.drift_and_diffusion(X, self.t_int, dt=self.dt, delta_t=self.delta_t, inc=self.inc)
		#self._drift, self._diff, self._avgdrift, self._avgdiff, self._op = drift, diff, avgdrift, avgdiff, op
		return output(X,t,drift, diff,avgdrift, avgdiff,op, self.drift_order, self.diff_order, self.__dict__)
		#return drift, diff, avgdrift, avgdiff, op
