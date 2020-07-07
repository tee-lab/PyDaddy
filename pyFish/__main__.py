import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pyFish.sde import SDE
from pyFish.analysis import underlying_noise
from pyFish.analysis import AutoCorrelation
from pyFish.analysis import gaussian_test
from pyFish.preprocessing import preprocessing
from pyFish.metrics import metrics
from pyFish.output import output
from pyFish.output import InputError

warnings.filterwarnings("ignore")

class Characterize(preprocessing, gaussian_test):
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
		self.order_metric = "R2_adj"
		#self.optimium_timescale = optimium_timescale()
		self.__dict__.update(kwargs)
		preprocessing.__init__(self)
		gaussian_test.__init__(self)
	
	def _timestep(self, t):
		return t[-1]/len(t)
	
	def __call__(self, data, t, inc=0.01, inc_x=0.1, inc_y=0.1, t_lag=1000, max_order=10, simple_method=True, dt='auto', **kwargs):
		self.__dict__.update(kwargs)
		if t is None and not hasattr(self,'t_int'):
			raise InputError('Characterize(data=[x1,x2], t=None', 'if t = None, then "t_int" parameter must be given')
		self._t = t
		if len(data) == 1:
			self._X = data[0]
			self.vector = False
		elif len(data) == 2:
			self._vel_x, self._vel_y = data
			vx = self.interploate_missing(self._vel_x)
			vy = self.interploate_missing(self._vel_y)
			#self._X = np.sqrt((np.square(vx) + np.square(vy)))
			self._X = vx
			self.vector = True
		else:
			raise InputError('Characterize(data=[x1,x2],...)', 'data input must be a list of length 1 or 2!')
		
		if not hasattr(self,'t_int'): self.t_int = self._timestep(t)
		self.dt = self.optimium_timescale(self._X, t_int=self.t_int, simple_method=self.simple_method, dt=dt, max_order=self.max_order, t_lag=self.t_lag, inc=self.inc)
		if not self.vector:
			self._diff, self._drift, self._avgdiff, self._avgdrift, self._op = self.drift_and_diffusion(self._X, self.t_int, dt=self.dt, delta_t=self.delta_t, inc=self.inc)
		else:
			self._avgdriftX, self._avgdriftY, self._avgdiffX, self._avgdiffY, self._avgdiffXY, self._op_x, self._op_y = self.vector_drift_diff(self._vel_x, self._vel_y, inc_x=self.inc_x, inc_y=self.inc_y, t_int=self.t_int, dt=self.dt, delta_t=self.delta_t)
		self.gaussian_noise, self._noise, self._kl_dist, self.k, self.l_lim, self.h_lim, self._noise_correlation = self.noise_analysis(self._X, self.dt, self.t_int, inc=self.inc, point=0)
		return output(self)

