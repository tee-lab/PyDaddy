import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import tqdm
import time
import os
from pyFish.sde import SDE
from pyFish.analysis import underlying_noise
from pyFish.analysis import AutoCorrelation
from pyFish.analysis import gaussian_test
from pyFish.preprocessing import preprocessing
from pyFish.metrics import metrics
from pyFish.output import output
from pyFish.output import InputError

warnings.filterwarnings("ignore")


class Main(preprocessing, gaussian_test, AutoCorrelation):
	"""
	main class
	"""
	def __init__(
			self,
			data,
			t=1,
			dt='auto',
			delta_t=1,
			#t_int=None,
			t_lag=1000,
			inc=0.01,
			inc_x=0.1,
			inc_y=0.1,
			max_order=10,
			fft=True,
			drift_order=None,
			diff_order=None,
			order_metric="R2_adj",
			n_trials=1,
			n_dt=8,
			**kwargs):

		self._data = data
		self._t = t
		self.dt_ = dt
		self.res_dir = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())

		#self.t_int = t_int
		self.t_lag = t_lag
		self.max_order = max_order
		self.inc = inc
		self.inc_x = inc_x
		self.inc_y = inc_y
		self.delta_t = delta_t
		self.order_metric = order_metric
		self.fft = fft
		self.drift_order = drift_order
		self.diff_order = drift_order
		self.n_trials = n_trials
		self.n_dt = n_dt

		# When t_lag is greater than timeseries length, reassign its value as length of data
		if self.t_lag > len(data[0]):
			print('Warning : t_lag is greater that the length of data; setting t_lag as {}\n'.format(len(data[0]) - 1))
			self.t_lag = len(data[0]) - 1

		self.__dict__.update(kwargs)
		preprocessing.__init__(self)
		gaussian_test.__init__(self)
		AutoCorrelation.__init__(self)

		#if t is None and t_int is None:
		#	raise InputError("Characterize(data, t, t_int)","Missing data. Either 't' ot 't_int' must be given, both cannot be None")

		return None

	def _timestep(self, t):
		return t[-1] / len(t)

	def _slider_data(self, Mx, My, save=False, savepath='results'):
		max_dt = np.ceil(self.autocorrelation_time) * 2
		drift_data_dict = dict()
		diff_data_dict = dict()
		for time_scale in tqdm.tqdm(np.linspace(1, max_dt, self.n_dt),
									desc='Generating Slider data'):
			if time_scale < 1:
				continue

			drift_data = self._vector_drift(Mx,
											My,
											inc_x=self.inc_x,
											inc_y=self.inc_y,
											t_int=self.t_int,
											dt=int(time_scale))
			diff_data = self._vector_diff(Mx,
										  My,
										  inc_x=self.inc_x,
										  inc_y=self.inc_y,
										  t_int=self.t_int,
										  delta_t=int(time_scale))

			for i in range(2):
				drift_data[i] = drift_data[i] / self.n_trials
				diff_data[i] = diff_data[i] / self.n_trials
			drift_data_dict[int(time_scale)] = drift_data
			diff_data_dict[int(time_scale)] = diff_data

		if save:
			savepath = self._make_directory(os.path.join(Docu, self.res_dir))
			with open(os.path.join(savepath, 'slider_data.pkl'), 'wb') as f:
				pickle.dump([drift_data_dict, diff_data_dict],
							f,
							protocol=pickle.HIGHEST_PROTOCOL)

		return drift_data_dict, diff_data_dict

	def __call__(self,
				 data,
				 t=1,
				 dt='auto',
				 inc=0.01,
				 inc_x=0.1,
				 inc_y=0.1,
				 t_lag=1000,
				 max_order=10,
				 simple_method=True,
				 **kwargs):
		self.__dict__.update(kwargs)
		#if t is None and t_int is None:
		#	raise InputError("Either 't' or 't_int' must be given, both cannot be None")
		self._t = t
		if len(data) == 1:
			self._X = np.array(data[0])
			self._M_square = np.array(data[0])
			self.vector = False
		elif len(data) == 2:
			self._Mx, self._My = np.array(data[0]), np.array(data[1])
			self._M_square = self._Mx**2 + self._My**2
			self._X = self._Mx.copy()
			self.vector = True
		else:
			raise InputError('Characterize(data=[Mx,My],...)',
							 'data input must be a list of length 1 or 2!')

		#if t_int is None: self.t_int = self._timestep(t)
		if not hasattr(t, "__len__"):
			self.t_int = t
		else:
			if len(t) != len(self._M_square):
				raise InputError(
					"len(Mx^2 + My^2) == len(t)",
					"TimeSeries and time-stamps must be of same length")
			self.t_int = self._timestep(t)

		#print('opt_dt')
		self.dt = self._optimium_timescale(self._X,
										   self._M_square,
										   t_int=self.t_int,
										   dt=dt,
										   max_order=self.max_order,
										   t_lag=self.t_lag,
										   inc=self.inc)
		if not self.vector:
			self._diff_, self._drift_, self._avgdiff_, self._avgdrift_, self._op_ = self._drift_and_diffusion(
				self._X,
				self.t_int,
				dt=self.dt,
				delta_t=self.delta_t,
				inc=self.inc)
			self._avgdiff_ = self._avgdiff_ / self.n_trials
			self._avgdrift_ = self._avgdrift_ / self.n_trials
		else:
			#print('drift diff')
			self._avgdriftX_, self._avgdriftY_, self._avgdiffX_, self._avgdiffY_, self._avgdiffXY_, self._op_x_, self._op_y_ = self._vector_drift_diff(
				self._Mx,
				self._My,
				inc_x=self.inc_x,
				inc_y=self.inc_y,
				t_int=self.t_int,
				dt=self.dt,
				delta_t=self.delta_t)
			self._avgdriftX_ = self._avgdriftX_ / self.n_trials
			self._avgdriftY_ = self._avgdriftY_ / self.n_trials
			self._avgdiffX_ = self._avgdiffX_ / self.n_trials
			self._avgdiffY_ = self._avgdiffY_ / self.n_trials
			self._avgdiffXY_ = self._avgdiffXY_ / self.n_trials
			self._drift_slider, self._diff_slider = self._slider_data(
				self._Mx, self._My)

		self.gaussian_noise, self._noise, self._kl_dist, self.k, self.l_lim, self.h_lim, self._noise_correlation = self._noise_analysis(
			self._X, self.dt, self.t_int, inc=self.inc, point=0)
		return output(self)


class Characterize(object):
	"""
	Input params:
	--------------
	data : list
		time series data to be analysed, data = [x] for scalar data and data = [x1, x2] for vector
		where x, x1 and x2 are of numpy.array object type
	t : float or numpy.array
		[float] if its time increment between observation
		[numpy.array] if time stamp of time series
	dt = 'auto' : 'auto' or int
		time scale to run the analysis on (for determinsitic part);
		algorithm estimates dt if 'auto' is passed, else takes the user input
	delta_t = 1 : int
		time scale to run the analysis on (for stochastic part)
	inc = 0.01 : float
		increment in order parameter for scalar data
	inc_x = 0.1 : float
		increment in order parameter for vector data x1
	inc_y = 0.1 : float
		increment in order parameter for vector data x2
	drift_order = None : int
		order of polynomial to be fit for calculated drift (deterministic part);
		if None, algorithim estimates the optimium drift_order
	diff_order = None : int
		order of polynomial to be fit for calculated diff (stochastic part);
		if None, algorithim estimates the optimium diff_order
	max_order = 10 : int
		maxmium drift_order and diff_order to consider
	fft = True : bool
		if true use fft method to calculate autocorrelation else, use standard method
	t_lag = 1000 : int
		maxmium lag to use to calculate acf

	**kwargs 
		all the parameters for pyFish.preporcessing and pyFish.noise_analysis

	returns:
	-----------
	output : pyFish.output
		object to access the analysed data, parameters, plots and save them.
	"""
	def __new__(
			cls,
			data,
			t=1,
			#t_int=None,
			dt='auto',
			delta_t=1,
			t_lag=1000,
			inc=0.01,
			inc_x=0.1,
			inc_y=0.1,
			max_order=10,
			fft=True,
			drift_order=None,
			diff_order=None,
			order_metric="R2_adj",
			n_trials=1,
			n_dt=8,
			**kwargs):
		"""
		Input params:
		--------------
		data : list
			time series data to be analysed, data = [x] for scalar data and data = [x1, x2] for vector
			where x, x1 and x2 are of numpy.array object type
		t : float or numpy.array
			[float] if its time increment between observation
			[numpy.array] if time stamp of time series
		dt = 'auto' : 'auto' or int
			time scale to run the analysis on (for determinsitic part);
			algorithm estimates dt if 'auto' is passed, else takes the user input
		delta_t = 1 : int
			time scale to run the analysis on (for stochastic part)
		inc = 0.01 : float
			increment in order parameter for scalar data
		inc_x = 0.1 : float
			increment in order parameter for vector data x1
		inc_y = 0.1 : float
			increment in order parameter for vector data x2
		drift_order = None : int
			order of polynomial to be fit for calculated drift (deterministic part);
			if None, algorithim estimates the optimium drift_order
		diff_order = None : int
			order of polynomial to be fit for calculated diff (stochastic part);
			if None, algorithim estimates the optimium diff_order
		max_order = 10 : int
			maxmium drift_order and diff_order to consider
		fft = True : bool
			if true use fft method to calculate autocorrelation else, use standard method
		t_lag = 1000 : int
			maxmium lag to use to calculate acf

		**kwargs 
			all the parameters for pyFish.preporcessing and pyFish.noise_analysis

		returns:
		-----------
		output : pyFish.output
			object to access the analysed data, parameters, plots and save them.
		"""
		sde = Main(
			data=data,
			t=t,
			dt=dt,
			delta_t=delta_t,
			#t_int=t_int,
			t_lag=t_lag,
			inc=inc,
			inc_x=inc_x,
			inc_y=inc_y,
			max_order=max_order,
			fft=fft,
			drift_order=drift_order,
			diff_order=diff_order,
			order_metric=order_metric,
			n_trials=n_trials,
			n_dt=n_dt,
			**kwargs)

		return sde(
			data=data,
			t=t,
			#t_int=t_int,
			dt=dt,
			inc=inc,
			inc_x=inc_x,
			inc_y=inc_y,
			t_lag=t_lag,
			max_order=max_order,
		)
