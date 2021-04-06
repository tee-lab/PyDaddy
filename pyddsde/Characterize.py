import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pkg_resources
import pickle
import tqdm
import time
import os
from pyddsde.sde import SDE
from pyddsde.analysis import underlying_noise
from pyddsde.analysis import AutoCorrelation
from pyddsde.analysis import gaussian_test
from pyddsde.preprocessing import preprocessing
from pyddsde.preprocessing import InputError
from pyddsde.metrics import metrics
from pyddsde.output import output

warnings.filterwarnings("ignore")

__all__ = ['Characterize', 'load_data']


class Main(preprocessing, gaussian_test, AutoCorrelation):
	"""
	main class

    :meta private:
	"""
	def __init__(
			self,
			data,
			t=1,
			Dt=None,
			dt=1,
			t_lag=1000,
			bins=20,
			inc=0.01,
			inc_x=0.1,
			inc_y=0.1,
			fft=True,
			slider_range='defult',
			slider_timescales = None,
			n_trials=1,
			show_summary=True,
			max_order = 9,
			**kwargs):

		self._data = data
		self._t = t
		self.Dt = Dt

		self.t_lag = t_lag
		self.max_order = max_order
		self.inc = inc
		self.inc_x = inc_x
		self.inc_y = inc_y
		self.dt = dt
		self.fft = fft
		self.n_trials = n_trials
		self._show_summary = show_summary

		#self.drift_order = None
		#self.diff_order = None

		self.op_range = None
		self.op_x_range = None
		self.op_y_range = None
		self.bins = bins
		self.slider_range = slider_range
		self.slider_timescales = slider_timescales

		"""
		# When t_lag is greater than timeseries length, reassign its value as length of data
		if self.t_lag > len(data[0]):
			print('Warning : t_lag is greater that the length of data; setting t_lag as {}\n'.format(len(data[0]) - 1))
			self.t_lag = len(data[0]) - 1
		"""

		self.__dict__.update(kwargs)
		preprocessing.__init__(self)
		gaussian_test.__init__(self)
		AutoCorrelation.__init__(self)
		#SDE.__init__(self)

		#if t is None and t_int is None:
		#	raise InputError("Characterize(data, t, t_int)","Missing data. Either 't' ot 't_int' must be given, both cannot be None")

		return None

	def _slider_data(self, Mx, My):
		drift_data_dict = dict()
		diff_data_dict = dict()
		cross_diff_dict = dict()
		time_scale_list = self._get_slider_timescales(self.slider_range, self.slider_timescales)
		for time_scale in tqdm.tqdm(time_scale_list, desc='Generating Slider data'):
			if self.vector:
				avgdriftX, avgdriftY, avgdiffX, avgdiffY, avgdiffXY, avgdiffYX, op_x, op_y = self._vector_drift_diff(Mx,My,inc_x=self.inc_x,inc_y=self.inc_y,t_int=self.t_int, Dt=time_scale, dt=time_scale)
				drift_data = [avgdriftX/self.n_trials, avgdriftY/self.n_trials, op_x, op_y]
				diff_data = [avgdiffX/self.n_trials, avgdiffY/self.n_trials, op_x, op_y]
				cross_diff_data = [avgdiffXY/self.n_trials, avgdiffYX/self.n_trials, op_x, op_y]
			else:
				_, _, avgdiff, avgdrift, op = self._drift_and_diffusion(Mx, t_int=self.t_int, Dt=time_scale, dt=time_scale, inc=self.inc)
				drift_data = [avgdrift/self.n_trials, op]
				diff_data = [avgdiff/self.n_trials, op]

			drift_data_dict[time_scale] = drift_data
			diff_data_dict[time_scale] = diff_data

			if self.vector: 
				cross_diff_dict[time_scale] = cross_diff_data

		if self.vector:
			return drift_data_dict, diff_data_dict, cross_diff_dict
		return drift_data_dict, diff_data_dict

	def __call__(self, data, t=1, Dt=None, **kwargs):
		self.__dict__.update(kwargs)
		#if t is None and t_int is None:
		#	raise InputError("Either 't' or 't_int' must be given, both cannot be None")
		self._t = t
		"""
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
		"""
		self._preprocess()
		"""
		self.dt_ = self._optimium_timescale(self._X,
										   self._M_square,
										   t_int=self.t_int,
										   Dt=Dt,
										   max_order=self.max_order,
										   t_lag=self.t_lag,
										   inc=self.inc_x)
	   	"""
		if not self.vector:
			if self.slider_range is None and self.slider_timescales is None:	
				self._drift_slider = dict()
				self._diff_slider = dict()
				_, _, self._avgdiff_, self._avgdrift_, self._op_ = self._drift_and_diffusion(
					self._X,
					self.t_int,
					Dt=self.Dt,
					dt=self.dt,
					inc=self.inc)
				self._avgdiff_ = self._avgdiff_ / self.n_trials
				self._avgdrift_ = self._avgdrift_ / self.n_trials
				self._drift_slider[self.Dt] = [self._avgdrift_, self._op_]
				self._diff_slider[self.dt] = [self._avgdiff_, self._op_]
			else:
				self._drift_slider, self._diff_slider = self._slider_data(self._X, None)
				self._avgdrift_, self._op_ = self._drift_slider[self.Dt]
				self._avgdiff_ = self._diff_slider[self.dt][0]
			self._cross_diff_slider = None

		else:
			if self.slider_range is None and self.slider_timescales is None:
				self._drift_slider = dict()
				self._diff_slider = dict()
				self._cross_diff_slider = dict()
				self._avgdriftX_, self._avgdriftY_, self._avgdiffX_, self._avgdiffY_, self._avgdiffXY_, self._avgdiffYX_, self._op_x_, self._op_y_ = self._vector_drift_diff(
					self._Mx,
					self._My,
					inc_x=self.inc_x,
					inc_y=self.inc_y,
					t_int=self.t_int,
					Dt=self.Dt,
					dt=self.dt)
				self._avgdriftX_ = self._avgdriftX_ / self.n_trials
				self._avgdriftY_ = self._avgdriftY_ / self.n_trials
				self._avgdiffX_ = self._avgdiffX_ / self.n_trials
				self._avgdiffY_ = self._avgdiffY_ / self.n_trials
				self._avgdiffXY_ = self._avgdiffXY_ / self.n_trials
				self._drift_slider[self.Dt] = [self._avgdriftX_, self._avgdriftY_, self._op_x_, self._op_y_]
				self._diff_slider[self.dt] = [self._avgdiffX_, self._avgdiffY_, self._op_x_, self._op_y_]
				self._cross_diff_slider[self.dt] = [self._avgdiffXY_, self._avgdiffYX_, self._op_x_, self._op_y_]
			else:
				self._drift_slider, self._diff_slider, self._cross_diff_slider = self._slider_data(self._Mx, self._My)
				self._avgdriftX_, self._avgdriftY_, self._op_x_, self._op_y_ = self._drift_slider[self.Dt]
				self._avgdiffX_, self._avgdiffY_ = self._diff_slider[self.dt][:2]
				self._avgdiffXY_, self._avgdiffYX_ = self._cross_diff_slider[self.dt][:2]

		inc = self.inc_x if self.vector else self.inc
		self.gaussian_noise, self._noise, self._kl_dist, self.k, self.l_lim, self.h_lim, self._noise_correlation = self._noise_analysis(
			self._X, self.Dt, self.dt, self.t_int, inc=inc, point=0)
		#X, Dt, dt, t_int, inc=0.01, point=0,
		return output(self)


class Characterize(object):
	"""
	Analyse a time series data and get drift and diffusion plots.

	Args
	----
	data : list
		time series data to be analysed, data = [x] for scalar data and data = [x1, x2] for vector
		where x, x1 and x2 are of numpy.array object type
	t : float, array, optional(default=1.0)
		float if its time increment between observation

		numpy.array if time stamp of time series
	Dt : int,'auto', optional(default='auto')
		time scale for drift

		if 'auto' time scale is decided based of drift order.
	dt : int, optional(default=1)
		time scale for difusion
	inc : float, optional(default=0.01)
		increment in order parameter for scalar data
	inc_x : float, optional(default=0.1)
		increment in order parameter for vector data x1
	inc_y : float, optional(default=0.1)
		increment in order parameter for vector data x2
	fft : bool, optional(default=True)
		if true use fft method to calculate autocorrelation else, use standard method
	slider_range : tuple, optional(default=None)
		range of the slider values, (start, stop, n_steps),
		if None, uses the default range, ie (1, 2*auto_correlation_time, 8)
	slider_timescales : list, optional(default=None)
		List of timescale values to include in slider.
	n_trials : int, optional(default=1)
		Number of trials, concatenated timeseries of multiple trials is used.
	show_summary : bool, optional(default=True)
		print data summary and show summary chart.

	**kwargs 
		all the parameters for inherited methods.

	returns
	-------
	output : pyddsde.output.output
		object to access the analysed data, parameters, plots and save them.
	"""
	def __new__(
			cls,
			data,
			t=1.0,
			Dt=None,
			dt=1,
			bins=20,
			inc=0.01,
			inc_x=0.1,
			inc_y=0.1,
			slider_range='default',
			slider_timescales=None,
			n_trials=1,
			show_summary=True,
			**kwargs):

		ddsde = Main(
			data=data,
			t=t,
			Dt=Dt,
			dt=dt,
			bins=bins,
			inc=inc,
			inc_x=inc_x,
			inc_y=inc_y,
			slider_range=slider_range,
			slider_timescales=slider_timescales,
			n_trials=n_trials,
			show_summary=show_summary,
			**kwargs)

		return ddsde(data=data, t=t, Dt=Dt)

def load_sample_data(data_path):
	r"""
	Load the sample distrubuted data

	data
	├── fish_data
	│   └── ectropus.csv
	└── model_data
		├── scalar
		│   ├── pairwise.csv
		│   └── ternary.csv
		└── vector
			├── pairwise.csv
			└── ternary.csv


	Each data file in pairwise, ternary and extras have two columns;
	first column is the timeseries data x, and the second one is the time stamp

	vector_data.csv also has two columns but contains the vector data x1 and x2 with missing time stamp. Use t_int=0.12.
	"""
	stream = pkg_resources.resource_stream('pyddsde', data_path)
	try:
		return np.loadtxt(stream, delimiter=',')
	except:
		return np.loadtxt(stream)

def load_sample_dataset(name):
	r"""
	Load sample data set provided.

	Available data sets:

	'fish-data-etroplus'

	'model-data-scalar-pairwise'

	'model-data-scalar-ternary'

	'model-data-vector-pairwise'

	'model-data-vector-ternary'

	Parameters
	----------
	name : str
		name of the data set

	Returns
	-------
	data : list
		timeseries data
	t : float, array
		timescale
	"""
	data_dict = {
	'fish-data-etroplus' : 'data/fish_data/ectropus.csv',
	'model-data-scalar-pairwise' : 'data/model_data/scalar/pairwise.csv',
	'model-data-scalar-ternary' : 'data/model_data/scalar/ternary.csv',
	'model-data-vector-pairwise' : 'data/model_data/vector/pairwise.csv',
	'model-data-vector-ternary' : 'data/model_data/vector/ternary.csv'
	}
	if name not in data_dict.keys():
		print('Invalid data set name\nAvaiable data set\n{}'.format(list(data_dict.keys())))
		raise InputError('','Invalid data set name')

	if 'scalar' in name:
		M, t = load_sample_data(data_dict[name]).T
		return [M], t
	Mx, My = load_sample_data(data_dict[name]).T
	return [Mx, My], 0.12


