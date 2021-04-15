import numpy as np
import sys
import os
import gc
import json
import numpy.matlib
import tqdm
import sdeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import scipy.optimize
import scipy.stats
import scipy.io
import pickle
import time
import statsmodels.api as sm
import statsmodels.stats.diagnostic
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import namedtuple

from pyddsde.preprocessing import preprocessing
from pyddsde.visualize import visualize

__all__ = ['output']

class output(preprocessing, visualize):
	"""
	Class to plot and save data and parameters
	"""
	def __init__(self, ddsde, **kwargs):
		self.vector = ddsde.vector
		self._ddsde = ddsde
		self.fft = ddsde.fft
		self.op_range = ddsde.op_range
		self.op_x_range = ddsde.op_x_range
		self.op_y_range = ddsde.op_y_range

		if not self.vector:
			self._data_X = ddsde._X
			self._data_t = ddsde._t
			#self._data_drift = ddsde._drift_
			#self._data_diff = ddsde._diff_
			self._data_avgdrift = ddsde._avgdrift_
			self._data_avgdiff = ddsde._avgdiff_
			self._data_op = ddsde._op_
			#self.drift_order = ddsde.drift_order
			#self.diff_order = ddsde.diff_order

			visualize.__init__(self, None, None, self._data_op,
							   self._ddsde.autocorrelation_time)
		else:
			self._data_Mx = ddsde._Mx
			self._data_My = ddsde._My
			self._data_M = np.sqrt(self._data_Mx**2 + self._data_My**2)
			self._data_avgdriftX = ddsde._avgdriftX_
			self._data_avgdriftY = ddsde._avgdriftY_
			self._data_avgdiffX = ddsde._avgdiffX_
			self._data_avgdiffY = ddsde._avgdiffY_
			self._data_avgdiffXY = ddsde._avgdiffXY_
			self._data_avgdiffYX = ddsde._avgdiffYX_
			self._data_op_x = ddsde._op_x_
			self._data_op_y = ddsde._op_y_


			#self._drift_slider = ddsde._drift_slider
			#self._diff_slider = ddsde._diff_slider

			visualize.__init__(self, self._data_op_x, self._data_op_y, None,
							   self._ddsde.autocorrelation_time, _act_mx=self._ddsde._act_mx, _act_my=self._ddsde._act_my)

		self._drift_slider = ddsde._drift_slider
		self._diff_slider = ddsde._diff_slider
		self._cross_diff_slider = ddsde._cross_diff_slider
		self._time_scale_list = list(self._drift_slider.keys())

		self.res_dir = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())


		self.__dict__.update(kwargs)
		preprocessing.__init__(self)

		if ddsde._show_summary:
			return self.summary(ret_fig=False)
		return None

	def release(self):
		"""
		Clears the memory, recommended to be used while analysing multiple
		data files in loop.

		Returns
		-------
			None
		"""
		plt.close('all')
		gc.collect()
		return None


	def export_data(self, dir_path=None, save_mat=True, zip=False):
		"""
		Export all drift and diffusion data, to csv and matlab (mat) files

		Args
		----
		dir_path : str, optional(default=None)
			path to save the results, if None, data will be saved in 'results' folder in current working directory
		save_mat : bool, optional(default=True)
			If True, export data as mat files also.
		zip : bool, optional(default=False)
			If True, creates zip files of exported data

		Returns
		-------
		path : str
			path where data is exported
		"""

		if dir_path != None and not os.path.exists(dir_path):
			raise PathNotFound(dir_path, "Entered directory path does not exists.")

		self.res_dir = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
		if dir_path is None:
			dir_path = self._make_directory(os.path.join('pyddsde_exports', self.res_dir))
			self.res_dir = os.path.join(os.getcwd(), dir_path)
		else:
			dir_path = self._make_directory(os.path.join(dir_path, 'pyddsde_exports', self.res_dir))

		data_dict = self._get_stacked_data()
		for key in data_dict:
			self._save_csv(dir_path=dir_path, file_name=key, data=data_dict[key], fmt='%.4f', add_headers=True)

		if save_mat:
			savedict = self._combined_data_dict()
			scipy.io.savemat(os.path.join(dir_path, 'drift_diff_data.mat'), savedict)

		if zip:
			self._zip_dir(dir_path)

		return "Exported to {}".format(os.path.join(dir_path,self.res_dir))


	def data(self, drift_time_scale=None, diff_time_scale=None):
		"""
		Get the drift, diffusion and order parameter data for any timescale the analysis is done.

		Args
		----
		drift_time_scale : int, optional(default=None)
			time-scale of drift data, if None, returns data analysed for given dt
		diff_time_scale : int, optional(default=None)
			time-scale of diffusion data, if None, returns data analysed for given delta_t

		Returns
		-------
		list
			- if vector, [avgdriftX, avgdriftY, avgdiffX, avgdiffY, op_x, op_y] 
			- else, [avgdrift, avgdiff, op]
		"""
		if not self.vector:
			Data = namedtuple('Data', ('drift', 'diff', 'op'))
			drift , _ = self._get_data_from_slider(drift_time_scale)
			_ , diff = self._get_data_from_slider(diff_time_scale)
			return Data(drift, diff, self._data_op)

		Data = namedtuple('Data', ('driftX', 'driftY', 'diffX', 'diffY', 'diffXY', 'diffYX' ,'op_x', 'op_y'))
		driftX, driftY, _, _, _, _ = self._get_data_from_slider(drift_time_scale)
		_, _, diffX, diffY, diffXY, diffYX = self._get_data_from_slider(diff_time_scale)
		return Data(driftX, driftY, diffX, diffY, diffXY, diffYX, self._data_op_x, self._data_op_y)

	#def kl_div(self, a,b):
	#	a, bins_a = np.histogram(a, bins=100, density=True)
	#	b, bins_b = np.histogram(b, bins=bins_a, density=True)
	#	a_b = np.sum(np.where((a != 0)&(b != 0), a * np.log(a / b), 0))
	#	b_a = np.sum(np.where((a != 0)&(b != 0), b * np.log(b / a), 0))
	#	return (a_b + b_a)/2


	def derive_functions(self, drift_order, diff_order, t_inc=None, T=None, m0=None, Dt='default', dt=1):
		"""
		Simulated to optimize the best fitting functional forms for the derrived drift and diffusion 
		coefficients.

		Args
		----
		drift_order : int
			order of the drift coefficient observed
		diff_order : int
			order of the diffusion coefficient observed
		t_inc : float (or None)
			time increment for the SDE integration, if None
			time_increment will be taken as 1/autocorrelation time or
			the observed (input) timeseries
		T : int (or None)
			total time units of simulation, if None,
			T will be taken as the fraction of  total time units of 
			observed (input) data and its autocorrealtion time.
		m0 : float (or None)
			initial state of the SDE, if None, then m0 will be the first 
			value of the input data
		Dt : list or 'default', 'all'
			list of drift timescales for to simulate reconstructed SDE and compare
			it with the observed data.

			If 'default', Dt = range(1, autocorrealtion_time+1, 10)

			if 'all', Dt will include all timescales for which the drift and 
			diffusion coefficients are derrived.
		dt : int, (default = 1)
			diffusion time scale 

		Returns
		-------
		opt_F : callable
			drift polynomila function of the given order, and the timescale for which the simulated
			SDE's PDF was in close match with that of the observed.
		opt_G : callable
			diffusion polynomial function of the given order and timescale `dt`.

		Note
		----
		If the Dt set contains, timescale values for which the drift and diffusion 
		coefficents have not been derrived, then such timescale values will be ignored.

		In other words, Dt set must be a subset of the set containing the slider timescales.

		"""
		if self.vector:
			print("Feature not implemented for vector data")
			return None

		if Dt == 'default':
			start, stop, n_steps = 1, self._ddsde.autocorrelation_time, 10
			Dt = set(np.linspace(1, self._ddsde.autocorrelation_time, 10, dtype=np.int))
			Dt = Dt.intersection(set(self._ddsde._avaiable_timescales))
		elif Dt == 'all':
			Dt = list(self._ddsde._avaiable_timescales)
		else:
			Dt = set(Dt)
			Dt = Dt.intersection(set(self._ddsde._avaiable_timescales))
		Dt = sorted(list(Dt))

		if t_inc is None:
			t_inc = 1/self._ddsde.autocorrelation_time
		if T is None:
			T = int(np.ceil(len(self._data_X)/self._ddsde.autocorrelation_time))
		t_span = np.arange(0, T, t_inc)

		if self._isnotebook():
			pbar = tqdm.tqdm_notebook(total=len(Dt))
		else:
			pbar = tqdm.tqdm(total=len(Dt))
		_g = self.fit("G", order=diff_order, diff_time_scale=dt)
		G = lambda x, t: _g(x)
		m0 = self._data_X[0]
		M = []
		for i in Dt:
			_f = self.fit("F", order=drift_order, drift_time_scale=i)
			F = lambda x, t: _f(x)
			n = sdeint.itoEuler(F, G, m0, t_span)
			M.append(n.flatten())
			pbar.update(1)
		pbar.close()

		M = np.array(M)
		divergence_list = []
		p = self._data_X,copy()
		for q in M:
			divergence_list.append(self._divergence(p, q))
		divergence_list = np.array(divergence_list)
		opt_Dt = divergence_list.argmin() + 1

		fig = plt.figure(dpi=300)
		plt.plot(list(Dt), divergence_list)
		plt.xlabel('Dt')
		plt.ylabel('KL Divergence')
		plt.show()
		print("optimium time scale Dt : {}".format(opt_Dt))
		opt_F = self.fit('F', order=drift_order, drift_time_scale=opt_Dt)
		opt_G = self.fit('G', order=diff_order, diff_time_scale=dt)
		print(opt_F)
		print(opt_G)
		return opt_F, opt_G

	def plot_data(self,
					data_in,
					ax=None,
					clear=False,
					title=None,
					x_label='x',
					y_label='y',
					z_label='z',
					tick_size=12,
					title_size=16,
					label_size=14,
					label_pad=12,
					legend_label=None,
					dpi=150):
		"""
		Plot and visualize vector drift or diffusion data of a 3d axis

		Can be used plot multiple data on the same figure and compare by passing the axis of 
		figure.

		Args
		----
		data_in : numpy.array
			vector drift or diffusion data to plot
		ax : figure axis, (default=None)
			Ia ax is None, a new axis will be created and data will be plotted on it.
		clear : bool, (default=False)
			if True, clear the figure.
		title : str, (default=None)
			title of the figure
		x_label : str, (default='x')
			x-axis label
		y_label : str, (default='y')
			y-axis label
		z_label : str, (default='z')
			z-axis label
		tick_size : int, (default=12)
			axis ticks font size
		title_size : int, (default=16)
			title font size
		label_size : int, (default=14)
			axis label font size
		label_pad : int, (default=12)
			axis label padding
		legend_label : str, (default=None)
			data legend label
		dpi : int, (default=150)
			figure resolution

		Returns
		-------
		ax : 3d figure axis
			axis of the 3d figure.
		fig : matplotlib figure
			returns figure only if the input ax is None.

		"""
		legend = True if legend_label else False
		DataPlot = namedtuple('DataPlot', ('fig', 'ax'))
		fig, ax = self._plot_data(data_in,
									ax=ax,
									title=title,
									x_label=x_label,
									y_label=y_label,
									z_label=z_label,
									clear=clear,
									legend=legend,
									tick_size=tick_size,
									title_size=title_size,
									label_size=label_size,
									label_pad=label_pad,
									label=legend_label,
									dpi=dpi)
		return DataPlot(ax.figure, ax)
		if fig is None:
			return ax
		return ax, fig



	def parameters(self):
		"""
		Get all given and assumed parameters used for the analysis

		Args
		----

		Returns
		-------
		params : dict, json
			all parameters given and assumed used for analysis
		"""
		params = dict()
		for keys in self._ddsde.__dict__.keys():
			if str(keys)[0] != '_':
				params[keys] = str(self._ddsde.__dict__[keys])
		return params


	def fit(self, function_name, order, drift_time_scale=None, diff_time_scale=None):
		"""
		Fit a polynomial or plane to the derrived data

		Args
		----
		function_name : str
			name of the function

			For scalar
			F : drift
			G : diffusion

			For Vector
			A1 : driftX
			A2 : driftY
			B11 : diffusionX
			B22 : diffusionY
			B12 : diffusionXY
			B21 : diffusionYx
		order : int
			order (degree) of the polynomial or plane to fit

		Returns
		-------
		func : poly or plane
			a callable function
			y = poly(x)
			z = plane(x,y)

		Note
		----
			Plane fitting is yet to be implemented
		"""
		fmap = {
		'F'   : 'drift',
		'G'   : 'diff',
		'A1'  : 'driftX',
		'A2'  : 'driftY',
		'B11' : 'diffX',
		'B12' : 'diffXY',
		'B21' : 'diffYX',
		'B22' : 'diffY'
		}
		if function_name not in fmap.keys():
			print("Invalid function name")
			return None

		if self.vector:
			if function_name not in list(fmap.keys())[2:]:
				print("Invalid function name for vector analysis")
				return None
			data = self.data(drift_time_scale=drift_time_scale, diff_time_scale=diff_time_scale)._asdict()

			x,y = np.meshgrid(data['op_x'], data['op_y'])
			z = data[fmap[function_name]]
			plane = self._fit_plane(x=x, y=y, z=z, order=order)
			return plane

		if function_name not in list(fmap.keys())[:2]:
			print("Invalid function name for scalar analysis")
			return None

		data = self.data(drift_time_scale=drift_time_scale, diff_time_scale=diff_time_scale)._asdict()
		if function_name in ['G']:
			y = np.sqrt(data[fmap[function_name]])
		else:
			y = data[fmap[function_name]]
		poly, _ = self._fit_poly(data['op'], y, order)
		#print(poly)
		return poly

	def simulate(self, sigma=4, dt=None, T=None, **functions):
		"""
		Simulate SDE

		Takes drift and diffusion functions as input and
		simuates the SDE using the analysis parameters.

		The drift and diffusion functions given must be callable type functions.

		For scalar F and G (drift and diffusion) must take one input and return a
		number


		Args
		----
		sigma : float
			magnitude of the noise eta(t)

		**functions:
			drift and diffusion callable functions

				For scalar analysis
					F : drift function

					G : diffusion dunction

				For vector analysis
					A1 : drift X

					A2 : drift Y

					B11 : diffusion X

					B22 : diffusion Y

					B12 : diffusion XY

					B21 : diffusion YX

		Returns
		-------
		simulated timeseries : list
			[M] if scalar

			[Mx, My] is vector

		Examples
		--------
		# For scalar analysis
		def drift_function(x):
			return 0.125 * x

		def diffusion_function(x):
			return -(x**2 + 1)

		simulated_data = ddsde.simulate(F=drift_function, G=diffusion_function)

		# For vector analysis
		def drift_x(x, y):
			return x*x + y*y * x*y**2

		def dirft_y(x, y):
			return x*y

		def diffusion_x(x,y):
			return x**2 + x*y

		def diffusion_y(x,y):
			return y**2 + x*y

		def diffusion_xy(x,y):
			return 0

		def diffusion_yx(x,y):
			rerutn 0
		
		simulated_data = ddsde.simulate(A1=drift_x,
						A2=drift_y,
						B11=diffusion_x,
						B22=diffusion_y,
						B12=diffusion_xy.
						B21=diffusion_yx
						)

		"""
		func = {
		'F'   : None,
		'G'   : None,
		'A1'  : None,
		'A2'  : None,
		'B11' : None,
		'B12' : None,
		'B21' : None,
		'B22' : None
		}

		func.update(functions)

		if dt is None: dt = self._ddsde.t_int

		if self.vector:
			for k in ['A1', 'A2', 'B11', 'B12', 'B21', 'B22']:
				if func[k] == None:
					print('Insufficient data, provide {}'.format(k))
					return None
			if T is None: T = len(self._data_Mx) * dt
			n_iter = int(T/dt)
			mx = [self._data_Mx[0]]
			my = [self._data_My[0]]
			for i in tqdm.tqdm(range(n_iter)):
				mx.append(mx[i] + func['A1'](mx[i], my[i])*dt + sigma*np.random.normal()*(func['B11'](mx[i], my[i]) + func['B12'](mx[i], my[i]))*np.sqrt(dt))
				my.append(my[i] + func['A2'](mx[i], my[i])*dt + sigma*np.random.normal()*(func['B22'](mx[i], my[i]) + func['B21'](mx[i], my[i]))*np.sqrt(dt))
			return np.array(mx), np.array(my)

		else:
			for k in ['F', 'G']:
				if func[k] == None:
					print('Insufficient data, provide {}'.format(k))
					return None

			if T is None: T = len(self._data_X) * dt

			n_iter = int(T/dt)

			m = [self._data_X[0]]

			for i in tqdm.tqdm(range(n_iter)):
				m.append(m[i] + func['F'](m[i])*dt + sigma*np.random.normal()*func['G'](m[i])*np.sqrt(dt))

			return np.array(m)



	def summary(self, start=0, end=1000, kde=False, tick_size=12, title_size=15, label_size=15, label_pad=8, n_ticks=3 ,ret_fig=True, **plot_text):
		"""
		Print summary of data and show summary plots chart

		Args
		----
			start : int, (default=0)
				starting index, begin plotting timeseries from this point
			end : int, default=1000
				end point, plots timeseries till this index
			kde : bool, (default=False)
				if True, plot kde for histograms
			title_size : int, (default=15)
				title font size
			tick_size : int, (default=12)
				axis tick size
			label_size : int, (default=15)
				label font size
			label_pad : int, (default=8)
				axis label padding
			n_ticks : int, (default=3)
				number of axis ticks
			ret_fig : bool, (default=True)
				if True return figure object

			**plot_text:
				plots' title and axis texts
				
				For scalar analysis summary plot:
					timeseries_title : title of timeseries plot

					timeseries_xlabel : x label of timeseries

					timeseries_ylabel : y label of timeseries

					drift_title : drift plot title

					drift_xlabel : drift plot x label

					drift_ylabel : drift plot ylabel

					diffusion_title : diffusion plot title

					diffusion_xlabel : diffusion plot x label

					diffusion_ylabel : diffusion plot y label
				
				For vector analysis summary plot:
					timeseries1_title : first timeseries plot title

					timeseries1_ylabel : first timeseries plot ylabel

					timeseries1_xlabel : first timeseries plot xlabel

					timeseries1_legend1 : first timeseries (Mx) legend label
	 
					timeseries1_legend2 : first timeseries (My) legend label

					timeseries2_title : second timeseries plot title

					timeseries2_xlabel : second timeseries plot x label

					timeseries2_ylabel : second timeseries plot y label

					2dhist1_title : Mx 2d histogram title

					2dhist1_xlabel : Mx 2d histogram x label

					2dhist1_ylabel : Mx 2d histogram y label

					2dhist2_title : My 2d histogram title

					2dhist2_xlabel : My 2d histogram x label

					2dhist2_ylabel : My 2d histogram y label

					2dhist3_title :  M 3d histogram title

					2dhist3_xlabel : M 2d histogram x label

					2dhist3_ylabel : M 2d histogram y label

					3dhist_title :  3d histogram title

					3dhist_xlabel : 3d histogram x label

					3dhist_ylabel : 3d histogram y label

					3dhist_zlabel : 3d histogram z label

					driftx_title : drift x plot title

					driftx_xlabel : drift x plot x label

					driftx_ylabel : drift x plot y label

					driftx_zlabel : drift x plot z label

					drifty_title : drift y plot title

					drifty_xlabel : drift y plot x label

					drifty_ylabel : drift y plot y label

					drifty_zlabel : drift y plot z label

					diffusionx_title : diffusion x plot title

					diffusionx_xlabel : diffusion x plot x label

					diffusionx_ylabel : diffusion x plot y label

					diffusionx_zlabel : diffusion x plot z label

					diffusiony_title : diffusion y plot title

					diffusiony_xlabel : diffusion y plot x label

					diffusiony_ylabel : diffusion y plot y label

					diffusiony_zlabel : diffusion y plot z label

		Returns
		-------
			None, or figure

		Raises
		------
		ValueError
			If start is greater than end
		"""
		if start > end:
			raise ValueError("'start' sould not be greater than 'end'")
		
		if not self.vector:
			feilds = [	'M range', 				'M mean', 
						'|M| range', 			'|M| mean', 
						'Autocorr time (M)', 	'(Dt, dt)',
						]
			
			values = [	self._get_data_range(self._data_X),	round(np.nanmean(self._data_X), 3),
						self._get_data_range(np.sqrt(self._data_X**2)), round(np.nanmean(np.sqrt(self._data_X**2)), 3),
						self.autocorrelation_time, (self._ddsde.Dt, self._ddsde.dt),
						]
			values = list(map(str, values))
			summary = []
			for i in range(len(feilds)):
				summary.append(feilds[i])
				summary.append(values[i])
			summary_format = ("| {:<20} : {:^15}"*1 +"|\n")*int(len(feilds)/1)
			print(summary_format.format(*summary))
			data = [self._data_X, self._data_avgdrift, self._data_avgdiff]
		
		else:
			feilds = [	'Mx range', 							'Mx mean', 
						'My range', 							'My mean', 
						'|M| range', 							'|M| mean',
						'Autocorr time (Mx, My, |M|)', 			'(Dt, dt)',
						]
			
			values = [	self._get_data_range(self._data_Mx), round(np.nanmean(self._data_Mx), 3),
						self._get_data_range(self._data_My), round(np.nanmean(self._data_My), 3),									
						self._get_data_range(self._data_M), round(np.nanmean(np.sqrt(self._data_Mx**2 + self._data_My**2)),3),
						(self._ddsde._act_mx, self._ddsde._act_my, self.autocorrelation_time), (self._ddsde.Dt, self._ddsde.dt)
						]
			values = list(map(str, values))
			summary = []
			for i in range(len(feilds)):
				summary.append(feilds[i])
				summary.append(values[i])
			summary_format = ("| {:<30} : {:^15}"*1 +"|\n")*int(len(feilds)/1)
			print("Note: All summary and plots are rounded to third decimal place.\nCalculations, however, are accurate and account for missing values too.\n\n")
			print(summary_format.format(*summary))
			data = [self._data_Mx, self._data_My, self._data_avgdriftX, self._data_avgdriftY, self._data_avgdiffX, self._data_avgdiffY]

		sys.stdout.flush()
		fig = self._plot_summary(data, self.vector, kde=kde, tick_size=tick_size, title_size=title_size, label_size=label_size, label_pad=label_pad, n_ticks=n_ticks, timeseries_start=start, timeseries_end=end, **plot_text)
		plt.show()
		if ret_fig:
			return fig
		return None

	def timeseries(self,
					start=0,
					 end=1000,
					 n_ticks=3,
					 dpi=150,
					 tick_size=12,
					 title_size=14,
					 label_size=14,
					 label_pad=0,
					 **plot_text):
		"""
		Show plot of input data

		Args
		----
		start : int, (default=0)
			starting index, begin plotting timeseries from this point
		end : int, default=1000
			end point, plots timeseries till this index
		n_ticks : int, (default=3)
			number of axis ticks
		dpi : int, (default=150)
			dpi of the figure
		title_size : int, (default=15)
			title font size
		tick_size : int, (default=12)
			axis tick size
		label_size : int, (default=15)
			label font size
		label_pad : int, (default=8)
			axis label padding
		**plot_text:
			plots' title and axis texts

			For scalar analysis plot:
				timeseries_title : title 

				timeseries_xlabel : x label

				timeseries_ylabel : y label

			For vector analysis plot:
				timeseries1_title : first timeseries plot title

				timeseries1_xlabel : first timeseries plot x label

				timeseries1_ylabel : first timeseries plot y lable

				timeseries2_title : second timeseries plot title

				timeseries2_xlabel : second timeseries plot x label

				timeseries2_ylabel : second timeseries plot y label

				timeseries3_title : third timeseries plot title

				timeseries3_xlabel : third timeseries plot x label

				timeseries3_ylabel : third timeseries plot y label

		Returns
		-------
		time series plot : matplotlib.pyplot.figure

		Raises
		------
		ValueError
			If start is greater than end
		"""
		if start > end:
			raise ValueError("'start' sould not be greater than 'end'")
		if self.vector:
			data = [self._data_Mx, self._data_My]
		else:
			data = [self._data_X]
		fig = self._plot_timeseries(data, self.vector,start=start, stop=end, n_ticks=n_ticks, dpi=dpi, tick_size=tick_size, title_size=title_size, label_size=label_size, label_pad=label_pad, **plot_text)
		plt.show()
		return fig

	def histogram(self,	 
					 kde=False,
					 dpi=150,
					 title_size=14,
					 label_size=15,
					 tick_size=12,
					 label_pad=8):
		"""
		Show histogram polt chart

		Args
		----
		kde : bool, (default=False)
			If True, plots kde for histograms
		dpi : int, (defautl=150)
			figure resolution
		title_size : int, (default=14)
			title font size
		label_size : int, (default=15)
			axis label font size
		tick_size : int, (default=12)
			axis ticks font size
		label_pad : int, (default=8)
			axis label padding
		**plot_text:
			plots' axis and title text

			For scalar analysis histograms:
				hist_title : title

				hist_xlabel : x label

				hist_ylabel : y label

			For vector analysis histograms:
				hist1_title : first histogram title

				hist1_xlabel : first histogram x label

				hist1_ylabel : first histogram y label

				hist2_title : second histogram title

				hist2_xlabel : second histogram x label

				hist2_ylabel : second histogram y label

				hist3_title : third histogram title

				hist3_xlabel : third histogram x label

				hist3_ylabel : third histogram y label

				hist4_title : fourth (3d) histogram title
				hist4_xlabel : fourth (3d) histogram x label
				hist4_ylabel : fourth (3d) histogram y label
				hist4_zlabel : fourth (3d) histogram z label

		Returns
		-------
		histogam chart : matplotlib.pyplot.figure
		"""
		if self.vector:
			data = [self._data_Mx, self._data_My]
		else:
			data = [self._data_X]
		fig = self._plot_histograms(data, 
									 self.vector,
									 dpi=dpi,
									 kde=kde,
									 title_size=title_size,
									 label_size=label_size,
									 tick_size=tick_size,
									 label_pad=label_pad)
		plt.show()
		return fig

	def drift(self, polynomial_order=None):
		"""
		Display drift slider figure

		Args
		----
		polynomial_order : None or int, default=None
			order of polynomial to fit, if None, no fitting is done.

		Returns
		-------
		opens drift slider : None
		"""
		dt_s = list(self._drift_slider.keys())
		if not len(dt_s): # empty slider
			return None
		init_pos = np.abs(np.array(dt_s) - self._ddsde.Dt).argmin()
		if self.vector:
			fig = self._slider_3d(self._drift_slider, prefix='Dt', init_pos=init_pos, order=polynomial_order)
		else:
			fig = self._slider_2d(self._drift_slider, prefix='Dt', init_pos=init_pos, polynomial_order=polynomial_order)
		fig.show()
		return None

	def diffusion(self, polynomial_order=None):
		"""
		Display diffusion slider figure

		Args
		----
		polynomial_order : None or int, default=None
			order of polynomial to fit, if None, no fitting is done.
		Returns
		-------
		opens diffusion slider : None
		"""
		dt_s = list(self._diff_slider.keys())
		if not len(dt_s): # empty slider
			return None
		if self.vector:
			fig = self._slider_3d(self._diff_slider, prefix='dt', init_pos=0, order=polynomial_order)
		else:
			fig = self._slider_2d(self._diff_slider, prefix='dt', init_pos=0, polynomial_order=polynomial_order)
		fig.show()
		return None

	def diffusion_cross(self, polynomial_order=None):
		"""
		Display diffusion cross correlation slider figure

		Args
		----
		polynomial_order : None or int, default=None
			order of polynomial to fit, if None, no fitting is done.
		Returns
		-------
		opens diffusion slider : None
		"""
		dt_s = list(self._cross_diff_slider.keys())
		if not len(dt_s): # empty slider
			return None
		if self.vector:
			fig = self._slider_3d(self._cross_diff_slider, prefix='c_dt', init_pos=0, order=polynomial_order)
			fig.show()
		else:
			print('N/A')
		return None


	def visualize(self, drift_time_scale=None, diff_time_scale=None):
		"""
		Display drift and diffusion plots for a time scale.

		Args
		----
		time_scale : int, optional(default=None)
			timescale for which drift and diffusion plots need to be shown.
			If None, displays the plots for inputed timescale.

		Returns
		-------
			displays plots : None
		"""
		if not self.vector:
			drift, _ = self._get_data_from_slider(drift_time_scale)
			_, diff = self._get_data_from_slider(diff_time_scale)

			#Time series
			fig1 = plt.figure(dpi=150)
			plt.suptitle("Time_Series")
			#l = int(len(self._data_X) / 4)
			l = 1000
			try:
				plt.plot(self._data_t[0:l], self._data_X[0:l])
			except:
				plt.plot(self._data_X[0:l])

			#PDF
			fig2 = plt.figure(dpi=150, figsize=(5, 5))
			plt.suptitle("PDF")
			sns.distplot(self._data_X)
			plt.xlim([min(self._data_X), max(self._data_X)])
			plt.xticks(np.linspace(min(self._data_X), max(self._data_X), 5))
			plt.ylabel('PDF')
			plt.xlabel('M')

			#Drift
			fig3 = plt.figure(dpi=150, figsize=(5, 5))
			plt.suptitle("Drift")
			#p_drift, _ = self._fit_poly(self._data_op, drift,
			#							self.drift_order)
			plt.scatter(self._data_op, drift, marker='.')
			"""
			plt.scatter(self._data_op,
						p_drift(self._data_op),
						marker='.',
						alpha=0.4)
			"""
			plt.xlabel('M')
			plt.ylabel("F")
			plt.xlim([min(self._data_op), max(self._data_op)])
			plt.xticks(np.linspace(min(self._data_op), max(self._data_op), 5))

			#Diffusion
			fig4 = plt.figure(dpi=150, figsize=(5, 5))
			plt.suptitle("Diffusion")
			#p_diff, _ = self._fit_poly(self._data_op, diff,
			#						   self.diff_order)
			plt.scatter(self._data_op, diff, marker='.')
			"""
			plt.scatter(self._data_op,
						p_diff(self._data_op),
						marker='.',
						alpha=0.4)
			"""
			plt.xlim([min(self._data_op), max(self._data_op)])
			plt.xticks(np.linspace(min(self._data_op), max(self._data_op), 5))
			plt.xlabel("M")
			plt.ylabel('$G^{2}$')

		else:
			driftX, driftY, _, _, _, _ = self._get_data_from_slider(drift_time_scale)
			_, _, diffX, diffY, diffXY, diffYX = self._get_data_from_slider(diff_time_scale)
			fig1, _ = self._plot_3d_hisogram(self._data_Mx, self._data_My, title='PDF',xlabel="$M_{x}$", tick_size=12, label_size=12, title_size=12, r_fig=True)

			fig5, _ = self._plot_data(driftX,
									  plot_plane=False,
									  title='Drift X',
									  z_label='$A_{1}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)
			"""
			fig5_1, _ = self._plot_data(self._data_avgdriftX,
										title='DriftX_heatmap',
										heatmap=True)
			"""

			fig4, _ = self._plot_data(driftY,
									  plot_plane=False,
									  title='Drift Y',
									  z_label='$A_{2}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)
			"""
			fig4_1, _ = self._plot_data(self._data_avgdriftY,
										title='DriftY_heatmap',
										heatmap=True)
			"""

			fig3, _ = self._plot_data(diffX,
									  plot_plane=False,
									  title='Diffusion X',
									  z_label='$B_{11}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)
			"""
			fig3_1, _ = self._plot_data(self._data_avgdiffX,
										title='DiffX_heatmap',
										heatmap=True)
			"""

			fig2, _ = self._plot_data(diffY,
									  plot_plane=False,
									  title='Diffusion Y',
									  z_label='$B_{22}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)
			"""
			fig2_1, _ = self._plot_data(self._data_avgdiffY,
										title='DiffY_heatmap',
										heatmap=True)
			"""

			fig6, _ = self._plot_data(diffXY,
									  plot_plane=False,
									  title='Diffusion XY',
									  z_label='$B_{12}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)


			fig7, _ = self._plot_data(diffYX,
									  plot_plane=False,
									  title='Diffusion YX',
									  z_label='$B_{21}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)
		return None

	def acf_diagnostic(self):
		"""
		Show autocorrealtion and autocorrelation time calculation plots.

		Args
		----

		Returns
		-------
		displays figures : None
		"""
		exp_fn = lambda t, a, b, c: a * np.exp((-1 / b) * t) + c

		print("Exponential function of the form: ")
		expression = r'a*\exp(\frac{-t}{\tau}) + C'
		ax = plt.axes([0,0,0.1,0.1]) #left,bottom,width,height
		ax.set_xticks([])
		ax.set_yticks([])
		ax.axis('off')
		plt.text(0.4,0.4,'${}$'.format(expression) ,size=20,color="black")
		plt.show()
		print("is fitted. `\\tau` is the autocorrelation time and, `a`, `c` are scalling and fitting parameters respectively.\n")

		if not self.vector:
			x_M, acf_M = self._acf(self._data_X, t_lag=self._ddsde.t_lag)
			[a_M, b_M, c_M], _ = self._fit_exp(x_M, acf_M)
			exp_M = exp_fn(x_M, a_M, b_M, c_M)

			fig1 = plt.figure(dpi=150)
			plt.suptitle("$Autocorrealtion\ M$")
			plt.plot(x_M, acf_M)
			plt.plot(x_M, exp_M)
			plt.legend(('acf', 'exp_fit'))
			plt.xlabel('Time Lag')
			plt.ylabel('$ACF\ (M)$')
			print("acf_M : a = {}, tau = {}, c = {}".format(a_M, b_M, c_M))
			return None

		x_M, acf_M = self._acf(self._data_M**2, t_lag=self._ddsde.t_lag)
		[a_M, b_M, c_M], _ = self._fit_exp(x_M, acf_M)
		exp_M = exp_fn(x_M, a_M, b_M, c_M)

		x_Mx, acf_Mx = self._acf(self._data_Mx, t_lag=self._ddsde.t_lag)
		[a_Mx, b_Mx, c_Mx], _ = self._fit_exp(x_Mx, acf_Mx)
		exp_Mx = exp_fn(x_Mx, a_Mx, b_Mx, c_Mx)

		x_My, acf_My = self._acf(self._data_My, t_lag=self._ddsde.t_lag)
		[a_My, b_My, c_My], _ = self._fit_exp(x_My, acf_My)
		exp_My = exp_fn(x_My, a_My, b_My, c_My)


		fig1 = plt.figure(dpi=150)
		plt.suptitle("$Autocorrealtion\ |M|^{2}$")
		plt.plot(x_M, acf_M)
		plt.plot(x_M, exp_M)
		plt.legend(('acf', 'exp_fit'))
		plt.xlabel('Time Lag')
		plt.ylabel('$ACF\ (|M|^{2})$')
		print("acf_|M|^2 : a = {}, tau = {}, c = {}".format(a_M, b_M, c_M))

		fig2 = plt.figure(dpi=150)
		plt.suptitle("$Autocorrealtion\ M_{x}$")
		plt.plot(x_Mx, acf_Mx)
		plt.plot(x_Mx, exp_Mx)
		plt.legend(('acf', 'exp_fit'))
		plt.xlabel('Time Lag')
		plt.ylabel('$ACF\ (M_{x})$')
		print("acf_M_x : a = {}, tau = {}, c = {}".format(a_Mx, b_Mx, c_Mx))

		fig3 = plt.figure(dpi=150)
		plt.suptitle("$Autocorrealtion\ M_{y}$")
		plt.plot(x_My, acf_My)
		plt.plot(x_My, exp_My)
		plt.legend(('acf', 'exp_fit'))
		plt.xlabel('Time Lag')
		plt.ylabel('$ACF\ (M_{y})$')
		print("acf_My : a = {}, tau = {}, c = {}".format(a_My, b_My, c_My))

		plt.show()
		return None



	def fitting_diagnostic(self):
		"""
		Show diagnostics figures like autocorrelation plots, r2 adjusted plots, for drift and diffusion
		for multiple dt.

		Args
		----

		Returns
		-------
		displays figures : None
		"""

		if self.vector:
			print("N/A")
			return None
		t1 = "R2_adj"
		"""
		#ACF
		fig1 = plt.figure(dpi=150)
		plt.suptitle("ACF")
		exp_fn = lambda t, a, b, c: a * np.exp((-1 / b) * t) + c
		plt.plot(self._ddsde._autocorr_x, self._ddsde._autocorr_y)
		y = exp_fn(self._ddsde._autocorr_x, self._ddsde._a,
				   self._ddsde.autocorrelation_time, self._ddsde._c)
		plt.plot(self._ddsde._autocorr_x, y)
		plt.legend(('ACF', 'exp_fit'))
		plt.xlabel('Time Lag')
		plt.ylabel('ACF')
		"""

		#R2 vs order for drift
		fig2 = plt.figure(dpi=150)
		plt.suptitle("{}_vs_drift_order".format(t1))
		plt.plot(range(self._ddsde.max_order), self._ddsde._r2_drift)
		plt.xlabel('order')
		plt.ylabel(t1)

		#R2 vs order for diff
		fig3 = plt.figure(dpi=150)
		plt.suptitle("{}_vs_Diff_order".format(t1))
		plt.plot(range(self._ddsde.max_order), self._ddsde._r2_diff)
		plt.xlabel('order')
		plt.ylabel(t1)
		#plt.title('{} Diff vs order'.format(t1))

		#R2 vs order for drift, multiple dt
		label = ["dt={}".format(i) for i in self._ddsde._r2_drift_m_dt[-1]]
		fig4 = plt.figure(dpi=150)
		plt.suptitle("{}_Drift_different_dt".format(t1))
		for i in range(len(self._ddsde._r2_drift_m_dt) - 1):
			plt.plot(range(self._ddsde.max_order),
					 self._ddsde._r2_drift_m_dt[i],
					 label=self._ddsde._r2_drift_m_dt[-1][i])
		plt.xlabel('order')
		plt.ylabel(t1)
		plt.legend()

		#R2 vs order for diff, multiple dt
		fig5 = plt.figure(dpi=150)
		plt.suptitle("{}_Diff_different_dt".format(t1))
		for i in range(len(self._ddsde._r2_drift_m_dt) - 1):
			plt.plot(range(self._ddsde.max_order),
					 self._ddsde._r2_diff_m_dt[i],
					 label=self._ddsde._r2_drift_m_dt[-1][i])
		plt.xlabel('order')
		plt.ylabel(t1)
		plt.legend()

		plt.show()

		return None

	def noise_diagnostic(self,
								dpi=150, 
								kde=True, 
								title_size=14, 
								tick_size=15, 
								label_size=15,
								label_pad=8):
		"""
		Show noise characterstics plots.

		Args
		----

		Returns
		-------
		displays plots : None
		"""
		#print("Noise is gaussian") if self._ddsde.gaussian_noise else print("Noise is not Gaussian")
		data = [self._ddsde._noise, self._ddsde._kl_dist, self._ddsde._X1, self._ddsde.h_lim, self._ddsde.k, self._ddsde.l_lim,self._ddsde._f, self._ddsde._noise_correlation]
		fig = self._plot_noise_characterstics(data,
												dpi=150, 
												kde=True, 
												title_size=14, 
												tick_size=15, 
												label_size=15,
												label_pad=8)

		plt.show()
		return None


class Error(Exception):
	"""
	Base class for exceptions in this module.
	
	:meta private:
	"""
	pass

class PathNotFound(Error):
	"""
	pass

	:meta private:
	"""
	def __init__(self, full_path, message):
		self.full_path = full_path
		self.message = message
