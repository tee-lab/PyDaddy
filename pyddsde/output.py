import numpy as np
import sys
import os
import gc
import json
import numpy.matlib
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

		if not self.vector:
			self._data_X = ddsde._X
			self._data_t = ddsde._t
			self._data_drift = ddsde._drift_
			self._data_diff = ddsde._diff_
			self._data_avgdrift = ddsde._avgdrift_
			self._data_avgdiff = ddsde._avgdiff_
			self._data_op = ddsde._op_
			self.drift_order = ddsde.drift_order
			self.diff_order = ddsde.diff_order

			visualize.__init__(self, None, None, self._data_op,
							   self._ddsde.autocorrelation_time, drift_order=self.drift_order, diff_order=self.diff_order)
		else:
			self._data_Mx = ddsde._Mx
			self._data_My = ddsde._My
			self._data_M = np.sqrt(self._data_Mx**2 + self._data_My**2)
			self._data_avgdriftX = ddsde._avgdriftX_
			self._data_avgdriftY = ddsde._avgdriftY_
			self._data_avgdiffX = ddsde._avgdiffX_
			self._data_avgdiffY = ddsde._avgdiffY_
			self._data_avgdiffXY = ddsde._avgdiffXY_
			self._data_op_x = ddsde._op_x_
			self._data_op_y = ddsde._op_y_


			#self._drift_slider = ddsde._drift_slider
			#self._diff_slider = ddsde._diff_slider

			visualize.__init__(self, self._data_op_x, self._data_op_y, None,
							   self._ddsde.autocorrelation_time)

		self._drift_slider = ddsde._drift_slider
		self._diff_slider = ddsde._diff_slider
		self._time_scale_list = list(self._drift_slider.keys())

		self.res_dir = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())


		self.__dict__.update(kwargs)
		preprocessing.__init__(self)

		return self.summary(ret_fig=False)

	def release(self):
		"""
		Clears the memory, recommended to be used while analysing multiple
		data files in loop.

		returns
		-------
			None
		"""
		self._visualize_figs = []
		self._diagnostics_figs = []
		self._noise_figs = []
		plt.close('all')
		gc.collect()
		return None

	"""
	def _make_directory(self, p, i=1):
		""
		Recursively create directories in given path

		Args
		----
		p : str
			destination path

		returns
		-------
		path : str
			same as input p
		""
		if type(p) != list: p = p.split('/')
		if i > len(p):
			return os.path.join(*p)
		try:
			os.mkdir(os.path.join(*p[0:i]))
		except FileExistsError:
			pass
		return self._make_directory(p, i=i + 1)
	"""

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

		returns
		-------
		str
			path where data is exported
		"""
		if dir_path is None:
			self.res_dir = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
			dir_path = self._make_directory(os.path.join('results', self.res_dir))
			self.res_dir = dir_path

		if not os.path.exists(dir_path):
			raise PathNotFound(dir_path, "Entered directory path does not exists.")

		data_dict = self._get_stacked_data()
		for key in data_dict:
			self._save_csv(dir_path=dir_path, file_name=key, data=data_dict[key], fmt='%.4f', add_headers=True)

		if save_mat:
			savedict = self._combined_data_dict()
			scipy.io.savemat(os.path.join(dir_path, 'drift_diff_data.mat'), savedict)

		if zip:
			self._zip_dir(dir_path)

		return "Exported to {}".format(self.res_dir)


	def data(self, time_scale=None):
		"""
		Get the drift, diffusion and order parameter data for any timescale the analysis is done.

		Args
		----
		time_scale : int, optional(default=None)
			time_scale corresponding to the data, if None, returns data for analysed given dt

		returns
		-------
		list
			- if vector, [avgdriftX, avgdriftY, avgdiffX, avgdiffY, op_x, op_y] 
			- else, [avgdrift, avgdiff, op]
		"""
		if not self.vector:
			drift , diff = self._get_data_from_slider(time_scale)
			return drift, diff, self._data_op

		driftX, driftY, diffX, diffY = self._get_data_from_slider(time_scale)
		return driftX, driftY, diffX, diffY, self._data_op_x, self._data_op_y

		"""
	def save_data(self, file_name='data', savepath='results', savemat=True):
		""
		Save calculated data to file

		Input params:
		--------------
		file_name : str
			name of the file, if None, file name will be the time a which the data was analysed
		savepath : str
			destination path to save data, if None, files will me saved in current/working/directory/results
		savemat : bool
			if True also saves the data in matlab compatable (.mat) format.

		returns:
		-------------
			None
		""
		if file_name is None: file_name = self.res_dir
		savepath = self._make_directory(os.path.join(savepath, self.res_dir))
		if not self.vector:
			data_dict = {
				'drift': self._data_drift,
				'diff': self._data_diff,
				'avgdrift': self._data_avgdrift,
				'avgdiff': self._data_avgdiff,
				'op': self._data_op
			}
		else:
			x, y = np.meshgrid(self._data_op_x, self._data_op_y)
			data_dict = {
				'avgdriftX': self._data_avgdriftX,
				'avgdriftY': self._data_avgdriftY,
				'avgdiffX': self._data_avgdiffX,
				'avgdiffY': self._data_avgdiffY,
				'avgdiffXY': self._data_avgdiffXY,
				'op_x': self._data_op_x,
				'op_y': self._data_op_y,
				'x': x,
				'y': y
			}
		with open(os.path.join(savepath, file_name + '.pkl'), 'wb') as file:
			pickle.dump(data_dict, file)
		if savemat:
			scipy.io.savemat(os.path.join(savepath, file_name + '.mat'),
							 mdict=data_dict)

		return None
		"""

		"""
	def save_all_data(self, savepath='results', file_name='data'):
		""
		Saves all data and figures

		Input params:
		--------------
		show : bool
			if True, shows the figure
		file_name : str
			name of the files, if None, time at which data was analysed is consideres as file name
		savepath : str
			save destination path, if None, data is saved in current/working/directory/results

		returns:
		-------------
			None
		""
		self.save_data(file_name=file_name, savepath=savepath)
		self.parameters(save=True, savepath=savepath)
		self.visualize(show=False, save=True, savepath=savepath)
		self.diagnostic(show=False, save=True, savepath=savepath)
		self.noise_characterstics(show=False, save=True, savepath=savepath)
		self.slices_2d(show=False, save=True, savepath=savepath)
		print('Results saved in: {}'.format(
			os.path.join(savepath, self.res_dir)))
		"""

	def parameters(self):
		"""
		Get all given and assumed parameters used for the analysis

		Args
		----

		returns
		-------
		params : dict, json
			all parameters given and assumed used for analysis
		"""
		params = dict()
		for keys in self._ddsde.__dict__.keys():
			if str(keys)[0] != '_':
				params[keys] = str(self._ddsde.__dict__[keys])
		return params

	def summary(self, ret_fig=True):
		"""
		Print summary of data and show summary plots chart

		Args
		----
			ret_fig : bool, optional(default=True)
				if True return figure object

		returns
		-------
			None, or figure
		"""
		if not self.vector:
			feilds = ['Data Type (vector)', 'Autocorrelation Time', 'Gaussian Noise', 'M range', 'M mean', '|M| mean']
			values = [self.vector, self.autocorrelation_time,self._ddsde.gaussian_noise,(round(min(self._data_X), 2), round(max(self._data_X), 2)),	round(np.mean(self._data_X), 2),round(np.mean(np.sqrt(self._data_X**2)), 2)]
			values = list(map(str, values))
			summary = []
			for i in range(len(feilds)):
				summary.append(feilds[i])
				summary.append(values[i])
			summary_format = ("| {:<20} : {:^15}"*2 +"|\n")*int(len(feilds)/2)
			print(summary_format.format(*summary))
			print("Dt = {}\ndt= {}".format(self._ddsde.dt, self._ddsde.delta_t))
			data = [self._data_X, self._data_avgdrift, self._data_avgdiff, self.drift_order, self.diff_order]
		else:
			feilds = ['Data Type (vector)', 'Autocorrelation Time', 'Gaussian Noise', 'Mx range', 'My range', 'range |M|', 'Mx mean', 'My mean', 'M mean', '(dt, delta_t)']
			values = [self.vector, self.autocorrelation_time, self._ddsde.gaussian_noise, (round(min(self._data_Mx), 2), round(max(self._data_Mx), 2)),	(round(min(self._data_My), 2), round(max(self._data_My), 2)), (round(min(self._data_M), 2), round(max(self._data_M), 2)), round(np.mean(self._data_Mx), 2), round(np.mean(self._data_My), 2), round(np.mean(np.sqrt(self._data_Mx**2 + self._data_My**2)),2), (self._ddsde.dt, self._ddsde.delta_t)]
			values = list(map(str, values))
			summary = []
			for i in range(len(feilds)):
				summary.append(feilds[i])
				summary.append(values[i])
			summary_format = ("| {:<20} : {:^15}"*2 +"|\n")*int(len(feilds)/2)
			print(summary_format.format(*summary))
			data = [self._data_Mx, self._data_My, self._data_avgdriftX, self._data_avgdriftY, self._data_avgdiffX, self._data_avgdiffY]

		sys.stdout.flush()
		fig = self._plot_summary(data, self.vector)
		plt.show()
		if ret_fig:
			return fig
		return None

	def timeseries(self):
		"""
		Show plot of input data

		Args
		----

		returns
		-------
		time series plot : matplotlib.pyplot.figure
		"""
		if self.vector:
			fig = self._plot_timeseries([self._data_Mx, self._data_My],
										self.vector)
		else:
			fig = self._plot_timeseries([self._data_X], self.vector)
		plt.show()
		return fig

	def histogram(self):
		"""
		Show histogram polt chart

		Args
		----

		returns
		-------
		histogam chart : matplotlib.pyplot.figure
		"""
		if self.vector:
			fig = self._plot_histograms([self._data_Mx, self._data_My],
										self.vector)
		else:
			fig = self._plot_histograms([self._data_X], self.vector)
		plt.show()
		return fig

	def drift(self):
		"""
		Display drift slider figure

		Args
		----

		returns
		-------
		opens drift slider : None
		"""
		dt_s = list(self._drift_slider.keys())
		if not len(dt_s): # empty slider
			return None
		init_pos = np.abs(np.array(dt_s) - self._ddsde.dt).argmin()
		if self.vector:
			fig = self._slider_3d(self._drift_slider, prefix='Dt', init_pos=init_pos)
		else:
			fig = self._slider_2d(self._drift_slider, prefix='Dt', init_pos=init_pos)
		fig.show()
		return None

	def diffusion(self):
		"""
		Display diffusion slider figure

		Args
		----

		returns
		-------
		opens diffusion slider : None
		"""
		dt_s = list(self._diff_slider.keys())
		if not len(dt_s): # empty slider
			return None
		if self.vector:
			fig = self._slider_3d(self._diff_slider, prefix='dt', init_pos=0)
		else:
			fig = self._slider_2d(self._diff_slider, prefix='dt', init_pos=0)
		fig.show()
		return None

	def visualize(self, time_scale=None):
		"""
		Display drift and diffusion plots for a time scale.

		Args
		----
		time_scale : int, optional(default=None)
			timescale for which drift and diffusion plots need to be shown.
			If None, displays the plots for inputed timescale.

		returns
		-------
			displays plots : None
		"""
		self._visualize_figs = []
		if not self.vector:
			drift, diff = self._get_data_from_slider(time_scale)
			#Time series
			fig1 = fig = plt.figure(dpi=150)
			plt.suptitle("Time_Series")
			l = int(len(self._data_X) / 4)
			try:
				plt.plot(self._data_t[0:l], self._data_X[0:l])
			except:
				plt.plot(self._data_X[0:l])
			self._visualize_figs.append(fig1)

			#PDF
			fig2 = fig = plt.figure(dpi=150, figsize=(5, 5))
			plt.suptitle("PDF")
			sns.distplot(self._data_X)
			plt.xlim([min(self._data_X), max(self._data_X)])
			plt.ylabel('PDF')
			plt.xlabel('Order Parameter')
			self._visualize_figs.append(fig2)

			#Drift
			fig3 = plt.figure(dpi=150, figsize=(5, 5))
			plt.suptitle("Average_Drift")
			p_drift, _ = self._fit_poly(self._data_op, drift,
										self.drift_order)
			plt.scatter(self._data_op, drift, marker='.')
			plt.scatter(self._data_op,
						p_drift(self._data_op),
						marker='.',
						alpha=0.4)
			plt.xlabel('Order Parameter')
			plt.ylabel("Deterministic")
			plt.xlim([min(self._data_X), max(self._data_X)])
			self._visualize_figs.append(fig3)

			#Diffusion
			fig4 = plt.figure(dpi=150, figsize=(5, 5))
			plt.suptitle("Average_Diffusion")
			p_diff, _ = self._fit_poly(self._data_op, diff,
									   self.diff_order)
			plt.scatter(self._data_op, diff, marker='.')
			plt.scatter(self._data_op,
						p_diff(self._data_op),
						marker='.',
						alpha=0.4)
			plt.xlim([min(self._data_X), max(self._data_X)])
			plt.xlabel("Order Parameter")
			plt.ylabel('Stochastic')
			self._visualize_figs.append(fig4)

		else:
			driftX, driftY, diffX, diffY = self._get_data_from_slider(time_scale)
			fig1, _ = self._plot_3d_hisogram(self._data_Mx, self._data_My, title='PDF',xlabel="$M_{x}$", tick_size=12, label_size=12, title_size=12, r_fig=True)
			self._visualize_figs.append(fig1)

			fig2, _ = self.plot_data(diffY,
									  plot_plane=False,
									  title='DiffY',
									  z_label='$B_{22}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)
			self._visualize_figs.append(fig2)
			"""
			fig2_1, _ = self.plot_data(self._data_avgdiffY,
										title='DiffY_heatmap',
										heatmap=True)
			self._visualize_figs.append(fig2_1)
			"""

			fig3, _ = self.plot_data(diffX,
									  plot_plane=False,
									  title='DiffX',
									  z_label='$B_{11}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)
			self._visualize_figs.append(fig3)
			"""
			fig3_1, _ = self.plot_data(self._data_avgdiffX,
										title='DiffX_heatmap',
										heatmap=True)
			self._visualize_figs.append(fig3_1)
			"""

			fig4, _ = self.plot_data(driftY,
									  plot_plane=False,
									  title='DriftY',
									  z_label='$A_{2}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)
			self._visualize_figs.append(fig4)
			"""
			fig4_1, _ = self.plot_data(self._data_avgdriftY,
										title='DriftY_heatmap',
										heatmap=True)
			self._visualize_figs.append(fig4_1)
			"""

			fig5, _ = self.plot_data(driftX,
									  plot_plane=False,
									  title='DriftX',
									  z_label='$A_{1}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)
			self._visualize_figs.append(fig5)
			"""
			fig5_1, _ = self.plot_data(self._data_avgdriftX,
										title='DriftX_heatmap',
										heatmap=True)
			self._visualize_figs.append(fig5_1)
			"""


		return None

	def diagnostic(self):
		"""
		Show diagnostics figures like autocorrelation plots, r2 adjusted plots, for drift and diffusion
		for multiple dt.

		Args
		----

		returns
		-------
		displays figures : None
		"""
		self._diagnostics_figs = []
		t1 = "R2" if self._ddsde.order_metric == "R2" else "R2_adj"
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
		self._diagnostics_figs.append(fig1)

		#R2 vs order for drift
		fig2 = plt.figure(dpi=150)
		plt.suptitle("{}_vs_drift_order".format(t1))
		plt.plot(range(self._ddsde.max_order), self._ddsde._r2_drift)
		plt.xlabel('order')
		plt.ylabel(t1)
		self._diagnostics_figs.append(fig2)

		#R2 vs order for diff
		fig3 = plt.figure(dpi=150)
		plt.suptitle("{}_vs_Diff_order".format(t1))
		plt.plot(range(self._ddsde.max_order), self._ddsde._r2_diff)
		plt.xlabel('order')
		plt.ylabel(t1)
		#plt.title('{} Diff vs order'.format(t1))
		self._diagnostics_figs.append(fig3)

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
		self._diagnostics_figs.append(fig4)

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
		self._diagnostics_figs.append(fig5)

		plt.show()

		return None

	def noise_characterstics(self):
		"""
		Show noise characterstics plots.

		Args
		----

		returns
		-------
		displays plots : None
		"""
		self._noise_figs = []
		#print("Noise is gaussian") if self._ddsde.gaussian_noise else print("Noise is not Gaussian")
		data = [self._ddsde._noise, self._ddsde._kl_dist, self._ddsde._X1, self._ddsde.h_lim, self._ddsde.k, self._ddsde.l_lim,self._ddsde._f, self._ddsde._noise_correlation]
		fig = self._plot_noise_characterstics(data)

		plt.show()
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

	def __str__(self):
		return self.message

class PathNotFound(Error):
	"""
	pass

    :meta private:
	"""
	def __init__(self, full_path, message):
		self.full_path = full_path
		self.message = message

	def __str__(self):
		return self.message