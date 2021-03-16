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
		self._visualize_figs = []
		self._diagnostics_figs = []
		self._noise_figs = []
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

		return "Exported to {}".format(os.path.join(os.getcwd(),self.res_dir))


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

		Data = namedtuple('Data', ('driftX', 'driftY', 'diffX', 'diffY', 'op_x', 'op_y'))
		driftX, driftY, _, _ = self._get_data_from_slider(drift_time_scale)
		_, _, diffX, diffY = self._get_data_from_slider(diff_time_scale)
		return Data(driftX, driftY, diffX, diffY, self._data_op_x, self._data_op_y)

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

	def summary(self, start=0, end=1000, kde=False, tick_size=12, title_size=15, label_size=15, label_pad=8, n_ticks=3 ,ret_fig=True):
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
			
			values = [	self._get_data_range(self._data_X),	round(np.nanmean(self._data_X), 2),
						self._get_data_range(np.sqrt(self._data_X**2)), round(np.nanmean(np.sqrt(self._data_X**2)), 2),
						np.ceil(self.autocorrelation_time), (self._ddsde.dt, self._ddsde.delta_t),
						]
			values = list(map(str, values))
			summary = []
			for i in range(len(feilds)):
				summary.append(feilds[i])
				summary.append(values[i])
			summary_format = ("| {:<20} : {:^15}"*2 +"|\n")*int(len(feilds)/2)
			print(summary_format.format(*summary))
			data = [self._data_X, self._data_avgdrift, self._data_avgdiff, self.drift_order, self.diff_order]
		
		else:
			feilds = [	'Mx range', 							'Mx mean', 
						'My range', 							'My mean', 
						'|M| range', 							'|M| mean',
						'Autocorr time (Mx, My, |M|)', 			'(Dt, dt)',
						]
			
			values = [	self._get_data_range(self._data_Mx), round(np.nanmean(self._data_Mx), 2),
						self._get_data_range(self._data_My), round(np.nanmean(self._data_My), 2),									
						self._get_data_range(self._data_M), round(np.nanmean(np.sqrt(self._data_Mx**2 + self._data_My**2)),2),
						(self._act(self._data_Mx), self._act(self._data_My), np.ceil(self.autocorrelation_time)), (self._ddsde.dt, self._ddsde.delta_t)
						]
			values = list(map(str, values))
			summary = []
			for i in range(len(feilds)):
				summary.append(feilds[i])
				summary.append(values[i])
			summary_format = ("| {:<30} : {:^15}"*2 +"|\n")*int(len(feilds)/2)
			print(summary_format.format(*summary))
			data = [self._data_Mx, self._data_My, self._data_avgdriftX, self._data_avgdriftY, self._data_avgdiffX, self._data_avgdiffY]

		sys.stdout.flush()
		fig = self._plot_summary(data, self.vector, kde=kde, tick_size=tick_size, title_size=title_size, label_size=label_size, label_pad=label_pad, n_ticks=n_ticks, timeseries_start=start, timeseries_end=end)
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
					 label_pad=0):
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
		fig = self._plot_timeseries(data, self.vector,start=start, stop=end, n_ticks=n_ticks, dpi=dpi, tick_size=tick_size, title_size=title_size, label_size=label_size, label_pad=label_pad)
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

	def drift(self, fit_poly=False, order=None):
		"""
		Display drift slider figure

		Args
		----
		fit_poly : bool, default=False
			If True fits a ploynomial for drift data, works only for scalar data analysis
		order : None or int, default=None
			order of polynomial to fit, if None, fits a polynomial of observed order of drift.

		Returns
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
			fig = self._slider_2d(self._drift_slider, prefix='Dt', init_pos=init_pos, fit_poly=fit_poly, order=order)
		fig.show()
		return None

	def diffusion(self, fit_poly=False, order=None):
		"""
		Display diffusion slider figure

		Args
		----
		fit_poly : bool, default=False
			If True fits a ploynomial for diffusion data, works only for scalar data analysis
		order : None or int, default=None
			order of polynomial to fit, if None, fits a polynomial of observed order of diffusion.
		Returns
		-------
		opens diffusion slider : None
		"""
		dt_s = list(self._diff_slider.keys())
		if not len(dt_s): # empty slider
			return None
		if self.vector:
			fig = self._slider_3d(self._diff_slider, prefix='dt', init_pos=0)
		else:
			fig = self._slider_2d(self._diff_slider, prefix='dt', init_pos=0, fit_poly=fit_poly, order=order)
		fig.show()
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
		self._visualize_figs = []
		if not self.vector:
			drift, _ = self._get_data_from_slider(drift_time_scale)
			_, diff = self._get_data_from_slider(diff_time_scale)

			#Time series
			fig1 = fig = plt.figure(dpi=150)
			plt.suptitle("Time_Series")
			#l = int(len(self._data_X) / 4)
			l = 1000
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
			driftX, driftY, _, _ = self._get_data_from_slider(drift_time_scale)
			_, _, diffX, diffY = self._get_data_from_slider(diff_time_scale)
			fig1, _ = self._plot_3d_hisogram(self._data_Mx, self._data_My, title='PDF',xlabel="$M_{x}$", tick_size=12, label_size=12, title_size=12, r_fig=True)
			self._visualize_figs.append(fig1)

			fig2, _ = self._plot_data(diffY,
									  plot_plane=False,
									  title='DiffY',
									  z_label='$B_{22}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)
			self._visualize_figs.append(fig2)
			"""
			fig2_1, _ = self._plot_data(self._data_avgdiffY,
										title='DiffY_heatmap',
										heatmap=True)
			self._visualize_figs.append(fig2_1)
			"""

			fig3, _ = self._plot_data(diffX,
									  plot_plane=False,
									  title='DiffX',
									  z_label='$B_{11}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)
			self._visualize_figs.append(fig3)
			"""
			fig3_1, _ = self._plot_data(self._data_avgdiffX,
										title='DiffX_heatmap',
										heatmap=True)
			self._visualize_figs.append(fig3_1)
			"""

			fig4, _ = self._plot_data(driftY,
									  plot_plane=False,
									  title='DriftY',
									  z_label='$A_{2}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)
			self._visualize_figs.append(fig4)
			"""
			fig4_1, _ = self._plot_data(self._data_avgdriftY,
										title='DriftY_heatmap',
										heatmap=True)
			self._visualize_figs.append(fig4_1)
			"""

			fig5, _ = self._plot_data(driftX,
									  plot_plane=False,
									  title='DriftX',
									  z_label='$A_{1}(m)$',
									  tick_size=12,
									  label_size=14,
									  title_size=16)
			self._visualize_figs.append(fig5)
			"""
			fig5_1, _ = self._plot_data(self._data_avgdriftX,
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

		Returns
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

		Returns
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
