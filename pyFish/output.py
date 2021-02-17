import numpy as np
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
from pyFish.preprocessing import preprocessing
from pyFish.visualize import visualize


class output(preprocessing, visualize):
	"""
	Class to plot and save data and parameters
	"""
	def __init__(self, ddsde, **kwargs):
		self.vector = ddsde.vector
		self._ddsde = ddsde

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

			self.res_dir = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())

			#self._drift_slider = ddsde._drift_slider
			#self._diff_slider = ddsde._diff_slider

			visualize.__init__(self, self._data_op_x, self._data_op_y, None,
							   self._ddsde.autocorrelation_time)

		self._drift_slider = ddsde._drift_slider
		self._diff_slider = ddsde._diff_slider
		self._time_scale_list = list(self._drift_slider.keys())
		self.__dict__.update(kwargs)
		preprocessing.__init__(self)

		return self.summary(r=False)

	def release(self):
		"""
		Clears the memory, recommended to be used while analysing multiple
		data files in loop.

		Input params:
		--------------
		None

		returns:
		--------------
		None
		"""
		self._visualize_figs = []
		self._diagnostics_figs = []
		self._noise_figs = []
		plt.close('all')
		gc.collect()
		return None

	def _make_directory(self, p, i=1):
		"""
		Recursively create directories in given path

		Input params:
		--------------
		path : str
			destination path

		returns:
		-------------
		path : str
		"""
		if type(p) != list: p = p.split('/')
		if i > len(p):
			return os.path.join(*p)
		try:
			os.mkdir(os.path.join(*p[0:i]))
		except FileExistsError:
			pass
		return self._make_directory(p, i=i + 1)

	def export_data(self, dir_path=None, include_mat=True, zip=False):
		if dir_path is None:
			self.res_dir = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
			dir_path = self._make_directory(os.path.join('results', self.res_dir))
			self.res_dir = dir_path

		if not os.path.exists(dir_path):
			raise PathNotFound(dir_path, "Entered directory path does not exists.")

		data_dict = self._get_stacked_data()
		for key in data_dict:
			self._save_csv(dir_path=dir_path, file_name=key, data=data_dict[key], fmt='%.4f', add_headers=True)

		if include_mat:
			savedict = self._combined_data_dict()
			scipy.io.savemat(os.path.join(dir_path, 'drift_diff_data.mat'), savedict)

		if zip:
			self._zip_dir(dir_path)

		return "Exported to {}".format(self.res_dir)


	def data(self):
		"""
		Get the calculated data

		Input params:
		--------------
		None

		returns:
		--------------
		data : list
			if vector [drift, diff, avgdrift, avgdiff, op]
			else, [avgdriftX, avgdriftY, avgdiffX, avgdiffY, avgdiffXY, op_x, op_y] 
		"""
		if not self.vector:
			return self._data_drift, self._data_diff, self._data_avgdrift, self._data_avgdiff, self._data_op
		return self._data_avgdriftX, self._data_avgdriftY, self._data_avgdiffX, self._data_avgdiffY, self._data_avgdiffXY, self._data_op_x, self._data_op_y

	def save_data(self, file_name='data', savepath='results', savemat=True):
		"""
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
		"""
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

	def save_all_data(self, savepath='results', file_name='data'):
		"""
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
		"""
		self.save_data(file_name=file_name, savepath=savepath)
		self.parameters(save=True, savepath=savepath)
		self.visualize(show=False, save=True, savepath=savepath)
		self.diagnostic(show=False, save=True, savepath=savepath)
		self.noise_characterstics(show=False, save=True, savepath=savepath)
		self.slices_2d(show=False, save=True, savepath=savepath)
		print('Results saved in: {}'.format(
			os.path.join(savepath, self.res_dir)))

	def parameters(self,
				   save=False,
				   savepath=None,
				   file_name="parameters.txt"):
		"""
		Get the parameters used and calculated for analysis

		Input params:
		--------------
		save : bool
			if True, save parameters to file
		savepath : str
			save destination path, if None, data is saved in current/working/directory/results
		file_name = 'parameters.txt' : str
			name of the file

		returns:
		------------
		params : dict, json
			all parameters used and calculated during analysis
		"""
		if savepath is None: savepath = "results"
		params = dict()
		for keys in self._ddsde.__dict__.keys():
			if str(keys)[0] != '_':
				params[keys] = str(self._ddsde.__dict__[keys])
		if save:
			savepath = self._make_directory(os.path.join(
				savepath, self.res_dir))
			with open(os.path.join(savepath, file_name), 'w',
					  encoding='utf-8') as f:
				json.dump(params, f, indent=True, separators='\n:')
		return params

	def summary(self, r=True):
		if not self.vector:
			feilds = ['Data Type (vector)', 'Autocorrelation Time', 'Gaussian Noise', 'M range', 'M mean', '|M| mean']
			values = [self.vector, self.autocorrelation_time,self._ddsde.gaussian_noise,(round(min(self._data_X), 2), round(max(self._data_X), 2)),	round(np.mean(self._data_X), 2),round(np.mean(np.sqrt(self._data_X**2)), 2)]
			values = list(map(str, values))
			summary = []
			for i in range(len(feilds)):
				summary.append(feilds[i])
				summary.append(values[i])
			summary_format = ("| {:<20} : {:^15}"*3 +"|\n")*int(len(feilds)/3)
			print(summary_format.format(*summary))
			print("Dt = {}\ndt= {}".format(self._ddsde.dt, self._ddsde.delta_t))
			data = [self._data_X, self._data_avgdrift, self._data_avgdiff, self.drift_order, self.diff_order]
		else:
			feilds = ['Data Type (vector)', 'Autocorrelation Time', 'Gaussian Noise', 'Mx range', 'My range', 'range |M|', 'Mx mean', 'My mean', 'M mean']
			values = [self.vector, self.autocorrelation_time, self._ddsde.gaussian_noise, (round(min(self._data_Mx), 2), round(max(self._data_Mx), 2)),	(round(min(self._data_My), 2), round(max(self._data_My), 2)), (round(min(self._data_M), 2), round(max(self._data_M), 2)), round(np.mean(self._data_Mx), 2), round(np.mean(self._data_My), 2), round(np.mean(np.sqrt(self._data_Mx**2 + self._data_My**2)),2)]
			values = list(map(str, values))
			summary = []
			for i in range(len(feilds)):
				summary.append(feilds[i])
				summary.append(values[i])
			summary_format = ("| {:<20} : {:^15}"*3 +"|\n")*int(len(feilds)/3)
			print(summary_format.format(*summary))
			print("Dt = {}\ndt= {}".format(self._ddsde.dt, self._ddsde.delta_t))
			data = [self._data_Mx, self._data_My, self._data_avgdriftX, self._data_avgdriftY, self._data_avgdiffX, self._data_avgdiffY]

		fig = self._plot_summary(data, self.vector)
		plt.show()
		if r:
			return fig
		return None

	def timeseries(self):
		if self.vector:
			fig = self._plot_timeseries([self._data_Mx, self._data_My],
										self.vector)
		else:
			fig = self._plot_timeseries([self._data_X], self.vector)
		fig.show()
		return fig

	def histogram(self):
		if self.vector:
			fig = self._plot_histograms([self._data_Mx, self._data_My],
										self.vector)
		else:
			fig = self._plot_histograms([self._data_X], self.vector)
		fig.show()
		return fig

	def drift(self):
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
		dt_s = list(self._diff_slider.keys())
		if not len(dt_s): # empty slider
			return None
		if self.vector:
			fig = self._slider_3d(self._diff_slider, prefix='dt', init_pos=0)
		else:
			fig = self._slider_2d(self._diff_slider, prefix='dt', init_pos=0)
		fig.show()
		return None

	def visualize(self, time_scale=None, show=True, save=False, savepath='results'):
		"""
		Plot the data

		Input params:
		--------------
		show = True : bool
			if True, show the figures 
		save = False : bool
			if True save the figures to disk
		savepath = None : str
			save destination path, if None, data is saved in current/working/directory/results/visualize

		returns:
		-------------
			None
		"""
		self._visualize_figs = []
		if not self.vector:
			drift, diff = self._get_data_from_slider(time_scale)
			savepath = os.path.join(savepath, self.res_dir, 'visualize')
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
			savepath = os.path.join(savepath, self.res_dir, 'visualize','plot_3d')
			driftX, driftY, diffX, diffY = self._get_data_from_slider(time_scale)
			fig1, _ = self._plot_3d_hisogram(self._data_Mx, self._data_My, title='PDF',xlabel="$M_{x}$", tick_size=12, label_size=12, title_size=12, r_fig=True)
			self._visualize_figs.append(fig1)

			"""
			num_ticks = 5
			fig1 = plt.figure()
			plt.suptitle("PDF")
			ax = fig1.add_subplot(projection="3d")
			H, edges, X, Y, Z, dx, dy, dz = self._histogram3d(
				self._remove_nan(self._data_Mx, self._data_My))
			colors = plt.cm.coolwarm(dz.flatten() / float(dz.max()))
			hist3d = ax.bar3d(X,
							  Y,
							  Z,
							  dx,
							  dy,
							  dz,
							  alpha=0.6,
							  cmap=plt.cm.coolwarm,
							  color=colors)
			ax.set_xlabel('Mx', fontsize=16, labelpad=10)
			ax.set_ylabel('My', fontsize=16, labelpad=10)
			ax.set_zlabel('Frequency', fontsize=16, labelpad=12)
			# make the panes transparent
			ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
			ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
			ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
			# make the grid lines transparent
			ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
			ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
			ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
			#Set ticks lable and its fontsize
			ax.tick_params(axis='both', which='major', labelsize=16)
			ax.set_xticks(np.linspace(-1, 1, 5))
			ax.set_yticks(np.linspace(-1, 1, 5))
			"""
			#H, edges, X, Y, Z, dx, dy, dz = self._histogram3d(self._remove_nan(self._data_Mx, self._data_My))
			#self._hist_data = (X, Y, Z, dx, dy, dz)
			"""
			fig1_1, ax  = plt.subplots()
			plt.suptitle("PDF_heatmap", verticalalignment='center', ha='right')
			ticks = self._data_op_x.copy()
			ticks_loc = np.linspace(0, len(ticks), num_ticks)
			ticks = np.linspace(min(ticks), max(ticks), num_ticks).round(2)
			bin_count = int(np.sqrt(len(dz)))
			dz = dz.reshape((bin_count, bin_count))
			ax = sns.heatmap(
				dz,
				xticklabels=ticks,
				yticklabels=ticks[::-1],
				cmap=plt.cm.coolwarm,
			)
			ax.set_xlabel('$m_x$', fontsize=16, labelpad=10)
			ax.set_ylabel('$m_y$', fontsize=16, labelpad=10)
			ax.set_xticks(ticks_loc)
			ax.set_yticks(ticks_loc)
			ax.tick_params(axis='both', which='major', labelsize=14)
			plt.tight_layout()
			self._visualize_figs.append(fig1_1)
			"""

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

		if show: plt.show()
		if save:
			dpi = 150
			savepath = self._make_directory(savepath)
			for fig in self._visualize_figs:
				fig.savefig(os.path.join(savepath,
										 fig.texts[0].get_text() + ".png"),
							dpi=dpi)

		return None

	def diagnostic(self, show=True, save=False, savepath='results'):
		"""
		Plot or save diagnostics data

		Input params:
		--------------
		show = True : bool
			if True, show the figures 
		save = False : bool
			if True save the figures to disk
		savepath = None : str
			save destination path, if None, data is saved in current/working/directory/results/diagnostics

		returns:
		-------------
			None
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

		if show: plt.show()
		if save:
			savepath = self._make_directory(
				os.path.join(savepath, self.res_dir, 'diagnostic'))
			for fig in self._diagnostics_figs:
				fig.savefig(
					os.path.join(savepath, fig.texts[0].get_text() + ".png"))

		return None

	def noise_characterstics(self, show=True, save=False, savepath='results'):
		"""
		Plot or save noise analysis data

		Input params:
		--------------
		show = True : bool
			if True, show the figures 
		save = False : bool
			if True save the figures to disk
		savepath = None : str
			save destination path, if None, data is saved in current/working/directory/results/noise_characterstics

		returns:
		--------------
			None
		"""
		self._noise_figs = []
		#print("Noise is gaussian") if self._ddsde.gaussian_noise else print("Noise is not Gaussian")
		data = [self._ddsde._noise, self._ddsde._kl_dist, self._ddsde._X1, self._ddsde.h_lim, self._ddsde.k, self._ddsde.l_lim,self._ddsde._f, self._ddsde._noise_correlation]
		fig = self._plot_noise_characterstics(data)
		"""
		fig1 = plt.figure(dpi=150)
		plt.suptitle("Noise_Distrubution")
		sns.distplot(self._ddsde._noise)
		self._noise_figs.append(fig1)

		fig2 = plt.figure(dpi=150)
		plt.suptitle("Test_of_hypothesis")
		sns.distplot(self._ddsde._kl_dist)
		start, stop = plt.gca().get_ylim()
		plt.plot(np.ones(len(self._ddsde._X1)) * self._ddsde.l_lim,
				 np.linspace(start, stop, len(self._ddsde._X1)),
				 'r',
				 label='upper_cl')
		plt.plot(np.ones(len(self._ddsde._X1)) * self._ddsde.h_lim,
				 np.linspace(start, stop, len(self._ddsde._X1)),
				 'r',
				 label="lower_cl")
		plt.plot(np.ones(len(self._ddsde._X1)) * self._ddsde.k,
				 np.linspace(start, stop, len(self._ddsde._X1)),
				 'g',
				 label='Test Stat')
		plt.legend()
		self._noise_figs.append(fig2)

		fig3 = plt.figure(dpi=150)
		plt.suptitle("CDF")
		plt.plot(self._ddsde._X1[1:], self._ddsde._f)
		plt.plot(np.ones(len(self._ddsde._X1[1:])) * self._ddsde.l_lim,
				 self._ddsde._f,
				 'r',
				 label='lower_cl')
		plt.plot(np.ones(len(self._ddsde._X1[1:])) * self._ddsde.h_lim,
				 self._ddsde._f,
				 'r',
				 label='upper_cl')
		plt.plot(np.ones(len(self._ddsde._X1[1:])) * self._ddsde.k,
				 self._ddsde._f,
				 'g',
				 label='Test Stat')
		plt.legend()
		self._noise_figs.append(fig3)

		fig4 = plt.figure(dpi=150)
		plt.suptitle("Noise_ACF")
		plt.plot(self._ddsde._noise_correlation[0],
				 self._ddsde._noise_correlation[1])
		self._noise_figs.append(fig4)
		"""
		if show: plt.show()
		if save:
			savepath = self._make_directory(
				os.path.join(savepath, self.res_dir, 'noise_characterstics'))
			for fig in self._noise_figs:
				fig.savefig(
					os.path.join(savepath, fig.texts[0].get_text() + ".png"))

		return None


class Error(Exception):
	"""Base class for exceptions in this module."""
	pass


class InputError(Error):
	"""Exception raised for errors in the input.

	Attributes:
		expression -- input expression in which the error occurred
		message -- explanation of the error
	"""
	def __init__(self, expression, message):
		self.expression = expression
		self.message = message

	def __str__(self):
		return self.message

class PathNotFound(Error):
	def __init__(self, full_path, message):
		self.full_path = full_path
		self.message = message

	def __str__(self):
		return self.message