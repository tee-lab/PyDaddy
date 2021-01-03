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
from pyFish.preprocessing import preprocessing

class output(preprocessing):
	"""
	Class to plot and save data and parameters
	"""
	def __init__(self, out, **kwargs):
		self.vector = out.vector
		self._res_dir = str(int(time.time()))

		if not self.vector:
			self._data_X = out._X
			self._data_t = out._t
			self._data_drift = out._drift_
			self._data_diff = out._diff_
			self._data_avgdrift = out._avgdrift_
			self._data_avgdiff = out._avgdiff_
			self._data_op = out._op_
			self.drift_order = out.drift_order
			self.diff_order = out.diff_order
		else:
			self._data_vel_x = out._vel_x
			self._data_vel_y = out._vel_y
			self._data_avgdriftX = out._avgdriftX_
			self._data_avgdriftY = out._avgdriftY_
			self._data_avgdiffX = out._avgdiffX_
			self._data_avgdiffY = out._avgdiffY_
			self._data_avgdiffXY = out._avgdiffXY_
			self._data_op_x = out._op_x_
			self._data_op_y = out._op_y_

		self._out = out

		self.__dict__.update(kwargs)
		preprocessing.__init__(self)

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

	def _make_dirctory(self, p, i=1):
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
		else:
			try:
				os.mkdir(os.path.join(*p[0:i]))
			except FileExistsError:
				pass
		return self._make_dirctory(p,i=i+1)
	
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

	def save_data(self, file_name=None, savepath=None, savemat=True):
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
		if savepath is None: savepath = 'results'
		if file_name is None: file_name = self._res_dir
		savepath = self._make_dirctory(os.path.join(savepath, self._res_dir))
		if not self.vector:
			data_dict = {'drift':self._data_drift, 'diff':self._data_diff, 'avgdrift':self._data_avgdrift, 'avgdiff':self._data_avgdiff, 'op':self._data_op}
		else:
			x,y = np.meshgrid(self._data_op_x, self._data_op_y)
			data_dict = {'avgdriftX':self._data_avgdriftX, 'avgdriftY':self._data_avgdriftY, 'avgdiffX':self._data_avgdiffX, 'avgdiffY':self._data_avgdiffY, 'avgdiffXY':self._data_avgdiffXY, 'op_x':self._data_op_x, 'op_y':self._data_op_y, 'x':x, 'y':y}
		with open(os.path.join(savepath, file_name+'.pkl'), 'wb') as file:
			pickle.dump(data_dict, file)
		if savemat:
			scipy.io.savemat(os.path.join(savepath, file_name+'.mat'), mdict=data_dict)

		return None

	def save_all_data(self, savepath=None ,show=False, file_name=None):
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
		if savepath is None: savepath = "results"
		self.save_data(file_name=file_name, savepath=savepath)
		self.parameters(save=True, savepath=savepath)
		self.visualize(show=show, save=True, savepath=savepath)
		self.diagnostic(show=show, save=True, savepath=savepath)
		self.noise_characterstics(show=show, save=True, savepath=savepath)
		self.slices_2d(show=show, save=True, savepath=savepath)
		print('Results saved in: {}'.format(os.path.join(savepath, self._res_dir)))

	def parameters(self, save=False, savepath=None, file_name="parameters.txt"):
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
		for keys in self._out.__dict__.keys():
			if str(keys)[0] != '_':
				params[keys] = str(self._out.__dict__[keys])
		if save:
			savepath = self._make_dirctory(os.path.join(savepath, self._res_dir))
			with open(os.path.join(savepath, file_name), 'w') as f:
				json.dump(params, f, indent=True, separators='\n:')
		return params

	def _thrace_pane(self, data):
		op_x = self._data_op_x.copy()
		op_y = self._data_op_y.copy()
		plane1 = []
		plane2 = []
		for y in data:
			nan_idx = np.where(np.isnan(y))
			try:
				p,x = self._fit_poly(op_x, y, deg=6)
				d = p(op_x)
			except Exception as e:
				d = np.zeros(y.shape)
			d[nan_idx] = np.nan 
			plane1.append(d)
		
		for y in data.T:
			nan_idx = np.where(np.isnan(y))
			try:
				p,x = self._fit_poly(op_x, y, deg=6)
				d = p(op_x)
			except:
				d = np.zeros(y.shape)
			d[nan_idx] = np.nan 
			plane2.append(d)
		
		plane1 = np.array(plane1)
		plane2 = np.array(plane2)
		err_1 = np.nanmean(np.sqrt(np.square(plane1 - data)))
		err_2 = np.nanmean(np.sqrt(np.square(plane2 - data.T)))
		if err_1 < err_2:
			return 0, plane1
		return 1, plane2

	def _plot_heatmap(self, data, title='title', num_ticks=5):
			fig = plt.figure()
			plt.suptitle(title,verticalalignment='center', ha='right')
			ticks = self._data_op_x.copy()
			ticks_loc = np.linspace(0, len(ticks), num_ticks)
			ticks = np.linspace(min(ticks), max(ticks), num_ticks).round(2)
			ax = sns.heatmap(data,xticklabels=ticks[::-1], yticklabels=ticks,cmap=plt.cm.coolwarm, center=0)
			ax.set_xlabel('$m_x$', fontsize=16, labelpad=10)
			ax.set_ylabel('$m_y$', fontsize=16, labelpad=10)
			ax.set_xticks(ticks_loc)
			ax.set_yticks(ticks_loc)
			ax.tick_params(axis='both', which='major', labelsize=14)
			plt.tight_layout()
			return fig

	def plot_data(self, data_in, ax=None, clear=True, plot_plane=False, title='title', z_label='z', label=None,order=3, m=False, m_th=2, dpi=150, heatmap=False):
		if heatmap:
			return self._plot_heatmap(data_in, title=title)
		if ax is None:
			fig = plt.figure(dpi=dpi)
			ax = fig.add_subplot(projection="3d")        
		data = data_in.copy()
		mask = np.where(((data > m_th*np.nanstd(data))|(data < -m_th*np.nanstd(data))))
		if m: 
			#print(mask)
			data[mask] = np.nan
		if clear:
			ax.cla()
		op_x = self._data_op_x.copy()
		op_y = self._data_op_y.copy()
		plane = []
		if plot_plane:
			plane_id, plane = self._thrace_pane(data)

		x,y = np.meshgrid(op_x, op_y)
		z = data.copy()
		plt.suptitle(title, fontsize=16)

		ax.scatter3D(x, y, z.ravel(), label=label)
		if plot_plane:
			if plane_id:
				#print('Plane 2')
				ax.plot_surface(y,x,plane, rstride=1, cstride=1, alpha=0.5,)
			else:
				#print('Plane 1')
				ax.plot_surface(x,y,plane, rstride=1, cstride=1, alpha=0.5,)
		ax.set_xlabel('$m_x$', fontsize=16,labelpad=11)
		ax.set_ylabel('$m_y$', fontsize=16,labelpad=11)
		ax.set_zlabel(z_label,fontsize=16,labelpad=11)
		# make the panes transparent
		ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
		# make the grid lines transparent
		ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
		#Set ticks lable and its fontsize
		ax.tick_params(axis='both', which='major', labelsize=14)
		ax.set_xticks(np.linspace(-1,1,5))
		ax.set_yticks(np.linspace(-1,1,5))
		#plt.tight_layout()
		#plt.legend(prop={'size': 14})
		return fig

	def visualize(self, show=True, save=False, savepath=None):
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
		if savepath is None: savepath = "results"
		if not self.vector:
			savepath = os.path.join(savepath, self._res_dir, 'visualize')
			#Time series
			fig1 = fig = plt.figure(dpi=150)
			plt.suptitle("Time_Series")
			l = int(len(self._data_X)/4)
			try:
				plt.plot(self._data_t[0:l],self._data_X[0:l])
			except:
				plt.plot(self._data_X[0:l])
			self._visualize_figs.append(fig1)

			#PDF
			fig2 = fig = plt.figure(dpi=150, figsize=(5,5))
			plt.suptitle("PDF")
			sns.distplot(self._data_X)
			plt.xlim([min(self._data_X),max(self._data_X)])
			plt.ylabel('PDF')
			plt.xlabel('Order Parameter')
			self._visualize_figs.append(fig2)

			#Drift
			fig3 = plt.figure(dpi=150,figsize=(5,5))
			plt.suptitle("Average_Drift")
			p_drift, _ = self._fit_poly(self._data_op, self._data_avgdrift, self.drift_order)
			plt.scatter(self._data_op, self._data_avgdrift, marker='.')
			plt.scatter(self._data_op, p_drift(self._data_op), marker='.', alpha=0.4)
			plt.xlabel('Order Parameter')
			plt.ylabel("Deterministic")
			plt.xlim([min(self._data_X),max(self._data_X)])
			self._visualize_figs.append(fig3)

			#Diffusion
			fig4 = plt.figure(dpi=150,figsize=(5,5))
			plt.suptitle("Average_Diffusion")
			p_diff, _ = self._fit_poly(self._data_op, self._data_avgdiff, self.diff_order)
			plt.scatter(self._data_op, self._data_avgdiff, marker='.')
			plt.scatter(self._data_op, p_diff(self._data_op), marker='.', alpha=0.4)
			plt.xlim([min(self._data_X),max(self._data_X)])
			plt.xlabel("Order Parameter")
			plt.ylabel('Stochastic')
			self._visualize_figs.append(fig4)

		else:
			num_ticks=5
			savepath = os.path.join(savepath ,self._res_dir, 'visualize','plot_3d')
			fig1 = plt.figure()
			plt.suptitle("PDF")
			ax = fig1.add_subplot(projection="3d")
			H, edges, X, Y, Z, dx, dy, dz = self._histogram3d(np.array([self._data_vel_x[~np.isnan(self._data_vel_x)], self._data_vel_y[~np.isnan(self._data_vel_y)]]))
			colors = plt.cm.coolwarm(dz.flatten()/float(dz.max()))
			hist3d = ax.bar3d(X,Y,Z,dx,dy,dz, alpha=0.6, cmap=plt.cm.coolwarm, color=colors)
			ax.set_xlabel('Mx', fontsize=16, labelpad=10)
			ax.set_ylabel('My', fontsize=16, labelpad=10)
			ax.set_zlabel('Frequency',fontsize=16, labelpad=12)
			# make the panes transparent
			ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
			ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
			ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
			# make the grid lines transparent
			ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
			ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
			ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
			#Set ticks lable and its fontsize
			ax.tick_params(axis='both', which='major', labelsize=16)
			ax.set_xticks(np.linspace(-1,1,5))
			ax.set_yticks(np.linspace(-1,1,5))
			self._hist_data = (X, Y, Z, dx, dy, dz)
			self._visualize_figs.append(fig1)

			fig1_1 = plt.figure()
			plt.suptitle("PDF_heatmap",verticalalignment='center', ha='right')
			ticks = self._data_op_x.copy()
			ticks_loc = np.linspace(0, len(ticks), num_ticks)
			ticks = np.linspace(min(ticks), max(ticks), num_ticks).round(2)
			bin_count = int(np.sqrt(len(dz)))
			dz = dz.reshape((bin_count, bin_count))
			ax = sns.heatmap(dz,xticklabels=ticks, yticklabels=ticks[::-1],cmap=plt.cm.coolwarm,)
			ax.set_xlabel('$m_x$', fontsize=16, labelpad=10)
			ax.set_ylabel('$m_y$', fontsize=16, labelpad=10)
			ax.set_xticks(ticks_loc)
			ax.set_yticks(ticks_loc)
			ax.tick_params(axis='both', which='major', labelsize=14)
			plt.tight_layout()
			self._visualize_figs.append(fig1_1)

			fig2 = self.plot_data(self._data_avgdiffY, plot_plane=True, title='DiffY', z_label='$B_{22}(m)$')
			self._visualize_figs.append(fig2)
			fig2_1 = self.plot_data(self._data_avgdiffY,title='DiffY_heatmap', heatmap=True)
			self._visualize_figs.append(fig2_1)


			fig3 = self.plot_data(self._data_avgdiffX, plot_plane=True, title='DiffX', z_label='$B_{11}(m)$')
			self._visualize_figs.append(fig3)
			fig3_1 = self.plot_data(self._data_avgdiffX,title='DiffX_heatmap', heatmap=True)
			self._visualize_figs.append(fig3_1)

			fig4 = self.plot_data(self._data_avgdriftY, plot_plane=False, title='DriftY', z_label='$A_{2}(m)$')
			self._visualize_figs.append(fig4)
			fig4_1 = self.plot_data(self._data_avgdriftY,title='DriftY_heatmap', heatmap=True)
			self._visualize_figs.append(fig4_1)

			fig5 = self.plot_data(self._data_avgdriftX, plot_plane=False, title='DriftX', z_label='$A_{1}(m)$')
			self._visualize_figs.append(fig5)
			fig5_1 = self.plot_data(self._data_avgdriftX,title='DriftX_heatmap', heatmap=True)
			self._visualize_figs.append(fig5_1)
		
		if show: plt.show()
		if save:
			dpi = 150
			savepath = self._make_dirctory(savepath)
			for fig in self._visualize_figs : fig.savefig(os.path.join(savepath, fig.texts[0].get_text()+".png"), dpi=dpi)

		return None


	def diagnostic(self, show=True, save=False, savepath=None):
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
		if savepath is None: savepath="results"
		t1 = "R2" if self._out.order_metric=="R2" else "R2_adj"
		#ACF
		fig1 = plt.figure(dpi=150)
		plt.suptitle("ACF")
		exp_fn = lambda t,a,b,c: a*np.exp((-1/b)*t)+c
		plt.plot(self._out._autocorr_x, self._out._autocorr_y)
		y = exp_fn(self._out._autocorr_x, self._out._a, self._out.autocorrelation_time, self._out._c)
		plt.plot(self._out._autocorr_x, y)
		plt.legend(('ACF', 'exp_fit'))
		plt.xlabel('Time Lag')
		plt.ylabel('ACF')
		self._diagnostics_figs.append(fig1)

		#R2 vs order for drift
		fig2 = plt.figure(dpi=150)
		plt.suptitle("{}_vs_drift_order".format(t1))
		plt.plot(range(self._out.max_order), self._out._r2_drift)
		plt.xlabel('order')
		plt.ylabel(t1)
		self._diagnostics_figs.append(fig2)

		#R2 vs order for diff
		fig3 = plt.figure(dpi=150)
		plt.suptitle("{}_vs_Diff_order".format(t1))
		plt.plot(range(self._out.max_order), self._out._r2_diff)
		plt.xlabel('order')
		plt.ylabel(t1)
		#plt.title('{} Diff vs order'.format(t1))
		self._diagnostics_figs.append(fig3)

		#R2 vs order for drift, multiple dt
		label = ["dt={}".format(i) for i in self._out._r2_drift_m_dt[-1]]
		fig4 = plt.figure(dpi=150)
		plt.suptitle("{}_Drift_different_dt".format(t1))
		for i in range(len(self._out._r2_drift_m_dt) -1): plt.plot(range(self._out.max_order), self._out._r2_drift_m_dt[i], label=self._out._r2_drift_m_dt[-1][i])
		plt.xlabel('order')
		plt.ylabel(t1)
		plt.legend()
		self._diagnostics_figs.append(fig4)

		#R2 vs order for diff, multiple dt
		fig5 = plt.figure(dpi=150)
		plt.suptitle("{}_Diff_different_dt".format(t1))
		for i in range(len(self._out._r2_drift_m_dt) -1): plt.plot(range(self._out.max_order), self._out._r2_diff_m_dt[i], label=self._out._r2_drift_m_dt[-1][i])
		plt.xlabel('order')
		plt.ylabel(t1)
		plt.legend()
		self._diagnostics_figs.append(fig5)


		
		if show: plt.show()
		if save:
			savepath = self._make_dirctory(os.path.join(savepath, self._res_dir, 'diagnostic'))
			for fig in self._diagnostics_figs: fig.savefig(os.path.join(savepath, fig.texts[0].get_text()+".png"))

		return None

	def noise_characterstics(self, show=True, save=False, savepath=None):
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
		if savepath is None: savepath = "results"
		#print("Noise is gaussian") if self._out.gaussian_noise else print("Noise is not Gaussian")

		fig1 = plt.figure(dpi=150)
		plt.suptitle("Noise_Distrubution")
		sns.distplot(self._out._noise)
		self._noise_figs.append(fig1)

		fig2 = plt.figure(dpi=150)
		plt.suptitle("Test_of_hypothesis")
		sns.distplot(self._out._kl_dist)
		start, stop = plt.gca().get_ylim()
		plt.plot(np.ones(len(self._out._X1))*self._out.l_lim, np.linspace(start,stop,len(self._out._X1)),'r', label='upper_cl')
		plt.plot(np.ones(len(self._out._X1))*self._out.h_lim, np.linspace(start,stop,len(self._out._X1)),'r', label="lower_cl")
		plt.plot(np.ones(len(self._out._X1))*self._out.k, np.linspace(start,stop,len(self._out._X1)),'g', label='Test Stat')
		plt.legend()
		self._noise_figs.append(fig2)

		fig3 = plt.figure(dpi=150)
		plt.suptitle("CDF")
		plt.plot(self._out._X1[1:], self._out._f)
		plt.plot(np.ones(len(self._out._X1[1:]))*self._out.l_lim, self._out._f, 'r', label='lower_cl')
		plt.plot(np.ones(len(self._out._X1[1:]))*self._out.h_lim, self._out._f, 'r', label='upper_cl')
		plt.plot(np.ones(len(self._out._X1[1:]))*self._out.k, self._out._f, 'g', label='Test Stat')
		plt.legend()
		self._noise_figs.append(fig3)

		fig4 = plt.figure(dpi=150)
		plt.suptitle("Noise_ACF")
		plt.plot(self._out._noise_correlation[0], self._out._noise_correlation[1])
		self._noise_figs.append(fig4)
		
		if show: plt.show()
		if save:
			savepath = self._make_dirctory(os.path.join(savepath, self._res_dir, 'noise_characterstics'))
			for fig in self._noise_figs: fig.savefig(os.path.join(savepath, fig.texts[0].get_text()+".png"))

		return None


	def slices_2d(self, show=True, save=False, savepath=None):
		"""
		Plot or save 2d slice of the vector 3d plots

		Input params:
		--------------
		show = True : bool
			if True, show the figures 
		save = False : bool
			if True save the figures to disk
		savepath = None : str
			save destination path, if None, data is saved in current/working/directory/results/visulize/slice_2d

		returns:
		--------------
			None
		"""
		self._slice_figs = []
		if savepath is None: savepath="results"

		if not self.vector: return None
		x,y = np.meshgrid(self._data_op_x, self._data_op_y)

		fig1 = plt.figure()
		plt.suptitle("PDF(2d_slice)")
		sns.distplot(self._data_vel_x[np.where((self._data_vel_y>=-1*self._out.inc_y) & (self._data_vel_y<=self._out.inc_y))])
		plt.xlabel('Mx', fontsize=16)
		plt.tight_layout()
		self._slice_figs.append(fig1)

		fig2 = plt.figure()
		ax = plt.gca()
		plt.suptitle("Average_Drift_X(2d_slice)")
		p, _ = self._fit_poly(x[10],self._data_avgdriftX[10], deg=self._out.drift_order)
		ax.scatter(x[10], self._data_avgdriftX[10], label='avgdriftX')
		ax.scatter(x[10], p(x[10]), label='poly')
		ax.set_xlabel('Mx', fontsize=16, labelpad=10)
		ax.set_ylabel('Deterministic Mx', fontsize=16, labelpad=10)
		ax.tick_params(axis='both', which='major', labelsize=16)
		ax.set_xticks(np.linspace(-1,1,5))
		ax.legend()
		plt.tight_layout()
		self._slice_figs.append(fig2)

		fig3 = plt.figure()
		plt.suptitle("Average_Drift_Y(2d_slice)")
		ax = plt.gca()
		p, _ = self._fit_poly(x[10],self._data_avgdriftY[10], deg=self._out.drift_order)
		ax.scatter(x[10], self._data_avgdriftY[10], label='avgdriftY')
		ax.scatter(x[10], p(x[10]), label='poly')
		ax.set_xlabel('Mx', fontsize=16, labelpad=10)
		ax.set_ylabel('Deterministic My,', fontsize=16, labelpad=10)
		ax.tick_params(axis='both', which='major', labelsize=16)
		ax.set_xticks(np.linspace(-1,1,5))
		ax.legend()
		plt.tight_layout()
		self._slice_figs.append(fig3)

		fig4 = plt.figure()
		plt.suptitle("Average_Diffusion_X(2d_slice)")
		p, _ = self._fit_poly(x[10],self._data_avgdiffX[10], deg=self._out.diff_order)
		ax = plt.gca()
		ax.scatter(x[10], self._data_avgdiffX[10], label='avgdriftY')
		ax.scatter(x[10], p(x[10]), label='poly')
		ax.set_xlabel('Mx', fontsize=16, labelpad=10)
		ax.set_ylabel('Stochastic Mx', fontsize=16, labelpad=10)
		ax.tick_params(axis='both', which='major', labelsize=16)
		ax.set_xticks(np.linspace(-1,1,5))
		ax.legend()
		plt.tight_layout()
		self._slice_figs.append(fig4)

		fig5 = plt.figure()
		plt.suptitle("Average_Diffusion_Y(2d_slice)")
		p, _ = self._fit_poly(x[10],self._data_avgdiffY[10], deg=self._out.diff_order)
		ax = plt.gca()
		ax.scatter(x[10], self._data_avgdiffY[10], label='avgdriftY')
		ax.scatter(x[10], p(x[10]), label='poly')
		ax.set_xlabel('Mx', fontsize=16, labelpad=10)
		ax.set_ylabel('Stochanstic My', fontsize=16, labelpad=10)
		ax.tick_params(axis='both', which='major', labelsize=16)
		ax.set_xticks(np.linspace(-1,1,5))
		ax.legend()
		plt.tight_layout()
		self._slice_figs.append(fig5)

		fig5_1 = plt.figure()
		plt.suptitle("Average_Diffusion_Y(2d_slice)_wrt_My")
		p, _ = self._fit_poly(x[10],self._data_avgdiffY.T[10], deg=self._out.diff_order)
		ax = plt.gca()
		ax.scatter(x[10], self._data_avgdiffY.T[10], label='avgdriftY')
		ax.scatter(x[10], p(x[10]), label='poly')
		ax.set_xlabel('My', fontsize=16, labelpad=10)
		ax.set_ylabel('Stochanstic My', fontsize=16, labelpad=10)
		ax.tick_params(axis='both', which='major', labelsize=16)
		ax.set_xticks(np.linspace(-1,1,5))
		ax.legend()
		plt.tight_layout()
		self._slice_figs.append(fig5_1)

		if show: plt.show()
		if save:
			savepath=self._make_dirctory(os.path.join(savepath, self._res_dir, 'visualize', 'slices_2d'))
			dpi=150
			for fig in self._slice_figs: fig.savefig(os.path.join(savepath, fig.texts[0].get_text()+".png"),dpi=150, transparent=True)

		return None

	def _histogram3d(self,x,bins = 20, normed = False, color = 'blue', alpha = 1, hold = False, plot_hist=False):
		"""
		Plotting a 3D histogram

		Parameters
		----------

		sample : array_like.		
			The data to be histogrammed. It must be an (N,2) array or data 
			that can be converted to such. The rows of the resulting array 
			are the coordinates of points in a 2 dimensional polytope.

		bins : sequence or int, optional, default: 10.
			The bin specification:
			
			* A sequence of arrays describing the bin edges along each dimension.
			* The number of bins for each dimension (bins =[binx,biny])
			* The number of bins for all dimensions (bins = bins).

		normed : bool, optional, default: False.
			If False, returns the number of samples in each bin. 
			If True, returns the bin density bin_count / sample_count / bin_volume.

		color: string, matplotlib color arg, default = 'blue'

		alpha: float, optional, default: 1.
			0.0 transparent through 1.0 opaque

		hold: boolean, optional, default: False

		Returns   
		--------
		H : ndarray.
			The bidimensional histogram of sample x.

		edges : list.
			A list of 2 arrays describing the bin edges for each dimension.
			
		Examples
		--------
		>>> r = np.random.randn(1000,2)
		>>> H, edges = np._histogram3d(r,bins=[10,15])
		"""

		if np.size(bins) == 1:
			bins = [bins,bins]

		if(len(x) == 2):
			x = x.T;
			

		H, edges = np.histogramdd(x, bins, normed = normed)

		H = H.T
		X = np.array(list(np.linspace(min(edges[0]),max(edges[0]),bins[0]))*bins[1])   
		Y = np.sort(list(np.linspace(min(edges[1]),max(edges[1]),bins[1]))*bins[0])    

		dz = np.array([]);

		for i in range(bins[1]):
			for j in range(bins[0]):
				dz = np.append(dz, H[i][j])

		Z = np.zeros(bins[0]*bins[1])

		dx = X[1] - X[0]   
		dy = Y[bins[0]] - Y[0]

		if plot_hist:
			if (not hold):
				fig = plt.figure(dpi = 300)
				ax = fig.add_subplot(111, projection='3d')
				colors = plt.cm.jet(dz.flatten()/float(dz.max()))
				ax.bar3d(X,Y,Z,dx,dy,dz, alpha = alpha, color = colors);
			else:
				try:
					ax = plt.gca();
					colors = plt.cm.jet(dz.flatten()/float(dz.max()))
					ax.bar3d(X,Y,Z,dx,dy,dz, alpha = alpha, color = colors);
				except:
					plt.close(plt.get_fignums()[-1])
					fig = plt.figure()
					ax = fig.add_subplot(111, projection='3d')
					colors = plt.cm.jet(dz.flatten()/float(dz.max()))
					ax.bar3d(X,Y,Z,dx,dy,dz, alpha = alpha, color = colors);
				
				
		plt.xlabel('X');
		plt.ylabel('Y');
		edges = [X,Y];
		H = dz.reshape(bins[0],bins[1]);

		#return H, edges;
		return H, edges, X, Y, Z, dx, dy, dz



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
