import gc
import os
import sys
import time
import warnings
from collections import OrderedDict, namedtuple

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import scipy.io
import scipy.optimize
import scipy.stats
from scipy.signal import correlate
from scipy.stats import skew, kurtosis
import seaborn as sns
import tqdm

from pydaddy.preprocessing import Preprocessing
from pydaddy.visualize import Visualize

__all__ = ['Daddy']


class Daddy(Preprocessing, Visualize):
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

        self.fit = ddsde.fit
        self.fast_mode = ddsde.fast_mode

        if not self.vector:
            self._data_X = ddsde._X
            self._data_t = ddsde._t
            self._data_op = ddsde._op_

            # self.drift_order = ddsde.drift_order
            # self.diff_order = ddsde.diff_order

            Visualize.__init__(self, None, None, self._data_op,
                               self._ddsde.autocorrelation_time)
        else:
            self._data_Mx = ddsde._Mx
            self._data_My = ddsde._My
            self._data_M = np.sqrt(self._data_Mx ** 2 + self._data_My ** 2)
            self._data_op_x = ddsde._op_x_
            self._data_op_y = ddsde._op_y_

            # FIXME: B12 = B21, no need to keep both.

            # self._drift_slider = ddsde._drift_slider
            # self._diff_slider = ddsde._diff_slider

            Visualize.__init__(self, self._data_op_x, self._data_op_y, None,
                               self._ddsde.autocorrelation_time, _act_mx=self._ddsde._act_mx,
                               _act_my=self._ddsde._act_my)

        self._drift_slider = ddsde._drift_slider
        self._diff_slider = ddsde._diff_slider
        self._cross_diff_slider = ddsde._cross_diff_slider

        self.res_dir = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

        self.__dict__.update(kwargs)
        Preprocessing.__init__(self)

        if ddsde._show_summary:
            self.summary(ret_fig=False)

    # The following attributes of ddsde may change during runtime (e.g. when fit() is called).
    # Defining them as property getters ensures that we always have the up-to-date value.

    @property
    def _data_avgdrift(self):
        return self._ddsde._avgdrift_

    @property
    def _data_avgdiff(self):
        return self._ddsde._avgdiff_

    @property
    def _data_drift_ebar(self):
        return self._ddsde._drift_ebar

    @property
    def _data_diff_ebar(self):
        return self._ddsde._diff_ebar

    @property
    def _data_avgdriftX(self):
        return self._ddsde._avgdriftX_

    @property
    def _data_avgdriftY(self):
        return self._ddsde._avgdriftY_

    @property
    def _data_avgdiffX(self):
        return self._ddsde._avgdiffX_

    @property
    def _data_avgdiffY(self):
        return self._ddsde._avgdiffY_

    @property
    def _data_avgdiffXY(self):
        return self._ddsde._avgdiffXY_

    @property
    def _data_avgdiffYX(self):
        return self._ddsde._avgdiffYX_

    @property
    def F(self):
        return self._ddsde.F

    @property
    def G(self):
        return self._ddsde.G

    @property
    def A1(self):
        return self._ddsde.A1

    @property
    def A2(self):
        return self._ddsde.A2

    @property
    def B11(self):
        return self._ddsde.B11

    @property
    def B22(self):
        return self._ddsde.B22

    @property
    def B12(self):
        return self._ddsde.B12

    @property
    def B21(self):
        return self._ddsde.B21

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

    def export_data(self, filename=None, raw=False):
        """
        Returns a pandas dataframe containing the drift and diffusion values.
        Args
        ----
        filename : str, optional (default=None).
            If provided, the data will be saved as a CSV at the given path. Else, a dataframe will be returned.
        raw : bool, optional (default=False)
            If True, the raw, the drift and diffusion will be returned as raw unbinned data. Otherwise (default),
            drift and diffusion as binwise-average Kramers-Moyal coefficients are returned.

        Returns
        -------
        df : Pandas dataframe containing the estimated drift and diffusion coefficients.

        """

        if not self.vector:
            if raw:
                data_dict = dict(
                    x=self._data_X[:-1],
                    drift=self._ddsde._drift_,
                    diffusion=self._ddsde._diffusion_,
                )
            else:
                data_dict = dict(
                    x=self._data_op,
                    drift=self._data_avgdrift,
                    diffusion=self._data_avgdiff,
                )
        else:
            if raw:
                data_dict = dict(
                    x=self._data_Mx[:-1],
                    y=self._data_My[:-1],
                    drift_x=self._ddsde._driftX_,
                    drift_y=self._ddsde._driftY_,
                    diffusion_x=self._ddsde._diffusionX_,
                    diffusion_y=self._ddsde._diffusionY_,
                    diffusion_xy=self._ddsde._diffusionXY_,
                )
            else:
                x, y = np.meshgrid(self._data_op_x, self._data_op_y)
                data_dict = dict(
                    x=x.flatten(),
                    y=y.flatten(),
                    drift_x=self._data_avgdriftX.flatten(),
                    drift_y=self._data_avgdriftY.flatten(),
                    diffusion_x=self._data_avgdiffX.flatten(),
                    diffusion_y=self._data_avgdiffX.flatten(),
                    diffusion_xy=self._data_avgdriftX.flatten(),
                )

        df = pd.DataFrame(data=data_dict)

        if self.vector:
            na_rows = df[['drift_x', 'drift_y', 'diffusion_x', 'diffusion_y', 'diffusion_xy']].isna().all(axis=1)
            df = df[~na_rows]

        if filename:
            df.to_csv(filename)
        else:
            return df

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
            Data = namedtuple('Data', ('drift', 'diff', 'drift_num', 'diff_num', 'op'))
            drift, diff = self._get_data_from_slider(drift_time_scale, diff_time_scale)
            # _ , diff = self._get_data_from_slider(diff_time_scale)
            drift_num, diff_num = self._get_num_points(drift_time_scale, diff_time_scale)
            return Data(drift, diff, drift_num, diff_num, self._data_op)

        Data = namedtuple('Data', ('driftX', 'driftY', 'diffX', 'diffY', 'diffXY', 'diffYX', 'op_x', 'op_y'))
        driftX, driftY, diffX, diffY, diffXY, diffYX = self._get_data_from_slider(drift_time_scale, diff_time_scale)
        # _, _, diffX, diffY, diffXY, diffYX = self._get_data_from_slider(diff_time_scale)
        return Data(driftX, driftY, diffX, diffY, diffXY, diffYX, self._data_op_x, self._data_op_y)

    # def kl_div(self, a,b):
    #	a, bins_a = np.histogram(a, bins=100, density=True)
    #	b, bins_b = np.histogram(b, bins=bins_a, density=True)
    #	a_b = np.sum(np.where((a != 0)&(b != 0), a * np.log(a / b), 0))
    #	b_a = np.sum(np.where((a != 0)&(b != 0), b * np.log(b / a), 0))
    #	return (a_b + b_a)/2

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
            'F': None,
            'G': None,
            'A1': None,
            'A2': None,
            'B11': None,
            'B12': None,
            'B21': None,
            'B22': None
        }

        func.update(functions)

        if dt is None: dt = self._ddsde.t_int

        if self.vector:
            for k in ['A1', 'A2', 'B11', 'B12', 'B21', 'B22']:
                if func[k] == None:
                    print('Insufficient data, provide {}'.format(k))
                    return None
            if T is None: T = len(self._data_Mx) * dt
            n_iter = int(T / dt)
            mx = [self._data_Mx[0]]
            my = [self._data_My[0]]
            for i in tqdm.tqdm(range(n_iter)):
                mx.append(mx[i] + func['A1'](mx[i], my[i]) * dt + sigma * np.random.normal() * (
                        func['B11'](mx[i], my[i]) + func['B12'](mx[i], my[i])) * np.sqrt(dt))
                my.append(my[i] + func['A2'](mx[i], my[i]) * dt + sigma * np.random.normal() * (
                        func['B22'](mx[i], my[i]) + func['B21'](mx[i], my[i])) * np.sqrt(dt))
            return np.array(mx), np.array(my)

        else:
            for k in ['F', 'G']:
                if func[k] == None:
                    print('Insufficient data, provide {}'.format(k))
                    return None

            if T is None: T = len(self._data_X) * dt

            n_iter = int(T / dt)

            m = [self._data_X[0]]

            for i in tqdm.tqdm(range(n_iter)):
                m.append(m[i] + func['F'](m[i]) * dt + sigma * np.random.normal() * func['G'](m[i]) * np.sqrt(dt))

            return np.array(m)

    def summary(self, start=0, end=1000, kde=True, tick_size=12, title_size=15, label_size=15, label_pad=8, n_ticks=3,
                ret_fig=False, **plot_text):

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
            fields = ['M range', 'M mean',
                      '|M| range', '|M| mean',
                      'Autocorr time (M)', '(Dt, dt)',
                      ]

            values = [self._get_data_range(self._data_X), round(np.nanmean(self._data_X), 3),
                      self._get_data_range(np.sqrt(self._data_X ** 2)),
                      round(np.nanmean(np.sqrt(self._data_X ** 2)), 3),
                      self.autocorrelation_time, (self._ddsde.Dt, self._ddsde.dt),
                      ]
            values = list(map(str, values))
            summary = []
            for i in range(len(fields)):
                summary.append(fields[i])
                summary.append(values[i])
            summary_format = ("| {:<20} : {:^15}" * 1 + "|\n") * int(len(fields) / 1)
            print(summary_format.format(*summary))
            if self._ddsde.F:
                print(f'Drift:\n{self._ddsde.F}\n')
            if self._ddsde.G:
                print(f'Diffusion:\n{self._ddsde.G}\n')
            data = [self._data_X, self._data_avgdrift, self._data_avgdiff, self._data_drift_ebar, self._data_diff_ebar]

        else:
            fields = ['Mx range', 'Mx mean',
                      'My range', 'My mean',
                      '|M| range', '|M| mean',
                      'Autocorr time (Mx, My, |M^2|)', '(Dt, dt)',
                      ]

            values = [self._get_data_range(self._data_Mx), round(np.nanmean(self._data_Mx), 3),
                      self._get_data_range(self._data_My), round(np.nanmean(self._data_My), 3),
                      self._get_data_range(self._data_M),
                      round(np.nanmean(np.sqrt(self._data_Mx ** 2 + self._data_My ** 2)), 3),
                      (self._ddsde._act_mx, self._ddsde._act_my, self.autocorrelation_time),
                      (self._ddsde.Dt, self._ddsde.dt)
                      ]
            values = list(map(str, values))
            summary = []
            for i in range(len(fields)):
                summary.append(fields[i])
                summary.append(values[i])
            summary_format = ("| {:<30} : {:^15}" * 1 + "|\n") * int(len(fields) / 1)
            # print(
            #     "Note: All summary and plots are rounded to third decimal place.\nCalculations, however, are accurate and account for missing values too.\n\n")
            print(summary_format.format(*summary))
            if self._ddsde.A1:
                print(f'Drift (A1): {self._ddsde.A1}')
            if self._ddsde.A2:
                print(f'Drift (A2): {self._ddsde.A2}')
            if self._ddsde.B11:
                print(f'Diffusion (B11): {self._ddsde.B11}')
            if self._ddsde.B22:
                print(f'Diffusion (B22): {self._ddsde.B22}')
            if self._ddsde.B21:
                print(f'Cross diffusion (B12, B21): {self._ddsde.B21}')
            data = [self._data_Mx, self._data_My, self._data_avgdriftX, self._data_avgdriftY, self._data_avgdiffX,
                    self._data_avgdiffY, self._data_avgdiffXY]

        sys.stdout.flush()
        fig = self._plot_summary(data, self.vector, kde=kde, tick_size=tick_size, title_size=title_size,
                                 label_size=label_size, label_pad=label_pad, n_ticks=n_ticks, timeseries_start=start,
                                 timeseries_end=end, **plot_text)
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
        fig = self._plot_timeseries(data, self.vector, start=start, stop=end, n_ticks=n_ticks, dpi=dpi,
                                    tick_size=tick_size, title_size=title_size, label_size=label_size,
                                    label_pad=label_pad, **plot_text)
        plt.show()
        return fig

    def histogram(self,
                  kde=False,
                  heatmap=False,
                  dpi=150,
                  title_size=14,
                  label_size=15,
                  tick_size=12,
                  label_pad=8,
                  **plot_text):
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
                                    heatmap=heatmap,
                                    dpi=dpi,
                                    kde=kde,
                                    title_size=title_size,
                                    label_size=label_size,
                                    tick_size=tick_size,
                                    label_pad=label_pad,
                                    **plot_text)
        plt.show()
        # return fig

    def autocorrelation(self):
        if not self.vector:
            lags, acf = self._ddsde._acf(self._data_X, min(1000, len(self._data_X)))
            self._plot_autocorrelation_1d(lags, acf)
        else:
            lags, acfm = self._ddsde._acf(self._data_M ** 2, min(1000, len(self._data_M)))
            _, acfx = self._ddsde._acf(self._data_Mx, min(1000, len(self._data_Mx)))
            _, acfy = self._ddsde._acf(self._data_My, min(1000, len(self._data_My)))
            _, ccf = self._ddsde._ccf(self._data_Mx, self._data_My, min(1000, len(self._data_Mx)))
            self._plot_autocorrelation_2d(lags, acfx, acfy, acfm, ccf)


    def _update_slider_data(self, slider_timescales):
        if self._is_valid_slider_timescale_list(slider_timescales):
            if self._ddsde.slider_timescales is None:
                self._ddsde.slider_timescales = sorted(map(int, set(slider_timescales)))
            else:
                self._ddsde.slider_timescales = sorted(
                    map(int, set(self._ddsde.slider_timescales).union(list(slider_timescales))))
        else:
            return None

        if self.vector:
            self._drift_slider, self._diff_slider, self._cross_diff_slider = self._ddsde._slider_data(self._ddsde._Mx,
                                                                                                      self._ddsde._My,
                                                                                                      update=True)
            self._drift_slider = OrderedDict(sorted(self._drift_slider.items()))
            self._diff_slider = OrderedDict(sorted(self._diff_slider.items()))
            self._cross_diff_slider = OrderedDict(self._cross_diff_slider.items())
        else:
            self._drift_slider, self._diff_slider, self._scalar_drift_ebars, self._scalar_diff_ebars, \
            self._scalar_drift_var, self.scalar_diff_var = self._ddsde._slider_data(self._ddsde._X, None, update=True)
            self._drift_slider = OrderedDict(sorted(self._drift_slider.items()))
            self._diff_slider = OrderedDict(sorted(self._diff_slider.items()))
        return None

    def drift(self, limits=None, polar=False, slider_timescales=None, **plot_text):
        """
		Display drift slider figure

		Args
		----
		polynomial_order : None or int, default=None
			order of polynomial to fit, if None, no fitting is done.
		**plot_text:
			plots' axis and text label

			For scalar analysis
				x_lable : x axis label

				y_label : y axis label

			For vector analysis
				title1 : first plot title

				x_label1 : first plot x label

				y_label1 : first plot y label

				z_label1 : first plot z label

				title2 : second plot title

				x_label2 : second plot x label

				y_label2 : seocnd plot y label

				z_label2 : second plot z label

		Returns
		-------
		opens drift slider : None
		"""
        self._update_slider_data(slider_timescales)
        dt_s = list(self._drift_slider.keys())
        if not len(dt_s):  # empty slider
            return None
        init_pos = np.abs(np.array(dt_s) - self._ddsde.Dt).argmin()
        if self.vector:
            fig = self._slider_3d(self._drift_slider, prefix='Dt', init_pos=init_pos, zlim=limits, polar=polar,
                                  **plot_text)
        else:
            fig = self._slider_2d(self._drift_slider, prefix='Dt', init_pos=init_pos, limits=limits,
                                  **plot_text)
        fig.show()
        return None

    def diffusion(self, slider_timescales=None, limits=None, polar=False, **plot_text):
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
        self._update_slider_data(slider_timescales)
        dt_s = list(self._diff_slider.keys())
        if not len(dt_s):  # empty slider
            return None
        if self.vector:
            fig = self._slider_3d(self._diff_slider, prefix='dt', init_pos=0, zlim=limits, polar=polar,
                                  **plot_text)
        else:
            fig = self._slider_2d(self._diff_slider, prefix='dt', init_pos=0, limits=limits,
                                  **plot_text)
        fig.show()
        return None

    def cross_diffusion(self, slider_timescales=None, limits=None, polar=False, **plot_text):
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
        if not self.vector:
            print("N/A")
            return None
        self._update_slider_data(slider_timescales)
        dt_s = list(self._cross_diff_slider.keys())
        if not len(dt_s):  # empty slider
            return None

        if limits:
            zlim = limits
        else:
            zlim = (-max(np.nanmax(self._data_avgdiffX), np.nanmax(self._data_avgdiffY)),
                    max(np.nanmax(self._data_avgdiffX), np.nanmax(self._data_avgdiffY)))

        fig = self._slider_3d(self._cross_diff_slider, prefix='c_dt', init_pos=0, zlim=zlim, polar=polar,
                              **plot_text)
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
        if not self.vector:
            drift, diff = self._get_data_from_slider(drift_time_scale, diff_time_scale)
            # _, diff = self._get_data_from_slider(diff_time_scale)
            if drift_time_scale is None:
                drift_ebar = self._data_drift_ebar
            else:
                drift_ebar = self._ddsde._scalar_drift_ebars[
                    self._closest_time_scale(drift_time_scale, self._ddsde._scalar_drift_ebars)]
            if diff_time_scale is None:
                diff_ebar = self._data_diff_ebar
            else:
                diff_ebar = self._ddsde._scalar_diff_ebars[
                    self._closest_time_scale(diff_time_scale, self._ddsde._scalar_diff_ebars)]

            # Time series
            fig1 = plt.figure(dpi=150)
            plt.suptitle("Time Series")
            # l = int(len(self._data_X) / 4)
            l = 1000
            try:
                plt.plot(self._data_t[0:l], self._data_X[0:l])
            except:
                plt.plot(self._data_X[0:l])

            # PDF
            fig2 = plt.figure(dpi=150, figsize=(5, 5))
            plt.suptitle("PDF")
            sns.distplot(self._data_X)
            plt.xlim([min(self._data_X), max(self._data_X)])
            plt.xticks(np.linspace(min(self._data_X), max(self._data_X), 5))
            plt.ylabel('PDF')
            plt.xlabel('M')

            # Drift
            fig3 = plt.figure(dpi=150, figsize=(5, 5))
            plt.suptitle("Drift")
            # p_drift, _ = self._fit_poly(self._data_op, drift,
            #							self.drift_order)
            # plt.scatter(self._data_op, drift, marker='.')
            plt.errorbar(self._data_op, drift, yerr=drift_ebar, fmt='o')
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

            # Diffusion
            fig4 = plt.figure(dpi=150, figsize=(5, 5))
            plt.suptitle("Diffusion")
            # p_diff, _ = self._fit_poly(self._data_op, diff,
            #						   self.diff_order)
            # plt.scatter(self._data_op, diff, marker='.')
            plt.errorbar(self._data_op, diff, yerr=diff_ebar, fmt='o')
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
            driftX, driftY, diffX, diffY, diffXY, diffYX = self._get_data_from_slider(drift_time_scale, diff_time_scale)
            # _, _, diffX, diffY, diffXY, diffYX = self._get_data_from_slider(diff_time_scale)
            fig1, _ = self._plot_3d_hisogram(self._data_Mx, self._data_My, title='PDF', xlabel="$M_{x}$", tick_size=12,
                                             label_size=12, title_size=12, r_fig=True)

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

    def noise_diagnostics(self, loc=None):
        if self.vector:
            if loc is None:
                loc = (0, 0)

            X, Y = self._ddsde._Mx, self._ddsde._My
            Dt = self._ddsde.Dt
            inc_x, inc_y = self._ddsde.inc_x, self._ddsde.inc_y
            t_int = self._ddsde.t_int
            op_x, op_y = self._ddsde._op_x_, self._ddsde._op_y_
            avg_drift_x, avg_drift_y = self._ddsde._avgdriftX_, self._ddsde._avgdriftY_
            res_x, res_y = self._ddsde._residual_timeseries_vector(
                X=X, Y=Y, Dt=Dt,
                bins_x=op_x, bins_y=op_y,
                avg_drift_x=avg_drift_x, avg_drift_y=avg_drift_y,
                t_int=t_int
            )

            noise_dist_x = res_x[(loc[0] <= X[:-Dt]) & (X[:-Dt] < loc[0] + inc_x) & (loc[1] <= Y[:-Dt]) & (Y[:-Dt] < loc[1] + inc_y)]
            noise_dist_y = res_y[(loc[0] <= X[:-Dt]) & (X[:-Dt] < loc[0] + inc_x) & (loc[1] <= Y[:-Dt]) & (Y[:-Dt] < loc[1] + inc_y)]
            noise_corr = np.ma.corrcoef([np.ma.masked_invalid(noise_dist_x),
                                         np.ma.masked_invalid(noise_dist_y)])

            lags, acf_x = self._ddsde._acf(res_x, t_lag=min(100, len(res_x)))
            _, acf_y = self._ddsde._acf(res_y, t_lag=min(100, len(res_y)))

            (_, bx, _), _ = self._ddsde._fit_exp(lags, acf_x)  # Fit a * exp(-t / b) + c
            act_x = bx

            (_, by, _), _ = self._ddsde._fit_exp(lags, acf_y)  # Fit a * exp(-t / b) + c
            act_y = by

            # Summary information
            print('Noise statistics:')
            print(f'Mean: ({np.nanmean(noise_dist_x):.4f}, {np.nanmean(noise_dist_y):.4f})')
            print(f'Correlation matrix:\n'
                  f'    {noise_corr[0, 0]:+.4f}    {noise_corr[0, 1]:+.4f}\n'
                  f'    {noise_corr[1, 0]:+.4f}    {noise_corr[1, 1]:+.4f}')

            print('\nNoise autocorrelation time:')
            print(f'    eta_x: {act_x:.3f} timesteps ({act_x * Dt * t_int:.3f}s)'
                  f'    eta_y: {act_y:.3f} timesteps ({act_y * Dt * t_int:.3f}s)')

            # Summary figures
            fig = plt.figure(figsize=(7, 7))
            gs = fig.add_gridspec(4, 2)
            ax_2d = fig.add_subplot(gs[:2, 0], projection='3d')
            ax_acf = fig.add_subplot(gs[:2, 1])
            # ax_acf2 = fig.add_subplot(gs[1, 1])
            ax_qqx = fig.add_subplot(gs[2:, 0])
            ax_qqy = fig.add_subplot(gs[2:, 1])
            ax_corr = inset_axes(ax_2d, width='30%', height='39%', loc='upper left')

            self._noise_plot_2d(ax_2d, noise_dist_x, noise_dist_y, title='Residual Distribution')
            self._matrix_plot(ax_corr, noise_corr)
            # self._acf_plot(ax_acf1, acf, lags, a, b, c, act, title='Autocorrelation: $|\\eta|$')
            self._acf_plot_multi(ax_acf, acf_x, acf_y, lags, act_x, act_y, title='Autocorrelation: $\\eta_x, \\eta_y$')
            self._qq_plot(ax_qqx, noise_dist_x, title='QQ Plot: $\\eta_x$')
            self._qq_plot(ax_qqy, noise_dist_y, title='QQ Plot: $\\eta_y$')

            plt.tight_layout()
            plt.show()
        else:
            if loc is None:
                loc = 0
            X = self._ddsde._X
            Dt = self._ddsde.Dt
            inc = self._ddsde.inc
            t_int = self._ddsde.t_int
            op = self._ddsde._op_
            avg_drift = self._ddsde._avgdrift_
            residual = self._ddsde._residual_timeseries(X=X,
                                                        Dt=Dt,
                                                        bins=op,
                                                        avg_drift=avg_drift,
                                                        t_int=t_int,
                                                        )

            noise_distribution = residual[(loc <= X[:-Dt]) & (X[:-Dt] < loc + inc)]

            # Compute residual autocorrelation
            lags, acf = self._ddsde._acf(residual, t_lag=min(100, len(residual)))
            (a, b, c), _ = self._ddsde._fit_exp(lags, acf)  # Fit a * exp(-t / b) + c
            act = b

            # Compute 2nd and 4th Kramers-Moyal coefficients
            km_2 = self._km_coefficient(2, X, t_int)
            km_4 = self._km_coefficient(4, X, t_int)

            km_2_avg = np.zeros(len(op))
            km_4_avg = np.zeros(len(op))
            X = X[:-1]
            for i, b in enumerate(op):
                km_2_avg[i] = np.nanmean(km_2[(b <= X) & (X < (b + inc))])
                km_4_avg[i] = np.nanmean(km_4[(b <= X) & (X < (b + inc))])

            # Print summary data
            print('Noise statistics:')
            print(f'\tMean: {np.nanmean(noise_distribution):.4f} \t\tStd. Dev.: {np.nanstd(noise_distribution):.4f}')
            print(f'\tSkewness: {skew(noise_distribution, nan_policy="omit"):.4f}'
                  f'\tKurtosis: {kurtosis(noise_distribution, nan_policy="omit"):.4f}')

            print(f'\nNoise autocorrelation time: {act:.3f} time-steps ({act * Dt * t_int:.3f}s)')

            # Plot figures
            fig, ax = plt.subplots(2, 2, figsize=(7, 7), dpi=100)
            self._noise_plot(ax[0, 0], noise_distribution, title='Residual Distribution')
            self._qq_plot(ax[0, 1], noise_distribution, title='QQ Plot')
            self._acf_plot(ax[1, 0], acf, lags, a, b, c, act, title='Residual Autocorrelation')
            self._km_plot(ax[1, 1], km_2_avg, km_4_avg, title='KM Coefficients')

            plt.tight_layout()
            plt.show()

    def fit_diagnostics(self):
        if self.vector:
            if not (self.A1 or self.A2 or self.B11 or self.B12 or self.B21):
                print('Use fit() to fit functions before calling fit_diagnostics().')
                return

            x, y = np.meshgrid(self._ddsde._op_x_, self._ddsde._op_y_)

            if self.A1:
                z = self._ddsde._avgdriftX_
                self._print_function_diagnostics_2d(self.A1, x, y, z, name='Drift', symbol='A1')

            if self.A2:
                z = self._ddsde._avgdriftY_
                self._print_function_diagnostics_2d(self.A2, x, y, z, name='Drift', symbol='A2')

            if self.B11:
                z = self._ddsde._avgdiffX_
                self._print_function_diagnostics_2d(self.B11, x, y, z, name='Diffusion', symbol='B11')

            if self.B22:
                z = self._ddsde._avgdiffY_
                self._print_function_diagnostics_2d(self.B22, x, y, z, name='Diffusion', symbol='B22')

            if self.B12:
                z = self._ddsde._avgdiffXY_
                self._print_function_diagnostics_2d(self.B21, x, y, z, name='Cross-diffusion', symbol='B12 = B21')

        else:
            if (self.F is None) and (self.G is None):
                print('Use fit() to fit functions before calling fit_diagnostics().')
                return

            x = self._ddsde._op_

            if self.F:
                y = self._ddsde._avgdrift_
                self._print_function_diagnostics(self.F, x, y, name='Drift', symbol='F')

            if self.G:
                y = self._ddsde._avgdiff_
                self._print_function_diagnostics(self.G, x, y, name='Diffusion', symbol='G')

    def _print_function_diagnostics(self, f, x, y, name, symbol):
        n, k = len(x), len(f)

        y_fit = f(x)
        (x_, y_fit_), y_ = self._remove_outliers([x, y_fit], y)
        r2 = 1 - np.nansum((y - y_fit) ** 2) / np.nansum((y - np.nanmean(y)) ** 2)
        r2_ = 1 - np.nansum((y_ - y_fit_) ** 2) / np.nansum((y_ - np.nanmean(y_)) ** 2)

        r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
        r2_ = 1 - ((1 - r2_) * (n - 1) / (n - k - 1))

        print(f'\n{name}:\n {symbol} = {f}')
        print(f'    Adjusted R-squared : {r2:.4f}')
        print(f'    Adjusted R-squared (without outliers) : {r2_:.4f}')

    def _print_function_diagnostics_2d(self, f, x, y, z, name, symbol):
        n, k = x.size, np.count_nonzero(f)

        z_fit = f(x, y)
        (x_, y_, z_fit_), z_ = self._remove_outliers([x, y, z_fit], z)
        r2 = 1 - (np.nansum((z - z_fit) ** 2) / np.nansum((z - np.nanmean(z)) ** 2))
        r2_ = 1 - (np.nansum((z_ - z_fit_) ** 2) / np.nansum((z_ - np.nanmean(z_)) ** 2))

        r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
        r2_ = 1 - ((1 - r2_) * (n - 1) / (n - k - 1))

        print(f'\n{name}:\n {symbol} = {f}')
        print(f'    Adjusted R-squared : {r2:.4f}')
        print(f'    Adjusted R-squared (without outliers) : {r2_:.4f}')


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
