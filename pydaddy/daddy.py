import sys
import time

from collections import OrderedDict, namedtuple

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.linalg import cholesky, sqrtm, LinAlgError
import seaborn as sns
#import sdeint

import pydaddy
from pydaddy.preprocessing import Preprocessing
from pydaddy.visualize import Visualize
from pydaddy.fitters import PolyFit1D, PolyFit2D

__all__ = ['Daddy']


class Daddy(Preprocessing, Visualize):
    """
    An object of this type is returned by :class:`pydaddy.daddy.Characterize`.
    This is the main workhorse class of PyDaddy, and contains functionality to compute drift and diffusion,
    visualize results, and perform diagnostic tests. See the individual method documentation for more details.
    """

    def __init__(self, ddsde, **kwargs):
        self.vector = ddsde.vector
        self._ddsde = ddsde
        self.fft = ddsde.fft
        self.op_range = ddsde.op_range
        self.op_x_range = ddsde.op_x_range
        self.op_y_range = ddsde.op_y_range
        self.Dt = ddsde.Dt
        self.dt = ddsde.dt

        self.fast_mode = ddsde.fast_mode
        self.fitters = dict()

        if not self.vector:
            self._data_X = ddsde._X
            self._data_t = ddsde._t
            self._data_op = ddsde._op_

            self.F, self.G = ddsde.F, ddsde.G

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

            self.F1, self.F2 = ddsde.F1, ddsde.F2
            self.G11, self.G22 = ddsde.G11, ddsde.G22
            self.G12, self.G21 = ddsde.G12, ddsde.G21

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

    def export_data(self, filename=None, raw=False):
        """
        Returns a pandas dataframe containing the drift and diffusion values. Optionally, the data is also saved
        as a CSV file.

        Args
        ----
        filename : str, optional(default=None)
            If provided, the data will be saved as a CSV at the given path. Else, a dataframe will be returned.
        raw : bool, optional(default=False)
            If True, the raw, the drift and diffusion will be returned as raw unbinned data. Otherwise (default),
            drift and diffusion as binwise-average Kramers-Moyal coefficients are returned.

        Returns
        -------
        DataFrame: Pandas dataframe containing the estimated drift and diffusion coefficients.

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
                    diffusion_y=self._data_avgdiffY.flatten(),
                    diffusion_xy=self._data_avgdiffXY.flatten(),
                )

        df = pd.DataFrame(data=data_dict)

        if self.vector:
            na_rows = df[['drift_x', 'drift_y', 'diffusion_x', 'diffusion_y', 'diffusion_xy']].isna().all(axis=1)
            df = df[~na_rows]

        if filename:
            df.to_csv(filename)
        else:
            return df

    def _data(self, drift_time_scale=None, diff_time_scale=None):
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
            drift_num, diff_num = self._get_num_points(drift_time_scale, diff_time_scale)
            return Data(drift, diff, drift_num, diff_num, self._data_op)

        Data = namedtuple('Data', ('driftX', 'driftY', 'diffX', 'diffY', 'diffXY', 'diffYX', 'op_x', 'op_y'))
        driftX, driftY, diffX, diffY, diffXY, diffYX = self._get_data_from_slider(drift_time_scale, diff_time_scale)
        return Data(driftX, driftY, diffX, diffY, diffXY, diffYX, self._data_op_x, self._data_op_y)

    def _plot_data_(self,
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

    def fit(self, function_name, order=None, threshold=0.05, alpha=0, tune=False, thresholds=None, library=None,
            plot=False):
        """

        Fit analytical expressions to drift/diffusion functions using sparse regression. By default, a polynomial with
        a specified maximum degree will be fitted. Alternatively, you can also provide a library of custom functions
        for fitting.

        Args
        ----

        function_name: str,
            Name of the function to fit. Can be 'F' or 'G' for scalar; 'F1', 'F2', 'G11', 'G22', 'G12', 'G21' for vector
        order: int,
            Order (maximum degree) of the polynomial to fit.
        threshold: float, (default=0.05)
            Sparsification threshold
        tune: bool, (default=False)
            If True, the sparsification threshold will be automatically set using cross-validation.
        alpha: float, (default=0.0)
            Optional regularization term for ridge regression. Useful when data is too noisy, but has a side effect of
            shrinking the estimated coefficients when set to high values.
        thresholds: list, (default=None)
            With :code:`tune=True`, a list of thresholds over which to search for can optionally be provided. If not
            present, this will be chosen automatically as the range between the minimum and maximum coefficients in
            the initial fit.
        library: list, (default=None)
            A custom library of non-polynomial functions can optionally be provided. If provided, the functions will be
            fitted as a sparse linear combination of the terms in the library.

        Returns
        -------

        res : fitters.Poly1D or fitters.Poly2D object, representing the fitted polynomial.

        """

        if not (order or library):
            raise TypeError('You should either specify the order of the polynomial, or provide a library.')

        if library:
            order = 1

        if self.vector:
            x = np.stack((self._ddsde._Mx, self._ddsde._My), axis=1)
            if function_name == 'F1':
                x = x[:-self.Dt]
                y = self._ddsde._driftX_
            elif function_name == 'F2':
                x = x[:-self.Dt]
                y = self._ddsde._driftY_
            elif function_name == 'G11':
                x = x[:-self.dt]
                y = self._ddsde._diffusionX_
            elif function_name == 'G22':
                x = x[:-self.dt]
                y = self._ddsde._diffusionY_
            elif function_name in ['G12', 'G21']:
                x = x[:-self.dt]
                y = self._ddsde._diffusionXY_
            else:
                raise TypeError('Invalid function name for vector analysis')

            # Handle missing values (NaNs) if present
            nan_idx = np.isnan(x).any(axis=1) | np.isnan(y)
            x = x[~nan_idx]
            y = y[~nan_idx]

            fitter = PolyFit2D(max_degree=order, threshold=threshold, alpha=alpha, library=library)
        else:
            x = self._ddsde._X[:-self.Dt]
            if function_name == 'G':
                y = self._ddsde._diffusion_
            elif function_name == 'F':
                y = self._ddsde._drift_
            else:
                raise TypeError('Invalid function name for scalar analysis')

            # Handle missing values (NaNs) if present
            nan_idx = np.isnan(x) | np.isnan(y)
            x = x[~nan_idx]
            y = y[~nan_idx]

            fitter = PolyFit1D(max_degree=order, threshold=threshold, alpha=alpha, library=library)

        if tune:
            res = fitter.tune_and_fit(x, y, thresholds, plot=plot)
        else:
            res = fitter.fit(x, y)

        setattr(self, function_name, res)
        self.fitters[function_name] = fitter

        if function_name in ['G12', 'G21']:
            self.G12 = res
            self.G21 = res

            self.fitters['G12'] = fitter
            self.fitters['G21'] = fitter

        return res

    def simulate(self, t_int, timepoints, x0=None):
        """
        Generate simulated time-series with the fitted SDE model.

        Generates a simulated timeseries, with specified sampling time and duration, based on the SDE model discovered
        by PyDaddy. The drift and diffusion functions should be fit using fit() function before using simulate().

        Args
        ----
        t_int : float
            Sampling time for the simulated time-series
        timepoints : int
            Number of time-points to simulate
        x0 : float (scalar) or list of two floats (vector), (default=None)
            Initial condition. If no value is passed, 0 ([0, 0] for vector) is taken as the initial condition.

        Returns
        -------
        x : Simulated timeseries with  `timepoints` timepoints.

        """

        try:
            sdeint = __import__('sdeint')
        except ModuleNotFoundError:
            print("Package sdeint not found")
            print("Please install sdeint using using:")
            print("python -m pip install sdeint")
            return None


        tspan = np.arange(0, t_int * timepoints, step=t_int)

        if self.vector:
            assert (self.F1 and self.F2 and self.G11 and self.G22 and self.G12), \
                """ Use fit() function to fit F1, F2, G11, G12, G21, G22 before using simulate(). """

            def F(x, t):
                return np.array([self.F1(*x), self.F2(*x)])

            if np.count_nonzero(self.G12) != 0:
                print('Warning: Cross-diffusion terms are present. Simulation results may be inaccurate.')

                def G(x, t):
                    # print(x)
                    G_ = np.array([[self.G11(*x), self.G12(*x)],
                                   [self.G21(*x), self.G22(*x)]])
                    try:
                        return sqrtm(G_)
                    except LinAlgError:
                        print(x)
                        print(G_)
                        raise LinAlgError('Simulation failed. Matrix G is not positive-definite.')
                    except ValueError:
                        # return np.array([[np.nan, np.nan], [np.nan, np.nan]])
                        raise ValueError('Simulation failed. System may be unstable.')
            else:
                def G(x, t):
                    return np.diag([np.sqrt(np.abs(self.G11(*x))), np.sqrt(np.abs(self.G22(*x)))])

            if x0 is None:
                x0 = np.array([0., 0.])

            # x = sdeint.itoint(f=F, G=G, y0=x0, tspan=tspan).T
            x = sdeint.itoEuler(f=F, G=G, y0=x0, tspan=tspan).T

        else:
            assert (self.F and self.G), \
                'Use fit() function to fit F, G before using simulate().'

            if x0 is None:
                x0 = 0.

            def F(x, t):
                return self.F(x)

            def G(x, t):
                return np.sqrt(np.abs(self.G(x)))

            x = sdeint.itoSRI2(f=F, G=G, y0=x0, tspan=tspan)

        return x

    def summary(self, start=0, end=1000, kde=True, tick_size=12, title_size=15, label_size=15, label_pad=8, n_ticks=3,
                ret_fig=False, **plot_text):
        """
        Print summary of data and show summary plots chart. (This is the same summary plot produced by Characterize().)

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
            fields = ['x range', 'x mean',
                      '|x| range', '|x| mean',
                      'Autocorr time (x)',
                      ]

            values = [self._get_data_range(self._data_X), round(np.nanmean(self._data_X), 3),
                      self._get_data_range(np.sqrt(self._data_X ** 2)),
                      round(np.nanmean(np.sqrt(self._data_X ** 2)), 3),
                      self.autocorrelation_time,
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
            fields = ['x range', 'x mean',
                      'y range', 'y mean',
                      '|(x, y)| range', '|(x, y)| mean',
                      'Autocorr time (x, y, |(x, y)|^2)',
                      ]

            values = [self._get_data_range(self._data_Mx), round(np.nanmean(self._data_Mx), 3),
                      self._get_data_range(self._data_My), round(np.nanmean(self._data_My), 3),
                      self._get_data_range(self._data_M),
                      round(np.nanmean(np.sqrt(self._data_Mx ** 2 + self._data_My ** 2)), 3),
                      (self._ddsde._act_mx, self._ddsde._act_my, self.autocorrelation_time),
                      ]
            values = list(map(str, values))
            summary = []
            for i in range(len(fields)):
                summary.append(fields[i])
                summary.append(values[i])
            summary_format = ("| {:<30} : {:^15}" * 1 + "|\n") * int(len(fields) / 1)
            print(summary_format.format(*summary))
            if self._ddsde.F1:
                print(f'Drift (F1): {self._ddsde.F1}')
            if self._ddsde.F2:
                print(f'Drift (F2): {self._ddsde.F2}')
            if self._ddsde.G11:
                print(f'Diffusion (G11): {self._ddsde.G11}')
            if self._ddsde.G22:
                print(f'Diffusion (G22): {self._ddsde.G22}')
            if self._ddsde.G21:
                print(f'Cross diffusion (G12, G21): {self._ddsde.G21}')
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

    def _timeseries(self,
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

    def _histogram(self,
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
        histogram chart : matplotlib.pyplot.figure
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

    def autocorrelation(self):
        """ Show the autocorrelation plot of the data. """
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
        Show an interactive figure of the drift function. The bin-wise averaged estimates of the drift will be shown.
        If a polynomial has already been fitted with fit() function, a line (scalar)/surface (vector) plot of the
        fitted function will also be shown.

        Args
        ----
        limits: tuple, (default=None)
            If specified, sets the y-axis limits for the drift function. Useful to get a clearer view when there are
            outliers.

        polar: bool, (default=False):
            If True, plot the drift function only within a unit circle. Useful to get a better visualization in
            situations where \|M\| has an upper bound. (Used only in vector case).

        **plot_text: dict:
            To customize the captions, labels and layout of the plot, plot parameters can be passed as a dict. Available
            options are given below:

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

    def diffusion(self, slider_timescales=None, limits=None, polar=False, **plot_text):
        """
        Show an interactive figure of the diffusion function. The bin-wise averaged estimates of the drift will be shown.
        If a polynomial has already been fitted with fit() function, a line (scalar)/surface (vector) plot of the
        fitted function will also be shown.

        Args
        ----
        limits: tuple, (default=None)
            If specified, sets the y-axis limits for the diffusion function. Useful to get a clearer view when there are
            outliers.

        polar: bool, (default=False):
            If True, plot the diffusion function only within a unit circle. Useful to get a better visualization in
            situations where \|M\| has an upper bound. (Used only in vector case).

        **plot_text: dict:
            To customize the captions, labels and layout of the plot, plot parameters can be passed as a dict. Available
            options are given below:

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
        Show an interactive figure of the cross-diffusion function (only for vector data).
        The bin-wise averaged estimates of the drift will be shown.
        If a polynomial has already been fitted with fit() function, a line (scalar)/surface (vector) plot of the
        fitted function will also be shown.

        Args
        ----
        limits: tuple, (default=None)
            If specified, sets the y-axis limits for the cross diffusion function. Useful to get a clearer view
            when there are outliers.

        polar: bool, (default=False):
            If True, plot the cross diffusion function only within a unit circle. Useful to get a better visualization
            in situations where \|M\| has an upper bound. (Used only in vector case).

        **plot_text: dict:
            To customize the captions, labels and layout of the plot, plot parameters can be passed as a dict. Available
            options are given below:

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

    def _visualize(self, drift_time_scale=None, diff_time_scale=None):
        """
        Display drift and diffusion plots for a time scale.

        Args
        ----

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
            plt.xlabel('x')

            # Drift
            fig3 = plt.figure(dpi=150, figsize=(5, 5))
            plt.suptitle("Drift")
            plt.errorbar(self._data_op, drift, yerr=drift_ebar, fmt='o')
            plt.xlabel('x')
            plt.ylabel("F")
            plt.xlim([min(self._data_op), max(self._data_op)])
            plt.xticks(np.linspace(min(self._data_op), max(self._data_op), 5))

            # Diffusion
            fig4 = plt.figure(dpi=150, figsize=(5, 5))
            plt.suptitle("Diffusion")

            plt.errorbar(self._data_op, diff, yerr=diff_ebar, fmt='o')
            plt.xlim([min(self._data_op), max(self._data_op)])
            plt.xticks(np.linspace(min(self._data_op), max(self._data_op), 5))
            plt.xlabel("x")
            plt.ylabel('$G^{2}$')

        else:
            driftX, driftY, diffX, diffY, diffXY, diffYX = self._get_data_from_slider(drift_time_scale, diff_time_scale)
            fig1, _ = self._plot_3d_hisogram(self._data_Mx, self._data_My, title='PDF', xlabel="$x$", tick_size=12,
                                             label_size=12, title_size=12, r_fig=True)

            fig5, _ = self._plot_data(driftX,
                                      plot_plane=False,
                                      title='Drift',
                                      z_label='$F_{1}(m)$',
                                      tick_size=12,
                                      label_size=14,
                                      title_size=16)

            fig4, _ = self._plot_data(driftY,
                                      plot_plane=False,
                                      title='Drift',
                                      z_label='$F_{2}(m)$',
                                      tick_size=12,
                                      label_size=14,
                                      title_size=16)

            fig3, _ = self._plot_data(diffX,
                                      plot_plane=False,
                                      title='Diffusion',
                                      z_label='$G_{11}(m)$',
                                      tick_size=12,
                                      label_size=14,
                                      title_size=16)

            fig2, _ = self._plot_data(diffY,
                                      plot_plane=False,
                                      title='Diffusion',
                                      z_label='$G_{22}(m)$',
                                      tick_size=12,
                                      label_size=14,
                                      title_size=16)

            fig6, _ = self._plot_data(diffXY,
                                      plot_plane=False,
                                      title='Cross-Diffusion',
                                      z_label='$G_{12}(m)$',
                                      tick_size=12,
                                      label_size=14,
                                      title_size=16)

            fig7, _ = self._plot_data(diffYX,
                                      plot_plane=False,
                                      title='Cross-Diffusion',
                                      z_label='$G_{21}(m)$',
                                      tick_size=12,
                                      label_size=14,
                                      title_size=16)
        return None

    def noise_diagnostics(self, loc=None):
        """
        Perform diagostics on the noise-residual, to ensure that all assumptions for SDE estimation are met.
        Generates a plot with:

          - Distribution (1D or 2D histogram) of the residuals, and their QQ-plots against a theoretically expected
            Gaussian. The residual distribution is expected to be a Gaussian.
          - Autocorrelation of the residuals. The autocorrelation time should be close to 0.
          - Plot of the 2nd versus 4th jump moments. This plot should be a straight line. (Only for scalar data.)

        Args
        ----
        loc: tuple, (default=None)
            The residual distribution is computed within a bin; (0, 0) by default. To compute the residual distribution
            in a different bin, specify the location of the bin as a tuple of floats. If loc='mode' is passed, the mode
            of the data distribution is used.
        """

        if self.vector:
            X, Y = self._ddsde._Mx, self._ddsde._My
            Dt = self._ddsde.Dt
            inc_x, inc_y = self._ddsde.inc_x, self._ddsde.inc_y
            t_int = self._ddsde.t_int
            op_x, op_y = self._ddsde._op_x_, self._ddsde._op_y_
            avg_drift_x, avg_drift_y = self._ddsde._avgdriftX_, self._ddsde._avgdriftY_
            res_x, res_y = self._ddsde._residual_timeseries_vector(
                X=X, Y=Y, Dt=Dt,
                bins_x=op_x, bins_y=op_y,
                inc_x=inc_x, inc_y=inc_y,
                avg_drift_x=avg_drift_x, avg_drift_y=avg_drift_y,
                t_int=t_int
            )

            if loc is None:
                loc = (0, 0)
            elif loc == 'mode':
                H, edges = np.histogramdd([X, Y], bins=[op_x, op_y])
                idx_x, idx_y = np.unravel_index(H.argmax(), H.shape)
                loc = [op_x[idx_x], op_y[idx_y]]

            noise_dist_x = res_x[
                (loc[0] - inc_x <= X[:-Dt]) & (X[:-Dt] < loc[0]) & (loc[1] - inc_y <= Y[:-Dt]) & (Y[:-Dt] < loc[1])]
            noise_dist_y = res_y[
                (loc[0] - inc_x <= X[:-Dt]) & (X[:-Dt] < loc[0]) & (loc[1] - inc_y <= Y[:-Dt]) & (Y[:-Dt] < loc[1])]
            noise_corr = np.ma.corrcoef([np.ma.masked_invalid(noise_dist_x),
                                         np.ma.masked_invalid(noise_dist_y)])

            if noise_dist_x.size <= 1 or noise_dist_y.size <= 1:
                print(f'There are no data points near the specified location ({loc[0]}, {loc[1]}).\n'
                      f'Specify a different location using the loc argument, or use loc=None to use '
                      f'the mode of the data distribution.')
                return

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
            ayd = fig.add_subplot(gs[:2, 0], projection='3d')
            ax_acf = fig.add_subplot(gs[:2, 1])
            ax_qqx = fig.add_subplot(gs[2:, 0])
            ax_qqy = fig.add_subplot(gs[2:, 1])
            ax_corr = inset_axes(ayd, width='30%', height='39%', loc='upper left')

            self._noise_plot_2d(ayd, noise_dist_x, noise_dist_y, title='Residual Distribution')
            self._matrix_plot(ax_corr, noise_corr)
            self._acf_plot_multi(ax_acf, acf_x, acf_y, lags, act_x, act_y, title='Autocorrelation: $\\eta_x, \\eta_y$')
            self._qq_plot(ax_qqx, noise_dist_x, title='QQ Plot: $\\eta_x$')
            self._qq_plot(ax_qqy, noise_dist_y, title='QQ Plot: $\\eta_y$')

            plt.tight_layout()
            plt.show()
        else:
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

            if loc is None:
                loc = 0
            elif loc == 'mode':
                H, edges = np.histogram(X, bins=op)
                loc = op[H.argmax()]

            noise_distribution = residual[(loc <= X[:-Dt]) & (X[:-Dt] < loc + inc)]

            if noise_distribution.size <= 1:
                print(f'There are no data points near the specified location ({loc}).\n'
                      f'Specify a different location using the loc argument, or use loc=None to use '
                      f'the mode of the data distribution.')
                return

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

    def _fit_diagnostics(self):
        """
        Show diagnostics about the quality of the fit.
        Shows the fitted functions (along with confidence intervals for each coefficient), and the adjusted R-squared
        errors (computed against the bin-wise averages).

        WARNING: Since the R2 values are computed against the binwise averages, R2 can be low even when the fits are
        accurate but the bin-wise estimates are off.
        """
        if self.vector:
            if not (self.F1 or self.F2 or self.G11 or self.G12 or self.G21):
                print('Use fit() to fit functions before calling fit_diagnostics().')
                return

            x, y = np.meshgrid(self._ddsde._op_x_, self._ddsde._op_y_)

            if self.F1:
                z = self._ddsde._avgdriftX_
                self._print_function_diagnostics_2d(self.F1, x, y, z, name='Drift', symbol='F1')

            if self.F2:
                z = self._ddsde._avgdriftY_
                self._print_function_diagnostics_2d(self.F2, x, y, z, name='Drift', symbol='F2')

            if self.G11:
                z = self._ddsde._avgdiffX_
                self._print_function_diagnostics_2d(self.G11, x, y, z, name='Diffusion', symbol='G11')

            if self.G22:
                z = self._ddsde._avgdiffY_
                self._print_function_diagnostics_2d(self.G22, x, y, z, name='Diffusion', symbol='G22')

            if self.G12:
                z = self._ddsde._avgdiffXY_
                self._print_function_diagnostics_2d(self.G21, x, y, z, name='Cross-diffusion', symbol='G12 = G21')

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

    def model_diagnostics(self, oversample=1):
        """
        Perform model self-consistency diagnostics.
        Generates a simulated time series with the same length and sampling interval as the original time series,
        and re-estimates the drift and diffusion from the simulated time series. The re-estimated drift and diffusion
        should match the original estimates.

        Args
        ----

        oversample: int, (default=1)
            Factor by which to oversample while simulating the SDE. If provided, the SDE will be simulated at
            `t_int / oversample` and then subsampled to `t_int`. This is useful when `t_int` is large enough to cause
            large errors in the SDE simulation.

        The following are plotted:
          - Histogram of the original time series overlaid with that of the simulated time series.
          - Drift and diffusion of the original time series overlaid with that of the simulated time series.
        """

        try:
            sdeint = __import__('sdeint')
        except ModuleNotFoundError:
            print("Package sdeint not found")
            print("Please install sdeint using using:")
            print("python -m pip install sdeint")
            return None

        if self.vector:
            print('Generating simulated timeseries ...')
            timepoints = self._data_Mx.shape[0]
            x = self.simulate(t_int=self._ddsde.t_int / oversample, timepoints=timepoints * oversample)
            x = x[:, ::oversample]

            print('Re-estimating drift and diffusion from simulated timeseries ...')
            ddsde = pydaddy.Characterize(data=x, t=self._ddsde.t_int, Dt=self.Dt, dt=self.dt, show_summary=False)

            F1hat = ddsde.fit('F1',
                              order=self.fitters['F1'].max_degree,
                              threshold=self.fitters['F1'].threshold,
                              alpha=self.fitters['F1'].alpha,
                              library=self.fitters['F1'].library)

            F2hat = ddsde.fit('F2',
                              order=self.fitters['F2'].max_degree,
                              threshold=self.fitters['F2'].threshold,
                              alpha=self.fitters['F2'].alpha,
                              library=self.fitters['F2'].library)

            G11hat = ddsde.fit('G11',
                               order=self.fitters['G11'].max_degree,
                               threshold=self.fitters['G11'].threshold,
                               alpha=self.fitters['G11'].alpha,
                               library=self.fitters['G11'].library)

            G22hat = ddsde.fit('G22',
                               order=self.fitters['G22'].max_degree,
                               threshold=self.fitters['G22'].threshold,
                               alpha=self.fitters['G22'].alpha,
                               library=self.fitters['G22'].library)

            G12hat = ddsde.fit('G12',
                               order=self.fitters['G12'].max_degree,
                               threshold=self.fitters['G12'].threshold,
                               alpha=self.fitters['G12'].alpha,
                               library=self.fitters['G12'].library)

            lags, acf_actual = self._ddsde._acf(self._data_M,
                                                t_lag=min(len(x[0]), 1000))

            lags, acf_est = self._ddsde._acf(np.sqrt(x[0] ** 2 + x[1] ** 2),
                                             t_lag=min(len(x[0]), 1000))

            fig = plt.figure(figsize=(12, 12), dpi=100)
            gs = fig.add_gridspec(3, 3)
            ax_mxmy = fig.add_subplot(gs[0, 0], projection='3d')
            ax_modm = fig.add_subplot(gs[1, 0])
            ax_acf = fig.add_subplot(gs[2, 0])
            ax_f1 = fig.add_subplot(gs[0, 1], projection='3d')
            ax_f2 = fig.add_subplot(gs[0, 2], projection='3d')
            ax_g11 = fig.add_subplot(gs[1, 1], projection='3d')
            ax_g22 = fig.add_subplot(gs[2, 2], projection='3d')
            ax_g12 = fig.add_subplot(gs[1, 2], projection='3d')
            ax_g21 = fig.add_subplot(gs[2, 1], projection='3d')

            self._show_histograms_2d(ax_mxmy, [self._data_Mx, self._data_My], x, title='$x$ histogram')
            self._show_histograms_1d(ax_modm, self._data_M, np.sqrt(x[0] ** 2 + x[1] ** 2), xlabel='$|\mathbf{x}|$',
                                     title='$|\mathbf{x}|$ histogram')

            self._acf_plot_multi(ax_acf, acf_actual, acf_est, lags, None, None,
                                 label1='Original', label2='Reestimated',
                                 title='$|\mathbf{x}|$ autocorrelation')

            self._show_functions_2d(ax_f1, self.F1, F1hat, title='$F_{1}$')
            self._show_functions_2d(ax_f2, self.F2, F2hat, title='$F_{2}$')

            self._show_functions_2d(ax_g11, self.G11, G11hat, title='$G_{11}$')
            self._show_functions_2d(ax_g22, self.G22, G22hat, title='$G_{22}$')
            self._show_functions_2d(ax_g12, self.G12, G12hat, title='$G_{12}$')
            self._show_functions_2d(ax_g21, self.G21, G12hat, title='$G_{21}$')

            print(f'F1:')
            print(f'  Original: {self.F1}')
            print(f'  Reestimated: {F1hat}')

            print(f'F2:')
            print(f'  Original: {self.F2}')
            print(f'  Reestimated: {F2hat}')

            print(f'G11:')
            print(f'  Original: {self.G11}')
            print(f'  Reestimated: {G11hat}')

            print(f'G22:')
            print(f'  Original: {self.G22}')
            print(f'  Reestimated: {G22hat}')

            print(f'G12 / G21:')
            print(f'  Original: {self.G12}')
            print(f'  Reestimated: {G12hat}')

        else:
            # Generate simulated time-series
            timepoints = self._data_X.shape[0]
            x = self.simulate(t_int=self._ddsde.t_int / oversample, timepoints=timepoints * oversample)
            x = x[::oversample]


            ddsde = pydaddy.Characterize(data=[x], t=self._ddsde.t_int, Dt=self.Dt, dt=self.dt, show_summary=False)

            Fhat = ddsde.fit('F', order=self.fitters['F'].max_degree,
                             threshold=self.fitters['F'].threshold,
                             alpha=self.fitters['F'].alpha,
                             library=self.fitters['F'].library)

            Ghat = ddsde.fit('G', order=self.fitters['G'].max_degree,
                             threshold=self.fitters['G'].threshold,
                             alpha=self.fitters['G'].alpha,
                             library=self.fitters['G'].library)

            # xs = np.linspace(np.nanmin(self._data_X), np.nanmax(self._data_X), 100)
            fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
            self._show_histograms_1d(ax[0], self._data_X, x, xlabel='$x$', title='Histogram')
            self._show_functions_1d(ax[1], self.op, self.F, Fhat, ylabel='F', title='Drift')
            self._show_functions_1d(ax[2], self.op, self.G, Ghat, ylabel='G', title='Diffusion')

            print('Drift:')
            print(f'    Original: {self.F}')
            print(f'    Reestimated: {Fhat}')
            print('\nDiffusion:')
            print(f'    Original: {self.G}')
            print(f'    Reestimated: {Ghat}')

        plt.tight_layout()
        plt.show()

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
