import numpy as np
import warnings
import pkg_resources
import tqdm
from scipy.stats import iqr
from pydaddy.analysis import AutoCorrelation
from pydaddy.analysis import GaussianTest
from pydaddy.preprocessing import Preprocessing
from pydaddy.preprocessing import InputError
from pydaddy.daddy import Daddy
from pydaddy.fitters import PolyFit1D, PolyFit2D

warnings.filterwarnings("ignore")

__all__ = ['Characterize', 'load_sample_data', 'load_sample_dataset']


class Main(Preprocessing, GaussianTest, AutoCorrelation):
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
            bins=None,
            inc=None,
            inc_x=None,
            inc_y=None,
            fft=True,
            slider_timescales=None,
            n_trials=1,
            show_summary=True,
            drift_threshold=None,
            diff_threshold=None,
            drift_degree=5,
            diff_degree=5,
            drift_alpha=0,
            diff_alpha=0,
            fast_mode=False,
            **kwargs):

        self._data = data
        self._t = t
        self.Dt = Dt

        self.t_lag = t_lag
        self.inc = inc
        self.inc_x = inc_x
        self.inc_y = inc_y
        self.dt = dt
        self.fft = fft
        self.n_trials = n_trials
        self._show_summary = show_summary

        # self.drift_order = None
        # self.diff_order = None

        self.op_range = None
        self.op_x_range = None
        self.op_y_range = None
        if bins:
            self.bins = bins
        elif not (self.inc or (self.inc_x and self.inc_y)):
            self.bins = self._autobins()
        else:
            self.bins = None
        self.slider_timescales = slider_timescales

        self.fast_mode = fast_mode

        self.drift_threshold = drift_threshold
        self.drift_degree = drift_degree
        self.drift_alpha = drift_alpha

        self.diff_threshold = diff_threshold
        self.diff_degree = diff_degree
        self.diff_alpha = diff_alpha

        """
        # When t_lag is greater than timeseries length, reassign its value as length of data
        if self.t_lag > len(data[0]):
            print('Warning : t_lag is greater that the length of data; setting t_lag as {}\n'.format(len(data[0]) - 1))
            self.t_lag = len(data[0]) - 1
        """

        self.__dict__.update(kwargs)
        Preprocessing.__init__(self)
        GaussianTest.__init__(self)
        AutoCorrelation.__init__(self)
        # SDE.__init__(self)

        # if t is None and t_int is None:
        #	raise InputError("Characterize(data, t, t_int)","Missing data. Either 't' ot 't_int' must be given, both cannot be None")

        return None

    def _autobins(self):
        """ Optimal number of bins using Freedman-Diaconis rule. """

        binwidth_x = 2 * iqr(self._data[0], nan_policy='omit') / np.cbrt(len(self._data[0]))
        n = int((np.nanmax(self._data[0]) - np.nanmin(self._data[0])) / binwidth_x)
        if len(self._data) > 1:  # Vector
            binwidth_y = 2 * iqr(self._data[1], nan_policy='omit') / np.cbrt(len(self._data[1]))
            n_y = int((np.nanmax(self._data[1]) - np.nanmin(self._data[1])) / binwidth_y)
            n = max(n, n_y)
        # print(f'Number of bins chosen: {n}')
        return n

    def _slider_data(self, Mx, My, update=False):
        if update:
            drift_data_dict = self._drift_slider
            diff_data_dict = self._diff_slider
            cross_diff_dict = self._cross_diff_slider
            if not self.vector:
                ebar_drift_dict = self._scalar_drift_ebars
                ebar_diff_dict = self._scalar_diff_ebars
                num_drift_dict = self._scalar_diff_nums
                num_diff_dict = self._scalar_drift_nums
        else:
            drift_data_dict = dict()
            diff_data_dict = dict()
            cross_diff_dict = dict()
            if not self.vector:
                ebar_drift_dict = dict()
                ebar_diff_dict = dict()
                num_drift_dict = dict()
                num_diff_dict = dict()
        time_scale_list = sorted(map(int, set(self.slider_timescales).union([self.dt, self.Dt])))
        for time_scale in tqdm.tqdm(time_scale_list, desc='Generating Slider data'):
            if update and time_scale in self._drift_slider.keys():
                continue
            if self.vector:
                dd = self._vector_drift_diff(
                        Mx,
                        My,
                        inc_x=self.inc_x,
                        inc_y=self.inc_y,
                        t_int=self.t_int,
                        Dt=time_scale,
                        dt=time_scale,
                        fast_mode=True,
                        drift_threshold=None,
                        drift_degree=None,
                        drift_alpha=None,
                        diff_threshold=None,
                        diff_degree=None,
                        diff_alpha=None
                    )
                #avgdriftX = dd.avgdriftX
                #avgdriftY = dd.avgdriftY
                #avgdiffX = dd.avgdiffX
                #avgdiffY = dd.avgdiffY
                #avgdiffXY = dd.avgdiffXY
                #avgdiffYX = dd.avgdiffYX
                #op_x = dd.op_x
                #op_y = dd.op_y
                self.A1, self.A2, self.B11, self.B22, self.B12, self.B21 = [None]*6
                #if time_scale == 1:
                #    self._driftX_ = dd.driftX
                #    self._driftY_ = dd.driftY
                #    self._diffusionX_ = dd.diffusionX
                #    self._diffusionY_ = dd.diffusionY
                #    self._diffusionXY_ = dd.diffusionXY
                #    self._diffusionYX_ = dd.diffusionYX
                #_, _, _, _, _, _, \
                #avgdriftX, avgdriftY, avgdiffX, avgdiffY, avgdiffXY, avgdiffYX, op_x, op_y = \
                #    self._vector_drift_diff(Mx,
                #                            My,
                #                            inc_x=self.inc_x,
                #                            inc_y=self.inc_y,
                #                            t_int=self.t_int,
                #                            Dt=time_scale,
                #                            dt=time_scale)
                drift_data = [dd.avgdriftX / self.n_trials, dd.avgdriftY / self.n_trials, dd.op_x, dd.op_y]
                diff_data = [dd.avgdiffX / self.n_trials, dd.avgdiffY / self.n_trials, dd.op_x, dd.op_y]
                cross_diff_data = [dd.avgdiffXY / self.n_trials, dd.avgdiffYX / self.n_trials, dd.op_x, dd.op_y]
            else:
                dd = self._drift_and_diffusion(self._X,
                                          self.t_int,
                                          Dt=time_scale,
                                          dt=time_scale,
                                          inc=self.inc,
                                          fast_mode=True,
                                          drift_threshold=None,
                                          drift_degree=None,
                                          drift_alpha=None,
                                          diff_threshold=None,
                                          diff_degree=None,
                                          diff_alpha=None)

                #_, _, avgdiff, avgdrift, op, drift_ebar, diff_ebar, drift_num, diff_num, _, _ \
                #    = self._drift_and_diffusion(Mx, t_int=self.t_int, Dt=time_scale, dt=time_scale, inc=self.inc)
                self.F, self.G = None, None
                drift_data = [dd.avgdrift / self.n_trials, dd.op]
                diff_data = [dd.avgdiff / self.n_trials, dd.op]
                ebar_drift_dict[time_scale] = dd.drift_ebar
                ebar_diff_dict[time_scale] = dd.diff_ebar
                num_drift_dict[time_scale] = dd.drift_num
                num_diff_dict[time_scale] = dd.diff_num

            drift_data_dict[time_scale] = drift_data
            diff_data_dict[time_scale] = diff_data

            if self.vector:
                cross_diff_dict[time_scale] = cross_diff_data

        if self.vector:
            return drift_data_dict, diff_data_dict, cross_diff_dict
        self._avaiable_timescales = time_scale_list
        return drift_data_dict, diff_data_dict, ebar_drift_dict, ebar_diff_dict, num_drift_dict, num_diff_dict

    def fit(self, function_name, order=None, threshold=0.05, alpha=0, tune=False, thresholds=None, library=None,
            plot=False):

        if not (order or library):
            raise TypeError('You should either specify the order of the polynomial, or provide a library.')

        if library:
            order = 1

        fmap = {
            'F': 'drift',
            'G': 'diff',
            # 'Gsquare': 'diff',
            'A1': 'driftX',
            'A2': 'driftY',
            'B11': 'diffX',
            'B12': 'diffXY',
            'B21': 'diffYX',
            'B22': 'diffY'
        }
        if function_name not in fmap.keys():
            print("Invalid function name")
            return None

        if self.vector:
            # x = [self._Mx[:-1], self._My[:-1]]
            #x = np.stack((self._Mx[:-1], self._My[:-1]), axis=1)
            x = np.stack((self._Mx, self._My), axis=1)
            if function_name == 'A1':
                x = x[:-self.Dt]
                y = self._driftX_
            elif function_name == 'A2':
                x = x[:-self.Dt]
                y = self._driftY_
            elif function_name == 'B11':
                x = x[:-self.dt]
                y = self._diffusionX_
            elif function_name == 'B22':
                x = x[:-self.dt]
                y = self._diffusionY_
            elif function_name in ['B12', 'B21']:
                x = x[:-self.dt]
                y = self._diffusionXY_
            else:
                raise TypeError('Invalid function name for vector analysis')

            # Handle missing values (NaNs) if present
            # nan_idx = np.isnan(x[0]) | np.isnan(x[1]) | np.isnan(y)
            # x[0] = np.delete(x[0], nan_idx)
            # x[1] = np.delete(x[1], nan_idx)
            # y = np.delete(y, nan_idx)
            nan_idx = np.isnan(x).any(axis=1) | np.isnan(y)
            x = x[~nan_idx]
            y = y[~nan_idx]

            fitter = PolyFit2D(max_degree=order, threshold=threshold, alpha=alpha, library=library)
        else:
            x = self._X[:-1]
            if function_name == 'G':
                # y = self._diffusion(self._X, t_int=self.t_int, dt=1)
                # F = self.fit('F', order=5, tune=True)
                # y = self._diffusion_from_residual(self._X, F=F, t_int=self.t_int, dt=1)
                y = self._diffusion_
            elif function_name == 'F':
                y = self._drift_
            else:
                raise TypeError('Invalid function name for scalar analysis')

            # Handle missing values (NaNs) if present
            nan_idx = np.isnan(x) | np.isnan(y)
            x = x[~nan_idx]
            y = y[~nan_idx]

            fitter = PolyFit1D(max_degree=order, threshold=threshold, alpha=alpha, library=library)
        #
        if tune:
            res = fitter.tune_and_fit(x, y, thresholds, plot=plot)
        else:
            res = fitter.fit(x, y)

        setattr(self, function_name, res)
        if function_name in ['B12', 'B21']:
            self.B12 = res
            self.B21 = res

        return res

    def __call__(self, data, t=1, Dt=None, **kwargs):
        self.__dict__.update(kwargs)
        # if t is None and t_int is None:
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
            #if not self._is_valid_slider_timescale_list(self.slider_timescales):
            self._drift_slider = dict()
            self._diff_slider = dict()
            self._scalar_drift_ebars = dict()
            self._scalar_diff_ebars = dict()
            self._scalar_drift_nums = dict()
            self._scalar_diff_nums = dict()

            self._diffusion_, self._drift_, self._avgdiff_, self._avgdrift_, self._op_, self._drift_ebar, self._diff_ebar, \
            self._drift_num, self._diff_num, F, G = self._drift_and_diffusion(self._X,
                                                                              self.t_int,
                                                                              Dt=self.Dt,
                                                                              dt=self.dt,
                                                                              inc=self.inc,
                                                                              fast_mode=self.fast_mode,
                                                                              drift_threshold=self.drift_threshold,
                                                                              drift_degree=self.drift_degree,
                                                                              drift_alpha=self.drift_alpha,
                                                                              diff_threshold=self.diff_threshold,
                                                                              diff_degree=self.diff_degree,
                                                                              diff_alpha=self.diff_alpha)
            self._avgdiff_ = self._avgdiff_ / self.n_trials
            self._avgdrift_ = self._avgdrift_ / self.n_trials
            self._drift_slider[self.Dt] = [self._avgdrift_, self._op_]
            self._diff_slider[self.dt] = [self._avgdiff_, self._op_]
            self._scalar_drift_ebars[self.Dt] = self._drift_ebar
            self._scalar_diff_ebars[self.dt] = self._diff_ebar
            self._scalar_drift_nums[self.dt] = self._drift_num
            self._scalar_diff_nums[self.dt] = self._diff_num
            self.F = F
            self.G = G
            #else:
            #    # FIXME self._drift_ and self._diffusion_ variable need to be set here.
            #    self._drift_slider, self._diff_slider, self._scalar_drift_ebars, self._scalar_diff_ebars, \
            #    self._scalar_drift_nums, self._scalar_diff_nums = self._slider_data(self._X, None)
            #    self._avgdrift_, self._op_ = self._drift_slider[self.Dt]
            #    self._avgdiff_ = self._diff_slider[self.dt][0]
            #    self._drift_ebar = self._scalar_drift_ebars[self.Dt]
            #    self._diff_ebar = self._scalar_diff_ebars[self.dt]
            #    self._drift_num = self._scalar_drift_nums[self.Dt]
            #    self._diff_num = self._scalar_diff_nums[self.Dt]
            self._cross_diff_slider = None

        else:
            #if not self._is_valid_slider_timescale_list(self.slider_timescales):
            self._drift_slider = dict()
            self._diff_slider = dict()
            self._cross_diff_slider = dict()
            self._driftX_, self._driftY_, self._diffusionX_, self._diffusionY_, \
                self._diffusionXY_, self._diffusionYX_, \
                self._avgdriftX_, self._avgdriftY_, \
                self._avgdiffX_, self._avgdiffY_, self._avgdiffXY_, self._avgdiffYX_, \
                self._op_x_, self._op_y_, \
                self.A1, self.A2, self.B11, self.B22, self.B12, self.B21 = self._vector_drift_diff(
                    self._Mx,
                    self._My,
                    inc_x=self.inc_x,
                    inc_y=self.inc_y,
                    t_int=self.t_int,
                    Dt=self.Dt,
                    dt=self.dt,
                    fast_mode=self.fast_mode,
                    drift_threshold=self.drift_threshold,
                    drift_degree=self.drift_degree,
                    drift_alpha=self.drift_alpha,
                    diff_threshold=self.diff_threshold,
                    diff_degree=self.diff_degree,
                    diff_alpha=self.diff_alpha
                )
            self._avgdriftX_ = self._avgdriftX_ / self.n_trials
            self._avgdriftY_ = self._avgdriftY_ / self.n_trials
            self._avgdiffX_ = self._avgdiffX_ / self.n_trials
            self._avgdiffY_ = self._avgdiffY_ / self.n_trials
            self._avgdiffXY_ = self._avgdiffXY_ / self.n_trials
            self._drift_slider[self.Dt] = [self._avgdriftX_, self._avgdriftY_, self._op_x_, self._op_y_]
            self._diff_slider[self.dt] = [self._avgdiffX_, self._avgdiffY_, self._op_x_, self._op_y_]
            self._cross_diff_slider[self.dt] = [self._avgdiffXY_, self._avgdiffYX_, self._op_x_, self._op_y_]
            #else:
            #    # FIXME self._driftX_, etc. need to be set here.
            #    self._drift_slider, self._diff_slider, self._cross_diff_slider = self._slider_data(self._Mx, self._My)
            #    self._avgdriftX_, self._avgdriftY_, self._op_x_, self._op_y_ = self._drift_slider[self.Dt]
            #    self._avgdiffX_, self._avgdiffY_ = self._diff_slider[self.dt][:2]
            #    self._avgdiffXY_, self._avgdiffYX_ = self._cross_diff_slider[self.dt][:2]

        # inc = self.inc_x if self.vector else self.inc
        # self.gaussian_noise, self._noise, self._kl_dist, self.k, self.l_lim, self.h_lim, self._noise_correlation = self._noise_analysis(
        #	self._X, self.Dt, self.dt, self.t_int, inc=inc, point=0)
        # X, Dt, dt, t_int, inc=0.01, point=0,
        return Daddy(self)


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
    output : pydaddy.daddy.Daddy
        object to access the analysed data, parameters, plots and save them.
    """

    def __new__(
            cls,
            data,
            t=1.0,
            Dt=1,
            dt=1,
            bins=None,
            inc=None,
            inc_x=None,
            inc_y=None,
            #slider_timescales=None,
            n_trials=1,
            show_summary=True,
            #drift_threshold=None,
            #diff_threshold=None,
            #drift_degree=5,
            #diff_degree=5,
            #drift_alpha=0,
            #diff_alpha=0,
            #fit_functions=False,
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
            slider_timescales=None,
            n_trials=n_trials,
            show_summary=show_summary,
            drift_threshold=None,
            diff_threshold=None,
            drift_degree=None,
            diff_degree=None,
            drift_alpha=None,
            diff_alpha=None,
            #fast_mode=not fit_functions,
            fast_mode=True,
            **kwargs)

        return ddsde(data=data, t=t, Dt=Dt)


def load_sample_data(data_path):
    r"""
    Load the sample distrubuted data

    data
    ├── fish_data
    │   └── ectropus.csv
    └── model_data
        ├── scalar
        │   ├── pairwise.csv
        │   └── ternary.csv
        └── vector
            ├── pairwise.csv
            └── ternary.csv


    Each data file in pairwise, ternary and extras have two columns;
    first column is the timeseries data x, and the second one is the time stamp

    vector_data.csv also has two columns but contains the vector data x1 and x2 with missing time stamp. Use t_int=0.12.
    """
    stream = pkg_resources.resource_stream('pydaddy', data_path)
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
        'fish-data-etroplus': 'data/fish_data/ectropus.csv',
        'model-data-scalar-pairwise': 'data/model_data/scalar/pairwise.csv',
        'model-data-scalar-ternary': 'data/model_data/scalar/ternary.csv',
        'model-data-vector-pairwise': 'data/model_data/vector/pairwise.csv',
        'model-data-vector-ternary': 'data/model_data/vector/ternary.csv'
    }
    if name not in data_dict.keys():
        print('Invalid data set name\nAvaiable data set\n{}'.format(list(data_dict.keys())))
        raise InputError('', 'Invalid data set name')

    if 'scalar' in name:
        M, t = load_sample_data(data_dict[name]).T
        return [M], t
    Mx, My = load_sample_data(data_dict[name]).T
    return [Mx, My], 0.12
