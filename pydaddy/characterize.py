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


warnings.filterwarnings("ignore")

__all__ = ['Characterize', 'load_sample_dataset']


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

        self.op_range = None
        self.op_x_range = None
        self.op_y_range = None
        if bins:
            self.bins = bins
        elif not (self.inc or (self.inc_x and self.inc_y)):
            self.bins = 20  # self._autobins()
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

        self.__dict__.update(kwargs)
        Preprocessing.__init__(self)
        GaussianTest.__init__(self)
        AutoCorrelation.__init__(self)

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

                self.F1, self.F2, self.G11, self.G22, self.G12, self.G21 = [None]*6

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

    def __call__(self, data, t=1, Dt=None, **kwargs):
        self.__dict__.update(kwargs)

        self._t = t
        self._preprocess()
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
                self.F1, self.F2, self.G11, self.G22, self.G12, self.G21 = self._vector_drift_diff(
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

        return Daddy(self)


class Characterize(object):
    """
    Intialize a PyDaddy object for further analysis.

    Args
    ----
    data : list
        Time series data to be analysed. data = [x] for scalar data and data = [x1, x2] for vector
        where x, x1 and x2 are of numpy.array object type

    t : float, array, optional(default=1.0)
        t can be either a float representing the time-interval between observations, or a numpy array containing the
        time-stamps of the individual observations (Note: PyDaddy only supports uniformly spaced time-series, even when
        time-stamps are provided).

    bins : int, optional(default=20)
        Number of bins for computing bin-wise averages of drift and diffusion (Binwise averages are used
        only for visualization.)
    show_summary : bool, optional(default=True)
        If true, a summary text and summary figure will be shown.

    Dt : int, optional(default=1)
        Subsampling factor for drift computation. When provided, the time-series will be sub-sampled by this factor
        while computing drift.
    dt : int, optional(default=1)
        Subsampling factor for diffusion computation. When provided, the time-series will be sub-sampled by this factor
        while computing diffusion.
    inc : float, optional(default=0.01)
        For scalar data, instead of specifying `bins`, the widths (increments) of the bins can also be provided.
    inc_x : float, optional(default=0.1)
        For vector data, instead of specifying `bins`, the widths (increments) of the bins can also be provided.
        inc_x is the increment in the x-dimension.
    inc_y : float, optional(default=0.1)
        For vector data, instead of specifying `bins`, the widths (increments) of the bins can also be provided.
        inc_y is the increment in the y-dimension.
    n_trials : int, optional(default=1)
        Number of trials, concatenated timeseries of multiple trials is used.

    Returns
    -------
    output : pydaddy.daddy.Daddy
        Daddy object which can be used for further analysis and visualization. See :class:`pyaddy.daddy.Daddy` for
        details.
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
            slider_timescales=None,
            n_trials=n_trials,
            show_summary=show_summary,
            drift_threshold=None,
            diff_threshold=None,
            drift_degree=None,
            diff_degree=None,
            drift_alpha=None,
            diff_alpha=None,
            fast_mode=True,
            **kwargs)

        return ddsde(data=data, t=t, Dt=Dt)


def _load_sample_data(data_path):
    r"""
    Load the sample distrubuted data

    ::

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
        res = np.loadtxt(stream, delimiter=',')
    except:
        res = np.loadtxt(stream)

    stream.close()
    return res


def load_sample_dataset(name):
    r"""
    Load one of the sample datasets. For more details on the datasets, see :ref:`sample datasets`.

    Available data sets:

    ::

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
        M, t = _load_sample_data(data_dict[name]).T
        return [M], t
    Mx, My = _load_sample_data(data_dict[name]).T
    return [Mx, My], 0.12
