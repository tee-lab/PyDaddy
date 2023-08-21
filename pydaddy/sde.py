import warnings
from collections import namedtuple
import numpy as np
from .fitters import PolyFit1D, PolyFit2D
from scipy.special import factorial

class SDE:
    """
    A class to form a basic SDE from data

    Args
    ----
    X : array_like
            time series data
    t_int : float
            time step in time series
    Dt : int
            analysis time step
    inc : float
            max increment for binning thae data

    Returns
    -------
    diff : array_like
            diffusion in time series
    drift : array_like
            drift in time series
    avgdiff : array_like
            avarage diffusion
    avgdrift : array_like
            average drift

    :meta private:
    """

    def __init__(self, **kwargs):
        """
        pass

        .. document private functions
        """
        self.__dict__.update(kwargs)

    def _drift(self, X, t_int, Dt):
        r"""
        Get Drift coeffecient vector of data.

        Args
        ----
        X : array_like
                Time Series data
        t_int : float
                time difference betwen consecutive observations
        Dt : float
                drift calculation timescale

        Returns
        -------
        diff : array_like
            Diffusion in time series

        Notes
        -----
        Drift is calculated as follows

        .. math::
            drift = \frac{x(i+Dt)-x(i)}{tint * Dt}
        """

        # return np.array([b - a for a, b in zip(X, X[Dt:])]) / (t_int * Dt)
        return (X[Dt:] - X[:-Dt]) / (t_int * Dt)

    def _residual(self, X, t_int, Dt, dt=1):
        """
        Get the residual.
        """
        p = len(X) - max(dt, Dt)
        drift = self._drift(X, t_int, Dt)[:p]
        res = (X[dt:] - X[:-dt])[:p] - drift*(t_int*dt)
        return res

    def _diffusion_from_residual(self, X, F, t_int, dt=1):
        """
        Get diffusion using residuals about drift function.

        Parameters
        ----------
        X (np.array): Time-series
        t_int (float): Time-step
        F (Callable): Drift function
        """
        drift = F(X[:-dt])
        finite_diff = X[dt:] - X[:-dt]
        residual = finite_diff - drift * t_int
        return residual ** 2 / t_int

    def _diffusion(self, X, t_int, dt=1):
        """
        Get Diffusion coefficient vector of data

        Parameters
        ----------
        X : array_like
            time series data
        t_int : float
            time step in time series
        dt : int
            diffusion calculation timescale

        Returns
        --------
        diff : array.
            Diffusion
        """

        # return np.square(np.array([b - a for a, b in zip(X, X[dt:])])) / (t_int * dt)
        return np.square(X[dt:] - X[:-dt]) / (t_int * dt)

    def _diffusion_x_from_residual(self, x, y, F1, t_int, dt):
        drift = F1(x[:-dt], y[:-dt])
        finite_diff = x[dt:] - x[:-dt]
        residual = finite_diff - drift * t_int
        return residual ** 2 / t_int

    def _diffusion_y_from_residual(self, x, y, F2, t_int, dt):
        drift = F2(x[:-dt], y[:-dt])
        finite_diff = y[dt:] - y[:-dt]
        residual = finite_diff - drift * t_int
        return residual ** 2 / t_int

    def _diffusion_xy_from_residual(self, x, y, F1, F2, t_int, dt):
        drift_x = F1(x[:-dt], y[:-dt])
        drift_y = F2(x[:-dt], y[:-dt])
        residual_x = (x[dt:] - x[:-dt]) - drift_x
        residual_y = (y[dt:] - y[:-dt]) - drift_y
        return residual_x * residual_y / dt * t_int

    def _diffusion_xy(self, x, y, t_int, dt):
        """
        Get cross-correlation coefficients between x and y arrays.

        Args
        ----
        x : numpy.array
                x data
        y : numpy.array
                y data
        t_int : float
                time difference betwen consecutive observations
        dt : diffusion calculation timescale

        Returns
        -------
        diffusion_xy : numpy.array
                cross-correlation coefficients between x and y data
        """
        return ((x[dt:] - x[:-dt]) * (y[dt:] - y[:-dt])) / (dt * t_int)
        #return np.array([(b - a) * (d - c) for a, b, c, d in zip(x, x[dt:], y, y[dt:])]) / (dt * t_int)

    def _diffusion_yx(self, x, y, t_int, dt):
        """
        Get cross-correlation coefficients between x and y arrays.

        Args
        ----
        x : numpy.array
                x data
        y : numpy.array
                y data
        t_int : float
                time difference betwen consecutive observations
        dt : diffusion calculation timescale

        Returns
        -------
        diffusion_xy : numpy.array
                cross-correlation coefficients between y and x data
        """
        return ((x[dt:] - x[:-dt]) * (y[dt:] - y[:-dt])) / (dt * t_int)

    def _km_coefficient(self, order, X, t_int):
        return (X[1:] - X[:-1]) ** order #/ (t_int) # * factorial(order))

    def _isValidRange(self, r):
        """
        Checks if the specified range of order parameter is valid range

        Args
        ----
        r : tuple, list
                range of order parameter

        Returns
        -------
        bool
                True if valid, False if not.
        """
        return isinstance(r, (list, tuple)) and len(r) == 2

    def _order_parameter(self, X, inc, r):
        """
        Get order parameter array for a given range and increments

        If range is None or not valid, order parameter array will be generated considering
        maxmium and mimimum limits of the data as the range

        Args
        ----
        X : numpy.array
                data
        inc : float
                step increments in order parameter
        r : tuple, list
                range of the order parameter

        Returns
        -------
        tuple
                first element will be the order parameter array
                second element is the rangen used
        """
        if r is None:
            r = (min(X), max(X))
        if not self._isValidRange(r):
            r = (min(X), max(X))
        if self.bins:
            #return np.linspace(min(X), max(X), self.bins)
            return np.linspace(r[0], r[-1], self.bins)
        return np.arange(min(r), max(r)+inc, inc)

    def _drift_and_diffusion(self, X, t_int, Dt, dt, inc,
                             drift_threshold, drift_degree, drift_alpha,
                             diff_threshold, diff_degree, diff_alpha,
                             fast_mode):
        """
        Get drift and diffusion coefficients for a given timeseries data

        Args
        ----
        X : numpy.array
                time series data
        t_int : float
                time difference betwen consecutive observations
        Dt : int
                timescale to calculate drift
        dt : int
                timescale to claculate diffusion
        inc : float
                step increments in order parameter
        drift_threshold : float or None
                threshold to use for fitting drift function. If None, automatic model selection will be used.
        diff_threshold : float or None
                threshold to use for fitting diffusion function. If None, automatic model selection will be used.
        Returns
        ----
        ---
        diff : array
            diffusion of the data
        drift : array.
            drift, of the data
        avgdiff : array
            average diffusion
        avgdrift : array
            average drift 
        op : array
            order parameter
        """
        op = self._order_parameter(X, inc, self.op_range)
        avgdiff, avgdrift = [], []
        drift = self._drift(X, t_int, Dt)

        if not fast_mode:
            X_ = X[:-Dt]
            nan_idx = np.isnan(X_) | np.isnan(drift)
            X_ = X_[~nan_idx]
            drift_ = drift[~nan_idx]

            fitter = PolyFit1D(max_degree=drift_degree, threshold=drift_threshold, alpha=drift_alpha)
            if drift_threshold is None:
                F = fitter.tune_and_fit(X_, drift_)
            else:
                F = fitter.fit(X[:-Dt], drift_)

            diff = self._diffusion_from_residual(X, F, t_int, dt=dt)
            diff_ = diff[~nan_idx]
            fitter = PolyFit1D(max_degree=diff_degree, threshold=diff_threshold, alpha=diff_alpha)
            if diff_threshold is None:
                G = fitter.tune_and_fit(X_, diff_)
            else:
                G = fitter.fit(X_, diff_)
        else:
            # r = self._residual_timeseries(X=X, dt=dt, bins=op, avg_drift, avg_diff, t_int)
            # diff = self._diffusion(X, t_int, dt=dt)
            F = G = None

        drift_ebar = []
        diff_ebar = []
        drift_num = []
        diff_num = []
        X_ = X[0 : -max(Dt, dt)]
        for b in op:
            i = np.where(np.logical_and(X_ < (b + inc), X_ >= b))[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # avgdiff.append(np.nanmean(diff[i]))
                avgdrift.append(np.nanmean(drift[i]))
                drift_ebar.append(np.nanstd(drift[i])/np.sqrt(len(drift[i])))
                # diff_ebar.append(np.nanstd(diff[i])/np.sqrt(len(diff[i])))
                drift_num.append(len(drift[i]))
                # diff_num.append(len(diff[i]))

        res = X.copy() #(X[Dt:] - X[:-Dt])
        for i, x in enumerate(X):
            # Find bin-index corresponding to x: minimum i such that x < bins[i], assuming bins is sorted
            try:
                bin = np.argwhere(x < op)[0][0]
            except IndexError:
                bin = len(op) - 1
            res[i] -= avgdrift[bin] * t_int

        diff = self._diffusion(res, t_int, dt=dt)

        for b in op:
            i = np.where(np.logical_and(X_ < (b + inc), X_ >= b))[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avgdiff.append(np.nanmean(diff[i]))
                diff_ebar.append(np.nanstd(diff[i])/np.sqrt(len(diff[i])))
                diff_num.append(len(diff[i]))


        # return diff, drift, np.array(avgdiff), np.array(avgdrift), op, drift_ebar, diff_ebar, drift_num, diff_num, F, G
        DD = namedtuple('DD', 'diff drift avgdiff avgdrift op drift_ebar diff_ebar drift_num diff_num F G')
        return DD(
            diff=diff, drift=drift,
            avgdiff=np.array(avgdiff), avgdrift=np.array(avgdrift), op=op,
            drift_ebar=drift_ebar, diff_ebar=diff_ebar,
            drift_num=drift_num, diff_num=diff_num,
            F=F, G=G
        )

    def _vector_drift_diff(self, x, y, inc_x, inc_y, t_int, Dt, dt,
                           drift_threshold, drift_degree, drift_alpha,
                           diff_threshold, diff_degree, diff_alpha,
                           fast_mode):
        """
        Get average binned drift and diffusion coefficients for given x and y data

        Args
        ----
        x : array_like
            timeseries x data
        y : array_like
            timesereis y data
        inc_x : float
            step increment of order parameter for x
        inc_y : float
            step increment of order parameter for y
        Dt : int
            timescale to calculate drift
        dt : int
            timescale to calculate diffusion

        Returns
        -------
        list
            [avgdriftX, avgdriftY, avgdiffX, avgdiffY, avgdiffXY, op_x, op_y]
        """

        # FIXME diffusionYX = diffusionXY, so all diffusionYX variables can be removed.
        op_x = self._order_parameter(x, inc_x, self.op_x_range)
        op_y = self._order_parameter(y, inc_y, self.op_y_range)

        driftX = self._drift(x, t_int, Dt)
        driftY = self._drift(y, t_int, Dt)

        v = np.stack((x[:-Dt], y[:-Dt]), axis=1)

        if not fast_mode:
            nan_idx = np.isnan(v).any(axis=1) | np.isnan(driftX) | np.isnan(driftY)
            v = v[~nan_idx]
            driftX_ = driftX[~nan_idx]
            driftY_ = driftY[~nan_idx]

            fitter = PolyFit2D(max_degree=drift_degree, threshold=drift_threshold, alpha=drift_alpha)
            if drift_threshold is None:
                F1 = fitter.tune_and_fit(v, driftX_)
                F2 = fitter.tune_and_fit(v, driftY_)
            else:
                F1 = fitter.fit(v, driftX_)
                F2 = fitter.fit(v, driftY_)

            diffusionX = self._diffusion_x_from_residual(x, y, F1, t_int, dt)
            diffusionY = self._diffusion_y_from_residual(x, y, F1, t_int, dt)
            diffusionXY = self._diffusion_xy_from_residual(x, y, F1, F2, t_int, dt)
            diffusionYX = diffusionXY  # self._diffusion_xy_from_residual(x, y, F1, F2, t_int, dt)

            diffusionX_ = diffusionX[~nan_idx]
            diffusionY_ = diffusionY[~nan_idx]
            diffusionXY_ = diffusionXY[~nan_idx]
            diffusionYX_ = diffusionXY_
            fitter = PolyFit2D(max_degree=diff_degree, threshold=diff_threshold, alpha=diff_alpha)
            if diff_threshold is None:
                G11 = fitter.tune_and_fit(v, diffusionX_)
                G22 = fitter.tune_and_fit(v, diffusionY_)
                G12 = fitter.tune_and_fit(v, diffusionXY_)
                G21 = G12
            else:
                G11 = fitter.fit(v, diffusionX_)
                G22 = fitter.fit(v, diffusionY_)
                G12 = fitter.fit(v, diffusionXY_)
                G21 = G12
        else:
            # diffusionX = self._diffusion(x, t_int, dt)
            # diffusionY = self._diffusion(y, t_int, dt)
            # diffusionXY = self._diffusion_xy(x, y, t_int, dt)
            # diffusionYX = self._diffusion_yx(x, y, t_int, dt)

            F1 = F2 = None
            G11 = G22 = G12 = G21 = None

        avgdriftX = np.zeros((len(op_x), len(op_y)))
        avgdriftY = np.zeros((len(op_x), len(op_y)))
        avgdiffX = np.zeros((len(op_x), len(op_y)))
        avgdiffY = np.zeros((len(op_x), len(op_y)))
        avgdiffXY = np.zeros((len(op_x), len(op_y)))
        avgdiffYX = np.zeros((len(op_x), len(op_y)))

        x_, y_ = x[0 : -max(Dt, dt)], y[0 : -max(Dt, dt)]
        for m, bin_x in enumerate(op_x):
            for n, bin_y in enumerate(op_y):
                i = (bin_x <= x_) & (x_ <= bin_x + inc_x) & (bin_y <= y_) & (y_ <= bin_y + inc_y)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    avgdriftX[n, m] = np.nanmean(driftX[i])
                    avgdriftY[n, m] = np.nanmean(driftY[i])
                    # avgdiffX[n, m] = np.nanmean(diffusionX[i])
                    # avgdiffY[n, m] = np.nanmean(diffusionY[i])
                    # avgdiffXY[n, m] = np.nanmean(diffusionXY[i])
                    # avgdiffYX[n, m] = np.nanmean(diffusionYX[i]xy
        res_x = x.copy() #x[Dt:] - x[:-Dt]
        res_y = y.copy() #y[Dt:] - y[:-Dt]

        for i, (x, y) in enumerate(zip(x, y)):
            try:
                bin_x = np.argwhere(x < op_x)[0][0]
            except IndexError:
                bin_x = len(op_x) - 1

            try:
                bin_y = np.argwhere(y < op_y)[0][0]
            except IndexError:
                bin_y = len(op_y) - 1

            res_x[i] -= avgdriftX[bin_x, bin_y] * t_int
            res_y[i] -= avgdriftY[bin_x, bin_y] * t_int

        diffusionX = self._diffusion(res_x, t_int, dt)
        diffusionY = self._diffusion(res_y, t_int, dt)
        diffusionXY = self._diffusion_xy(res_x, res_y, t_int, dt)
        diffusionYX = diffusionXY

        for m, bin_x in enumerate(op_x):
            for n, bin_y in enumerate(op_y):
                i = (bin_x <= x_) & (x_ <= bin_x + inc_x) & (bin_y <= y_) & (y_ <= bin_y + inc_y)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    avgdiffX[n, m] = np.nanmean(diffusionX[i])
                    avgdiffY[n, m] = np.nanmean(diffusionY[i])
                    avgdiffXY[n, m] = np.nanmean(diffusionXY[i])
                    avgdiffYX[n, m] = np.nanmean(diffusionYX[i])

        DD = namedtuple('DD',
                        'driftX driftY diffusionX diffusionY diffusionXY diffusionYX '
                        'avgdriftX avgdriftY avgdiffX avgdiffY avgdiffXY avgdiffYX '
                        'op_x op_y F1 F2 G11 G22 G12 G21')
        return DD(
            driftX=driftX, driftY=driftY, diffusionX=diffusionX, diffusionY=diffusionY,
            diffusionXY=diffusionXY, diffusionYX=diffusionYX,
            avgdriftX=avgdriftX, avgdriftY=avgdriftY,
            avgdiffX=avgdiffX, avgdiffY=avgdiffY, avgdiffXY=avgdiffXY, avgdiffYX=avgdiffYX,
            op_x=op_x, op_y=op_y,
            F1=F1, F2=F2, G11=G11, G22=G22, G12=G12, G21=G21,
        )

        # return [avgdriftX, avgdriftY, avgdiffX, avgdiffY, avgdiffXY, avgdiffYX, op_x, op_y]

    def __call__(self, X, t_int, Dt, dt=1, inc=0.01, **kwargs):
        """
        Calcualtes drift, diffusion, average drift and avarage difussion.

        Parameters
        ----------
        X : list, array_like
                time series data
        t_int :float
                time step in time series
        Dt : float
                analysis time step
        inc = 0.01 : float
                max increment for binning thae data

        returns
        -------
        diff : array_like
                diffusion in time series
        drift : array_like
                drift in time series
        avgdiff : array_like
                avarage diffusion
        avgdrift : array_like
                avaerage drift
        """
        self.__dict__.update(kwargs)
        return self._drift_and_diffusion(X, t_int, Dt, inc)
