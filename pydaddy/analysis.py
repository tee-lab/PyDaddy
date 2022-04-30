import numpy as np
import scipy.optimize
import scipy.stats
from tqdm import tqdm
from pydaddy.sde import SDE
from pydaddy.metrics import Metrics


# FIXME This file can be removed after reorganizing the functions elsewhere.

class AutoCorrelation:
    """
	This class defines methods to calculate the _autocorrelation function of time series,
	fit an exponential curve to it and calculate the _autocorrealtion time.

	Parameters:
	fft : bool
	If True, use fft method (wiener khinchin theorem) to calculate acf.

	:meta private:

	"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def _acf(self, data, t_lag):
        """
		Get auto correaltion function for given `data` and lag `t_lag`

		Parameters
		----------
		data : array
			timeseries data
		t_lag : int
			maxmium lag

		Returns
		-------
		x : array
			lags
		c : array
			correlation values
		
		Notes
		-----
		If fft flag is set True and no valid fft points are found,
		the method uses standard formula method to calculate the autocorrealtion function
		"""
        if self.fft:
            return self._acf_fft(data, t_lag)
        if np.isnan(data).any():
            return self._nan_acf(data, t_lag)
        x = np.arange(0, t_lag)
        c = [np.corrcoef(data[:-i], data[i:])[0][1] for i in x[1:]]
        c.insert(0, 1)
        return x, np.array(c)

    def _ccf(self, x, y, t_lag):
        """" Returns the cross-correlation function between x and y. """

        if np.isnan(x).any() or np.isnan(y).any():
            return self._nan_ccf(x, y, t_lag)

        lags = np.arange(0, t_lag)
        c = [np.corrcoef(x[:-i], y[i:])[0][1] for i in lags[1:]]
        c.insert(0, 1)
        return lags, np.array(c)

    def _acf_fft(self, data, t_lag):
        """
		Calculates autocorrelation using wiener khinchin theorem.
		"""
        if np.isnan(data).any():
            # print('Missing values in time series')
            self.fft = False
            return self._nan_acf(data, t_lag)
        data = data - data.mean()
        x = np.arange(0, t_lag)
        c = np.fft.ifft(np.square(np.abs(np.fft.fft(data))))
        c /= c[0]
        return x, c[0:t_lag]

    def _nan_acf(self, data, t_lag):
        """
		Calculates autocorrealtion using the correaltion formula, ignoring all points
		with nan's
		"""
        c = []
        mue = np.nanmean(data)
        c.append((np.nanmean(
            (data - mue) * (data - mue))) / np.nanvar(data - mue))
        for i in range(1, t_lag):
            c.append((np.nanmean((data[:-i] - mue) * (data[i:] - mue))) /
                     (np.sqrt(np.nanvar(data[:-i]) * np.nanvar((data[i:])))))
        return np.arange(t_lag), np.array(c)

    def _nan_ccf(self, data_x, data_y, t_lag):
        """
		Calculates cross-correlation using the correaltion formula, ignoring all points
		with nan's
		"""

        c = []
        mu_x, mu_y = np.nanmean(data_x), np.nanmean(data_y)
        c.append((np.nanmean(
            (data_x - mu_x) * (data_y - mu_y))) /
                 (np.nanstd(data_x - mu_x) * np.nanstd(data_y - mu_y))
                 )
        for i in range(1, t_lag):
            c.append((np.nanmean((data_x[:-i] - mu_x) * (data_y[i:] - mu_y))) /
                     (np.nanstd(data_x[:-i]) * np.nanstd((data_y[i:]))))
        return np.arange(t_lag), np.array(c)

    def _fit_exp(self, x, y):
        """
		Fits an exponential function of the form a*exp((-1/b)*t) + c

		Parameters
		----------
		x : array
			x data
		y : array
			y data

		Returns
		-------
		params: Tuple (a, b, c) containing the fitted parameters.
		cov: Covariance matrix of errors
		Notes
		-----
		Reference : scipy.optimize.curve_fit
		"""
        fn = lambda t, a, b, c: a * np.exp((-t / b)) + c
        params, cov = scipy.optimize.curve_fit(fn, x, y)
        return params, cov

    def _get_autocorr_time(self, X, t_lag=1000, update=True):
        """
		Get the autocorrelation time of data `X`, for the analysis.
		"""
        t_lag, c = self._acf(X, t_lag)
        self._autocorr_x, self._autocorr_y = t_lag, c
        # t_lag, c = self._autocorr(X, t_lag)
        (a, b, c), _ = self._fit_exp(t_lag, c)
        if update:
            self._a, self.autocorrelation_time, self._c = a, int(np.ceil(b)), c
        return int(np.ceil(b))

    def _act(self, X, t_lag=1000):
        """
		Get autocorrelation time of X.
		"""
        return self._get_autocorr_time(X, t_lag=t_lag, update=False)


class UnderlyingNoise(SDE):
    """
	Extract noise from time series

    :meta private:
	"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        SDE.__init__(self)

    def _noise(self, X, bins, avg_drift, inc, t_int, point=0):
        """
		Get noise from `X` at a paticular point

		Parameters
		----------
		X : array
			time series
		inc : float
			binning increments
		point : float
			point at which noise is to be extracted
		Dt : int
			drift time scale
		t_int : int
			time difference between consecutive observations

		Returns
		-------
		array
			 noise extracted from data at given point 
		"""

        x = X[:-1]
        # i = np.where(np.logical_and(x <= (point + inc), x >= point))[0]
        # noise = self._residual(X, t_int=t_int, Dt=Dt, dt=dt) / np.sqrt(t_int * dt)
        noise = self._residual_timeseries(X, bins, avg_drift, t_int)
        return noise[(point <= x) & (x < point + inc)]

    # return noise

    def _noise_vector(self, X, Y, bins_x, bins_y, avg_drift_x, avg_drift_y, inc_x, inc_y, t_int, point_x=0, point_y=0):
        x, y = X[:-1], Y[:-1]
        noise_x, noise_y = self._residual_timeseries_vector(X=X, Y=Y,
                                                            bins_x=bins_x, bins_y=bins_y,
                                                            avg_drift_x=avg_drift_x, avg_drift_y=avg_drift_y,
                                                            t_int=t_int)
        return noise_x[(point_x <= x) & (x < point_x + inc_x)], noise_y[(point_y <= y) & (y < point_y + inc_y)]

    # cond = (point <= x) & (x < point + inc_x) & (point <= y) & (point <= y + inc_y)
    # return noise[cond]

    def _residual_timeseries(self, X, Dt, bins, avg_drift, t_int):
        res = (X[Dt:] - X[:-Dt])
        for i, x in enumerate(X[:-Dt]):
            # Find bin-index corresponding to x: minimum i such that x < bins[i], assuming bins is sorted
            try:
                bin = np.argwhere(x < bins)[0][0]
            except IndexError:
                bin = len(bins) - 1
            res[i] -= avg_drift[bin] * t_int

        return res / np.sqrt(t_int * Dt)

    def _residual_timeseries_vector(self, X, Y, Dt, bins_x, bins_y, avg_drift_x, avg_drift_y, t_int):
        res_x = X[Dt:] - X[:-Dt]
        res_y = Y[Dt:] - Y[:-Dt]

        for i, (x, y) in enumerate(zip(X[:-Dt], Y[:-Dt])):
            try:
                bin_x = np.argwhere(x < bins_x)[0][0]
            except IndexError:
                bin_x = len(bins_x) - 1

            try:
                bin_y = np.argwhere(y < bins_y)[0][0]
            except IndexError:
                bin_y = len(bins_y) - 1

            res_x[i] -= avg_drift_x[bin_x, bin_y] * t_int
            res_y[i] -= avg_drift_y[bin_x, bin_y] * t_int

        return res_x / np.sqrt(Dt * t_int), res_y / np.sqrt(Dt * t_int)


class GaussianTest(UnderlyingNoise, Metrics, AutoCorrelation):
    """
	Used to check if the noise is gaussian in nature

    :meta private:
	"""

    def __init__(self, **kwargs):
        UnderlyingNoise.__init__(self)
        Metrics.__init__(self)
        AutoCorrelation.__init__(self)
        self.__dict__.update(kwargs)

    def _get_critical_values(self, kl_dist):
        """
		Get critical values of null hypothesis, 
		i.e values at the boundries of 2.5% and 97.5% of null hypothesis
		"""
        hist, self._X1 = np.histogram(kl_dist, normed=True)
        dx = self._X1[1] - self._X1[0]
        self._f = np.cumsum(hist) * dx
        l_lim = self._X1[1:][np.where(self._f <= 0.025)][-1]
        h_lim = self._X1[1:][np.where(self._f >= 0.975)][0]
        return l_lim, h_lim

    def _noise_analysis(self, X, Dt, dt, t_int, inc, point=0, **kwargs):
        """
		Check if noise is gaussian

		Parameters
		----------
		X : array
			timeseries data
		Dt : int
			drift timescale
		inc : float
			increment in order parameter of X
		point : int
			point at which noise is to be extracted
		
		Returns
		-------
		tuple
			- gaussian_noise (bool) : True is the noise is gaussian
			- noise (array) : extracted noise
			- kl_dist (array) : null hypothesis uses
			- k (float) : test statistics used for test of hypothesis
			- l_lim (float) : lower critical limit
			- h_lim (float) : upper critical limit
			- noise_correlation (array) : noise autocorrelation
		"""
        self.__dict__.update(kwargs)
        noise = self._noise(X, Dt, dt, t_int, inc, point)
        # noise[np.isnan(noise)] = 0
        s = noise.size
        if s == 0:
            print('Warning : Length of noise is 0')
        kl_dist = []
        # for _ in tqdm(range(10000), desc='Gaussian check for underlying noise'):
        for _ in tqdm(range(1000)):
            p = np.random.normal(size=s)
            q = np.random.normal(size=s)
            kl_dist.append(self._kl_divergence(p, q))
        l_lim, h_lim = self._get_critical_values(kl_dist)
        k = self._kl_divergence(noise, np.random.normal(size=s))
        gaussian_noise = True if k >= l_lim and k <= h_lim else False
        if s:
            t_lag = s - 1 if s <= 10 else 10
        else:
            t_lag = 0
        try:
            noise_correlation = self._acf(noise, t_lag=10)
        except ValueError as e:
            print('Warning : ValueError ', e, 'While finding noise correlation\n')
            noise_correlation = np.arange(0, t_lag), np.arange(0, t_lag) * np.nan
        return gaussian_noise, noise, kl_dist, k, l_lim, h_lim, noise_correlation
