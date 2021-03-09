import numpy as np
import scipy.optimize
import scipy.stats
import statsmodels.api as sm
import statsmodels.stats.diagnostic
from statsmodels.stats import weightstats as stests
from tqdm import tqdm
from pyddsde.sde import SDE
from pyddsde.metrics import metrics


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

	def _acf_fft(self, data, t_lag):
		"""
		Calculates autocorrelation using wiener khinchin theorem.
		"""
		if np.isnan(data).any():
			print('Missing values in time series')
			self.fft = False
			return self._nan_acf(data, t_lag)
		x = np.arange(0, t_lag)
		c = np.fft.ifft(np.square(np.abs(np.fft.fft(data))))
		c /= max(c)
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
		coeff1 : array
			Optimal values for the parameters so that the sum of the squared residuals of
		coeff2 : 2-D array
			The estimated covariance of popt. The diagonals provide the variance of the parameter estimate. 
			To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).
			How the sigma parameter affects the estimated covariance depends on absolute_sigma argument, as described above.
			If the Jacobian matrix at the solution doesn’t have a full rank, then ‘lm’ method returns a matrix filled with np.inf, 
			on the other hand ‘trf’ and ‘dogbox’ methods use Moore-Penrose pseudoinverse to compute the covariance matrix.

		Notes
		-----
		Reference : scipy.optimize.curve_fit
		"""
		fn = lambda t, a, b, c: a * np.exp((-1 / b) * t) + c
		coeff1, coeff2 = scipy.optimize.curve_fit(fn, x, y)
		return coeff1, coeff2

	def _get_autocorr_time(self, X, t_lag=1000):
		"""
		Get the autocorrelation time of data `X`
		"""
		t_lag, c = self._acf(X, t_lag)
		self._autocorr_x, self._autocorr_y = t_lag, c
		#t_lag, c = self._autocorr(X, t_lag)
		coeff1, coeff2 = self._fit_exp(t_lag, c)
		a, b, c = coeff1
		self._a, self.autocorrelation_time, self._c = a, b, c
		return int(np.ceil(b))


class underlying_noise(SDE):
	"""
	Extract noise from time series

    :meta private:
	"""
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
		SDE.__init__(self)

	def _noise(self, X, dt, delta_t, t_int, inc, point=0):
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
		dt : int
			drift time scale
		t_int : int
			time difference between consecutive observations

		Returns
		-------
		array
			 noise extracted from data at given point 
		"""

		"""
		#op = np.arange(min(X), max(X), inc).round(4)
		op, _ = self._order_parameter(X, inc, None)
		avgDrift = []
		drift = self._drift(X, t_int, dt)
		x = X[0:-dt]
		#for b in np.arange(point, point + inc, inc):
		i = np.where(np.logical_and(x <= (point + inc), x >= point))[0]
		avgDrift.append(drift[i].mean())
		avgDrift = np.array(avgDrift)
		#j = np.where(op == point)[0]
		#print(j)
		#_avgDrift = 0 if j >= len(avgDrift) else avgDrift[j]
		#print(len(_avgDrift))
		#print(_avgDrift)
		x = X
		try:
			#noise = ((x[delta_t:] - x[:-delta_t]) - avgDrift * (t_int * delta_t)) / np.sqrt(t_int*delta_t)
			#noise = ((x[delta_t:] - x[:-delta_t]) - drift[i][:-1] * (t_int * delta_t)) / np.sqrt(t_int*delta_t)
			noise = ((x[i+delta_t] - x[i]) - drift[i]*(t_int*delta_t)) / np.sqrt(t_int*delta_t)
		except IndexError:
			print("Exception")
			noise = ((x[i[:-1] + delta_t] - x[i[:-1]]) - avgDrift * (t_int * delta_t) ) / np.sqrt(t_int*delta_t)
		return noise[~np.isnan(noise)]
		"""
		x = X[:len(X)-max(dt,delta_t)]
		i = np.where(np.logical_and(x <= (point + inc), x >= point))[0]
		noise = self._residual(X, t_int=t_int, dt=dt, delta_t=delta_t) / np.sqrt(t_int * delta_t)
		return noise[i]






class gaussian_test(underlying_noise, metrics, AutoCorrelation):
	"""
	Used to check if the noise is gaussian in nature

    :meta private:
	"""
	def __init__(self, **kwargs):
		underlying_noise.__init__(self)
		metrics.__init__(self)
		AutoCorrelation.__init__(self)
		self.__dict__.update(kwargs)

	def _get_critical_values(self, kl_dist):
		"""
		Get critical values of null hypothesis, 
		i.e values at the boundries of 5% and 95% of null hypothesis
		"""
		hist, self._X1 = np.histogram(kl_dist, normed=True)
		dx = self._X1[1] - self._X1[0]
		self._f = np.cumsum(hist) * dx
		l_lim = self._X1[1:][np.where(self._f <= 0.05)][-1]
		h_lim = self._X1[1:][np.where(self._f >= 0.95)][0]
		return l_lim, h_lim

	def _noise_analysis(self, X, dt, delta_t, t_int, inc, point=0, **kwargs):
		"""
		Check if noise is gaussian

		Parameters
		----------
		X : array
			timeseries data
		dt : int
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
		noise = self._noise(X, dt, delta_t, t_int, inc, point)
		s = noise.size
		if s == 0:
			print('Warning : Length of noise is 0')
		kl_dist = []
		#for _ in tqdm(range(10000), desc='Gaussian check for underlying noise'):
		for _ in range(10000):
			p = np.random.normal(size=s)
			q = np.random.normal(size=s)
			kl_dist.append(self._kl_divergence(p, q))
		l_lim, h_lim = self._get_critical_values(kl_dist)
		k = self._kl_divergence(noise, np.random.normal(size=s))
		gaussian_noise = True if k >= l_lim and k <= h_lim else False
		if s:
			t_lag = s-1 if s <= 10 else 10
		else:
			t_lag = 0
		try:
			noise_correlation = self._acf(noise, t_lag=10)
		except ValueError as e:
			print('Warning : ValueError ', e, 'While finding noise correlation\n')
			noise_correlation = np.arange(0, t_lag), np.arange(0,t_lag)*np.nan
		return gaussian_noise, noise, kl_dist, k, l_lim, h_lim, noise_correlation
