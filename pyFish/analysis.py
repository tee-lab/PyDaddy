import numpy as np
import scipy.optimize
import scipy.stats
import statsmodels.api as sm 
import statsmodels.stats.diagnostic
from statsmodels.stats import weightstats as stests
from tqdm import tqdm
from pyFish.sde import SDE
from pyFish.metrics import metrics

class AutoCorrelation:
	"""
	This class defines methods to calculate the _autocorrelation function of time series,
	fit an exponential curve to it and calculate the _autocorrealtion time. 
	"""
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

	def _acf_nfft(self, data, t_lag):
		t = np.arange(len(data))
		x = np.linspace(-0.5, 0.5, len(data))
		x = np.delete(x, np.isnan(data))
		t = np.delete(t, np.isnan(data))
		data = np.delete(data, np.isnan(data))
		if len(data) % 2 == 1:
			data = data[:-1]
			x = x[:-1]
		c = np.fft.ifft(np.square(np.abs(self._nfft(x,data))))
		c /= max(c)
		return t[0:t_lag],c[0:t_lag]

	def _acf_fft(self, data, t_lag):
		if np.isnan(data).any():
			return self._acf_nfft(data, t_lag)
		x = np.arange(0, t_lag)
		c = np.fft.ifft(np.square(np.abs(np.fft.fft(data))))
		c /= max(c)
		return x,c[0:t_lag]

	def _acf(self, data, t_lag):
		if self.fft: 
			return self._acf_fft(data, t_lag)
		if np.isnan(data).any():
			return self._acf_nfft(data, t_lag)
		x = np.arange(0, self.t_lag+1)
		c = [np.corrcoef(data[:-i],data[i:])[0][1] for i in x[1:]]
		c.insert(0,1)
		return x, np.array(c)

	"""
	def _autocorr(self, data, t_lag):
		""
		Calculate the auto correlation  function

		input params:
		data 	: time series
		t_lag 	: max lag to calculate acf

		returns:
		x : array of lags
		c : array of auto correlation factors 
		""
		x, c = self._acf(data, t_lag)
		self._autocorr_x, self._autocorr_y = x, c
		return x, c
	"""

	def _fit_exp(self, x, y):
		"""
		Fits an exponential function of the form a*exp((-1/b)*t)

		input parms:
		x : x-data
		y : y-data

		returns:
		coeff1 : [a,b]
		coeff2 :
		"""
		fn = lambda t,a,b,c: a*np.exp((-1/b)*t) + c
		coeff1, coeff2 = scipy.optimize.curve_fit(fn, x, y)
		return coeff1, coeff2

	def _get_autocorr_time(self, X, t_lag=1000):
		"""
		Calculate _autocorrelation time
		
		input parms:
		X 			: time series
		t_lag=1000 	: max lag

		returns:
		b : lag corresponding to _autocorrelation time
		"""
		t_lag, c = self._acf(X, t_lag)
		self._autocorr_x, self._autocorr_y = t_lag, c
		#t_lag, c = self._autocorr(X, t_lag)
		coeff1, coeff2 = self._fit_exp(t_lag, c)
		a,b,c = coeff1
		self._a, self.autocorrelation_time, self._c = a, b, c
		return int(np.ceil(b))

	def _phi(self, x, n, m, sigma):
		b = (2 * sigma * m) / ((2 * sigma - 1) * np.pi)
		return np.exp(-(n * x) ** 2 / b) / np.sqrt(np.pi * b)

	def _phi_hat(self, k, n, m, sigma):
		b = (2 * sigma * m) / ((2 * sigma - 1) * np.pi)
		return np.exp(-b * (np.pi * k / n) ** 2)

	def _C_phi(self, m, sigma):
		return 4 * np.exp(-m * np.pi * (1 - 1. / (2 * sigma - 1)))

	def _m_from_C_phi(self, C, sigma):
		return np.ceil(-np.log(0.25 * C) / (np.pi * (1 - 1 / (2 * sigma - 1))))

	def _nfft(self, x, f, sigma=2, tol=1E-8):
		"""Alg 3 from https://www-user.tu-chemnitz.de/~potts/paper/nfft3.pdf"""
		N = len(f)
		n = N * sigma  # size of oversampled grid
		m = self._m_from_C_phi(tol / N, sigma)
		
		# 1. Express f(x) in terms of basis functions phi
		shift_to_range = lambda x: -0.5 + (x + 0.5) % 1
		col_ind = np.floor(n * x[:, np.newaxis]).astype(int) + np.arange(-m, m)
		vals = self._phi(shift_to_range(x[:, None] - col_ind / n), n, m, sigma)
		col_ind = (col_ind + n // 2) % n
		indptr = np.arange(len(x) + 1) * col_ind.shape[1]
		mat = scipy.sparse.csr_matrix((vals.ravel(), col_ind.ravel(), indptr), shape=(len(x), n))
		g = mat.T.dot(f)
		
		# 2. Compute the Fourier transform of g on the oversampled grid
		k = -(N // 2) + np.arange(N)
		g_k_n = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(g)))
		g_k = n * g_k_n[(n - N) // 2: (n + N) // 2]
		
		# 3. Divide by the Fourier transform of the convolution kernel
		f_k = g_k / self._phi_hat(k, n, m, sigma)
		
		return np.fft.fftshift(f_k)

class underlying_noise(SDE):
	"""
	Calculates noise in time series

	input parms:
	X 			: time series
	inc 		: max binning increments
	point = 0 	: measurement in time sereis when moise in maxmium
	dt 			: analysis time step
	t_int 		:

	returns:
	noise : noise in time series 
	"""
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
		SDE.__init__(self)

	def _noise(self, X, dt, t_int, inc=0.01, point=0):
		op = np.arange(-1,1,inc).round(4)
		avgDrift = []
		x = X[0:-dt]
		drift = self._drift(X,t_int,dt)
		for b in np.arange(point, point+inc, inc):
			i = np.where(np.logical_and(x<(b+inc), x>=b))[0]
			avgDrift.append(drift[i].mean())
		avgDrift = np.array(avgDrift)
		j = np.where(op==point)[0]
		_avgDrift = 0 if j>len(avgDrift) else avgDrift[j]
		try:
			noise = ((x[i+1] - x[i]) - (t_int*dt)*_avgDrift)/np.sqrt(t_int)
		except IndexError:
			noise = ((x[i[:-1]+1] - x[i[:-1]]) - (t_int*dt)*_avgDrift)/np.sqrt(t_int)
		return noise[~np.isnan(noise)]

class gaussian_test(underlying_noise, metrics, AutoCorrelation):
	"""
	This class is used to chack if the noise is gaussian in nature
	it uses three well known tests:
	Shapiro Wiki test
	Skewness and Kurtosis test (normaltest)
	Anderson-Darling test

	input parms:
	noise 			: noise data
	sh_alpha = 0.05 : threshold for shapiro test
	K2_alpha = 0.05 : threshold for normaltest
	pass_difficulty = 1 : minimum of the 3 tests to pass to be accepted as gaussian

	returns:
	<bool> : True or False
	"""
	def __init__(self, **kwargs):
		underlying_noise.__init__(self)
		metrics.__init__(self)
		AutoCorrelation.__init__(self)
		self.__dict__.update(kwargs)

	def _get_critical_values(self, kl_dist):
		hist, self._X1 = np.histogram(kl_dist, normed=True)
		dx = self._X1[1] - self._X1[0]
		self._f = np.cumsum(hist)*dx
		l_lim = self._X1[1:][np.where(self._f <= 0.05)][-1]
		h_lim = self._X1[1:][np.where(self._f >= 0.95)][0]
		return l_lim, h_lim

	def _noise_analysis(self,  X, dt, t_int, inc=0.01, point=0, **kwargs):
		self.__dict__.update(kwargs)
		noise = self._noise(X, dt, t_int, inc, point)
		s = noise.size
		kl_dist = []
		for _ in tqdm(range(10000), desc='Gaussian check for underlying noise'):
			p = np.random.normal(size = s)
			q = np.random.normal(size = s)
			kl_dist.append(self._kl_divergence(p,q))
		l_lim, h_lim = self._get_critical_values(kl_dist)
		k = self._kl_divergence(noise, np.random.normal(size=s))
		gaussian_noise = True if k >= l_lim and k <= h_lim else False
		noise_correlation = self._acf(noise, t_lag=10)
		return gaussian_noise, noise, kl_dist, k, l_lim, h_lim, noise_correlation
