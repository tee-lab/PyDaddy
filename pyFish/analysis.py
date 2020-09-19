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
	This class defines methods to calculate the autocorrelation function of time series,
	fit an exponential curve to it and calculate the autocorrelation time. 
	"""
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

	def acf_fft(self, data, t_lag):
		"""
		Autocorrelation function using fft mothod

		Input params:
		--------------
		data : numpy.array
			1 dimentional time series data
		t_lag : int
			maxmium time lag to be considered

		returns:
		-------------
		x : numpy.array
			x = numpy.arrange(0, t_lag+1)
		c : numpy.array
			correlation values
		"""
		x = np.arange(0, t_lag+1)
		try:
			c = np.fft.ifft(np.square(np.abs(np.fft.fft(data))))[0:t_lag+1]
		except ValueError:
			print("Warning: Invalid FFT points {}. returning array of zeros".format(len(x)))
			c = np.ones(t_lag+1)*0
		return x,c

	def acf(self, data, t_lag):
		"""
		Calculates acf using standard method if fft is False, else
		uses fft method to calculate acf

		Input params:
		--------------
		data : numpy.array:
			1 dimentional time series data
		t_lag : int
			maxmium time lag to be considered

		returns:
		--------------
		x : numpy.array:
			x = numpy.arrange(0, t_lag+1)
		c : numpy.array
			correlation values
		"""
		if self.fft: self.acf_fft(data, t_lag)
		x = np.arange(0, t_lag+1)
		c = [np.corrcoef(data[:-i],data[i:])[0][1] for i in x[1:]]
		c.insert(0,1)
		return x, np.array(c)

	def autocorr(self, data, t_lag):
		"""
		Calculate the auto correlation  function

		Input params:
		--------------
		data : numpy.array
			time series
		t_lag : int
			max lag to calculate acf

		returns:
		-------------
		x : numpy.array
			x = numpy.arrange(0, t_lag+1)
		c : numpy.array
			correlation values
		"""
		x, c = self.acf(data, t_lag)
		self._autocorr_x, self._autocorr_y = x, c
		return x, c

	def fit_exp(self, x, y):
		"""
		Fits an exponential function of the form a*exp((-1/b)*t)

		Input params:
		--------------
		x : numpy.array
			x-data
		y : numpy.array
			y-data

		returns:
		-------------
		coeff1 : list
			[a,b]
		coeff2 : list
		"""
		fn = lambda t,a,b: a*np.exp((-1/b)*t)
		coeff1, coeff2 = scipy.optimize.curve_fit(fn, x, y)
		return coeff1, coeff2

	def get_autocorr_time(self, X, t_lag=1000):
		"""
		Calculate autocorrelation time
		
		Input params:
		--------------
		X : numpy.array
			time series
		t_lag=1000 : int
			max lag

		returns:
		--------------
		b : float
			lag corresponding to autocorrelation time
		"""
		t_lag, c = self.autocorr(X, t_lag)
		coeff1, coeff2 = self.fit_exp(t_lag, c)
		a,b = coeff1
		self._a, self.autocorrelation_time = a, b
		return int(np.ceil(b))

class underlying_noise(SDE):
	"""
	Calculates noise in time series

	Input params:
	--------------
	X : numpy.array
		time series
	inc : float
		max binning increments
	point = 0 : float
		measurement in time sereis when moise in maxmium
	dt : float
		analysis time step
	t_int : float
		time step increments

	returns:
	-------------
	noise : numpy.array
		noise in time series 
	"""
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
		SDE.__init__(self)

	def noise(self, X, dt, t_int, inc=0.01, point=0):
		"""
		Calculates noise in time series

		Input params:
		--------------
		X : numpy.array
			time series
		inc : float
			max binning increments
		point = 0 : float
			measurement in time sereis when moise in maxmium
		dt : float
			analysis time step
		t_int : float
			time step increments

		returns:
		-------------
		noise : numpy.array
			noise in time series 
		"""
		op = np.arange(-1,1,inc).round(4)
		avgDrift = []
		x = X[0:-dt]
		drift = self.drift(X,t_int,dt)
		for b in np.arange(point, point+inc, inc):
			i = np.where(np.logical_and(x<(b+inc), x>=b))[0]
			avgDrift.append(drift[i].mean())
		avgDrift = np.array(avgDrift)
		j = np.where(op==point)[0]
		_avgDrift = 0 if j>len(avgDrift) else avgDrift[j]
		noise = ((x[i+1] - x[i]) - (t_int*dt)*_avgDrift)/np.sqrt(t_int)
		return noise 

class gaussian_test(underlying_noise, metrics, AutoCorrelation):
	"""
	This class is used to check if the noise is gaussian in nature
	by genetating a test distrubution does a test of hypothesis 
	of noise on the generated test distrubution.

	Noise is gaussian in nature if kl_divergence between test distrubution
	and noise is within 95% of the test distrubution.
	"""
	def __init__(self, **kwargs):
		underlying_noise.__init__(self)
		metrics.__init__(self)
		AutoCorrelation.__init__(self)
		self.__dict__.update(kwargs)

	def generate_test_distrubution(self, s):
		"""
		Generates the test distrubution

		Input params:
		--------------
		s : int
			size of the noise array

		returns:
		--------------
		kl_dist : list
			test distrubution
		"""
		kl_dist = []
		for _ in tqdm(range(10000), desc = "Gaussian check for underlying noise"):
			p = np.random.normal(size=s)
			q = np.random.normal(size=s)
			kl_dist.append(self.kl_divergence(p,q))
		return kl_dist

	def get_critical_values(self, kl_dist):
		"""
		upper and lower critical values corresponding to 95% of a distrubution

		Input params:
		--------------
		kl_dist : list, numpy.array
			test distrubution

		returns:
		-------------
		l_lim : float
			lower critical limit
		h_lim : float
			upper critial limit
		"""
		hist, self._X1 = np.histogram(kl_dist, normed=True)
		dx = self._X1[1] - self._X1[0]
		self._f = np.cumsum(hist)*dx
		l_lim = self._X1[1:][np.where(self._f <= 0.05)][-1]
		h_lim = self._X1[1:][np.where(self._f >= 0.95)][0]
		return l_lim, h_lim

	def noise_analysis(self,  X, dt, t_int, inc=0.01, point=0, **kwargs):
		"""
		Does the noise analysis i.e, checks if the underlying noise
		is gaussian or not

		Input params:
		--------------
		X : numpy.array
			time series data
		dt : float
			time scale
		t_int : float
			increments in time steps
		inc = 0.01 : float
			increments in order parameter
		point = 0 : float
			point where noise is maxmium

		returns:
		-------------
		gaussian_noise : bool
			True in noise is gaussian, else False
		noise : numpy.array
			noise array
		kl_dist : list
			test distrubution used
		k : float
			test statistics
		l_lim : float
			lower critical limit
		h_kim : float
			upper critical limit
		noise_correlation : list
			[x, c], noise autocorrelation 'c' as function of lag 'x'

		"""
		self.__dict__.update(kwargs)
		noise = self.noise(X, dt, t_int, inc, point)
		"""
		s = noise.size
		kl_dist = []
		for _ in tqdm(range(10000), desc='Gaussian check for underlying noise'):
			p = np.random.normal(size = s)
			q = np.random.normal(size = s)
			kl_dist.append(self.kl_divergence(p,q))
		"""
		kl_dist = self.generate_test_distrubution(s = noise.size)
		l_lim, h_lim = self.get_critical_values(kl_dist)
		k = self.kl_divergence(noise, np.random.normal(size=noise.size))
		gaussian_noise = True if k >= l_lim and k <= h_lim else False
		noise_correlation = self.acf(noise, t_lag=10)
		return gaussian_noise, noise, kl_dist, k, l_lim, h_lim, noise_correlation
