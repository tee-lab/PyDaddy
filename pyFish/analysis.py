import numpy as np
import scipy.optimize
import scipy.stats
import statsmodels.api as sm 
import statsmodels.stats.diagnostic
from pyFish.sde import SDE
from pyFish.metrics import metrics

class underlying_noise:
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

	def bin(self, x, point, inc):
		op = np.arange(-1,1,inc).round(4)
		for b in np.arange(point, point+inc, inc):
			i = np.where(np.logical_and(x<(b+inc), x>b))[0]
			avgDrift.append(drift[i].mean())
		return np.array(avgDrift), i

	def __call__(self, X, inc, dt, t_int, point=0):
		drift = SDE.drift(X, t_int, dt)
		x = X[0:-dt]
		avgDrift, i = self.bin(x, point, inc)
		j = np.where(op==point)[0]
		avgDrift = 0 if j >len(avgDrift) else avgDrift[j]
		noise = ((x[i+i] - x[i]) - (t_int*dt)*avgDrift)/np.sqrt(t_int)
		return noise

class AutoCorrelation:
	"""
	This class defines methods to calculate the autocorrelation function of time series,
	fit an exponential curve to it and calculate the autocorrealtion time. 
	"""
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

	def autocorr(self, data, t_lag):
		"""
		Calculate the auto correlation  function

		input params:
		data 	: time series
		t_lag 	: max lag to calculate acf

		returns:
		x : array of lags
		c : array of auto correlation factors 
		"""
		x = np.arange(0, t_lag+1)
		c = [np.corrcoef(data[:-i],data[i:])[0][1] for i in x[1:]]
		c.insert(0,1)
		self._autocorr_x, self._autocorr_y = x, np.array(c)
		return x, np.array(c)

	def fit_exp(self, x, y):
		"""
		Fits an exponential function of the form a*exp((-1/b)*t)

		input parms:
		x : x-data
		y : y-data

		returns:
		coeff1 : [a,b]
		coeff2 :
		"""
		fn = lambda t,a,b: a*np.exp((-1/b)*t)
		coeff1, coeff2 = scipy.optimize.curve_fit(fn, x, y)
		return coeff1, coeff2

	def get_autocorr_time(self, X, t_lag=1000):
		"""
		Calculate autocorrelation time
		
		input parms:
		X 			: time series
		t_lag=1000 	: max lag

		returns:
		b : lag corresponding to autocorrelation time
		"""
		t_lag, c = self.autocorr(X, t_lag)
		coeff1, coeff2 = self.fit_exp(t_lag, c)
		a,b = coeff1
		self._a, self.autocorrelation_time = a, b
		return b

class gaussian_test:
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
		self.sh_alpha = 0.05
		self.K2_alpha = 0.05
		self.pass_difficulty = 1
		self.__dict__.update(kwargs)

	def shapiro_wiki(self, noise, **kwargs):
		self.__dict__.update(kwargs)
		stats,  = scipy.stats.shapiro(noise)
		return True if p > self.sh_alpha else False

	def agostinoK2(self, noise, **kwargs):
		self.__dict__.update(kwargs)
		ststs, p = scipy.stats.normaltest(noise)
		return True if p > self.K2_alpha else False

	def andreson(self, noise):
		stats, cv, cl = scipy.stats.anderson(noise)
		return np.array([True if stats < cv[i] else False for i in range(len(cv))])

	def __call__(self, noise, **kwargs):
		self.__dict__.update(kwargs)
		results = []
		results.append(self.shapiro_wiki(noise))
		results.append(self.agostinok2(noise))
		results.append(self.andreson(noise).all())
		return True if np.where(results == True)[0].size >= self.pass_difficulty else False

class analysis(underlying_noise, AutoCorrelation, gaussian_test):
	def __init__(self):
		self.underlying_noise = underlying_noise()
		self.AutoCorrelation = AutoCorrelation()
		self.gaussian_test = gaussian_test()

