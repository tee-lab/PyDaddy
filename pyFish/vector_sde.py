import numpy as np 

class SDE():
	"""Calculates Drift and Diffusion derrived from 
	general form of Stochastic Differential Eqution

	input params:
	X 			: time series data <list, numpy.ndarray>
	t_int 		: time step in time series <float>
	dt 			: analysis time step <float>
	inc = 0.01 	: max increment for binning thae data <float

	returns:
	diff 	: diffusion in time series <numpy.ndarray>
	drift 	: drift in time series <numpy.ndarray>
	avgdiff : avarage diffusion <numpy.ndarray>
	avgdrift: avaerage drift <numpy.ndarray>
	"""

	def __init__(self, **kwargs):
		#self.avgdiff, self.avgdrift = [], []
		self.__dict__.update(kwargs)

	def drift(self, X, t_int, dt):
		"""
		Calculate Diffusion.

		input params:
		X 		: Time Series data 					<list, numpy.ndarray>
		t_int 	: time step in time series data 	<float>
		dt 		: analysis time step			 	<float>

		returns:
		diff 	: Diffusion in time series <numpy.ndarray>
		"""
		return np.array([b-a for a,b in zip(X, X[dt:])])/(t_int*dt)

	def diffusion(self, X, t_int, delta_t=1):
		"""
		Calculates Drift

		input params:
		X 		: time series data 				<list, numpy.ndarray>
		t_int 	: time step in time series 		<float>
		dt 		: analysis time step 			<float>

		retruns:
		drift 	: Drift in time series <numpy.ndarray>
		"""
		return np.square(np.array([b-a for a, b in zip(X, X[delta_t:])]))/(t_int*delta_t)

	def drift_and_diffusion(self, X, t_int, dt,delta_t=1, inc = 0.01):
		"""
		Calcualtes drift, diffusion, average drift and avarage difussion.

		input params:
		X 			: time series data 						<list, numpy.ndarray>
		t_int 		: time step in time series 				<float>
		dt 			: analysis time step 					<float>
		inc = 0.01 	: max increment for binning thae data 	<float

		returns:
		diff 	: diffusion in time series 		<numpy.ndarray>
		drift 	: drift in time series 			<numpy.ndarray>
		avgdiff : avarage diffusion 			<numpy.ndarray>
		avgdrift: avaerage drift 				<numpy.ndarray>
		"""
		op = np.arange(-1,1,inc)
		#self.avgdiff, self.avgdrift = [], []
		avgdiff, avgdrift = [], []
		drift = self.drift(X, t_int, dt)
		diff = self.diffusion(X, t_int)
		X = X[0:-dt]
		for b in op:
			i = np.where(np.logical_and(X<(b+inc), X>=b))[0]
			avgdiff.append(diff[i].mean())
			avgdrift.append(drift[i].mean())
		return diff, drift, np.array(avgdiff), np.array(avgdrift), op


	def __call__(self, X, t_int, dt, delta_t=1, inc=0.01, **kwargs):
		"""
		Calcualtes drift, diffusion, average drift and avarage difussion.

		input params:
		X 			: time series data 						<list, numpy.ndarray>
		t_int 		: time step in time series 				<float>
		dt 			: analysis time step 					<float>
		inc = 0.01 	: max increment for binning thae data 	<float

		returns:
		diff 	: diffusion in time series 		<numpy.ndarray>
		drift 	: drift in time series 			<numpy.ndarray>
		avgdiff : avarage diffusion 			<numpy.ndarray>
		avgdrift: avaerage drift 				<numpy.ndarray>
		"""
		self.__dict__.update(kwargs)
		return self.drift_and_diffusion(X, t_int, dt, inc)
