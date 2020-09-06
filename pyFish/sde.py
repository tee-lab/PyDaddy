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

	#def drift_xy(self, vel_x, vel_y, t_int, dt):
	#	return np.array([(b-a)*(d-c) for a,b,c,d in zip(vel_x, vel_x[dt:], vel_y, vel_y[dt:])])/(dt*t_int)

	def diffusion_xy(self, vel_x, vel_y, t_int, delta_t):
		return np.array([(b-a)*(d-c) for a,b,c,d in zip(vel_x, vel_x[delta_t:], vel_y, vel_y[delta_t:])])/(delta_t*t_int)

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
		#op = np.arange(-1,1,inc)
		op = np.arange(min(X), max(X), inc)
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

	def vector_drift_diff(self, vel_x, vel_y, inc_x=0.1, inc_y=0.1, t_int=0.12, dt=40, delta_t=1):
		op_x = np.arange(-1, 1, inc_x)
		op_y = np.arange(-1, 1, inc_y)

		driftX = self.drift(vel_x, t_int, dt)
		driftY = self.drift(vel_y, t_int, dt)
		diffusionX = self.diffusion(vel_x, t_int, delta_t)
		diffusionY = self.diffusion(vel_y, t_int, delta_t)
		diffusionXY = self.diffusion_xy(vel_x, vel_y, t_int, delta_t)

		avgdriftX = np.zeros((len(op_x), len(op_y)))
		avgdriftY = np.zeros((len(op_x), len(op_y)))
		avgdiffX = np.zeros((len(op_x), len(op_y)))
		avgdiffY = np.zeros((len(op_x), len(op_y)))
		avgdiffXY = np.zeros((len(op_x), len(op_y)))

		m = 0
		vel_x_, vel_y_ = vel_x[0:-dt], vel_y[0:-dt]
		#print(len(vel_x_), len(vel_y_))
		for bin_x in op_x:
			n = 0
			for bin_y in op_y:
				i = np.where(np.logical_and(np.logical_and(vel_x_<(bin_x+inc_x), vel_x_>=bin_x), np.logical_and(vel_y_<(bin_y+inc_y), vel_y_>=bin_y)))[0]
				#if not len(i): continue
				avgdriftX[n,m] = np.nanmean(driftX[i])
				avgdriftY[n,m] = np.nanmean(driftY[i])
				avgdiffX[n,m] = np.nanmean(diffusionX[i])
				avgdiffY[n,m] = np.nanmean(diffusionY[i])
				avgdiffXY[n,m] = np.nanmean(diffusionXY[i])
				n = n + 1
			m = m + 1
		return avgdriftX, avgdriftY, avgdiffX, avgdiffY, avgdiffXY, op_x, op_y




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
