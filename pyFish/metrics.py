import numpy as np 

class metrics:
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

	def rms(self,x1,x2):
		return np.nanmean(np.sqrt(np.square(x2 - x1)))

	def R2(self,data,op,poly):
		return 1 - (np.nanmean(np.square(data - poly(op)))/np.nanmean(np.square(data - np.nanmean(data))))

	def fit_poly(self,x,y,deg):
		nan_idx = np.argwhere(np.isnan(y))
		x_ = np.delete(x,nan_idx)
		y_ = np.delete(y,nan_idx)
		z = np.polyfit(x_,y_,deg)
		return np.poly1d(z), x_

	def nan_helper(self, x):
		return np.isnan(x), lambda z: z.nonzero()[0]

	def interploate_missing(self, y):
		nans, x = self.nan_helper(y)
		y[nans] = np.interp(x(nans), x(~nans), y[~nans])
		return y

	def kl_divergence(self, p, q):
		k = p*np.log(np.abs(((p+1e-100)/(q+1e-100))))
		#k[np.where(np.isnan(k))] = 0
		return np.sum(k)