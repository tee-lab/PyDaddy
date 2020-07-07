import numpy as np 
import scipy.linalg

class metrics:
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

	def rms(self,x1,x2):
		return np.nanmean(np.sqrt(np.square(x2 - x1)))

	def R2(self,data,op,poly,k,adj=False):
		if adj: return self.R2_adj(data, op, poly, k)
		return 1 - (np.nanmean(np.square(data - poly(op)))/np.nanmean(np.square(data - np.nanmean(data))))

	def R2_adj(self, data, op, poly, k):
		r2 = 1 - (np.nanmean(np.square(data - poly(op)))/np.nanmean(np.square(data - np.nanmean(data))))
		n = len(op)
		return 1-(((1-r2)*(n-1))/(n-k-1))

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

	def fit_plane(self, x,y,z, order=2, inc_x=0.1, inc_y=0.1, range_x=(-1,1), range_y=(-1,1)):
		x = x[~np.isnan(z)]
		y = y[~np.isnan(z)]
		z = z[~np.isnan(z)]
		data = np.array(list(zip(x,y,z)))

		x_, y_ = np.meshgrid(np.arange(range_x[0], range_x[-1], inc_x), np.arange(range_y[0], range_y[-1], inc_y))
		X = x_.flatten()
		Y = y_.flatten()

		if order == 1:
			# best-fit linear plane
			A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
			C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
			return Plane(coefficients=C, order=order)
		elif order == 2:
			# best-fit quadratic curve
			A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
			C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
			return Plane(coefficients=C, order=order)


class Plane:
	def __init__(self, coefficients, order):
		self.coeff = coefficients
		self.order = order

	def __str__(self):
		str1 = """2D plane\nOrder: {}\nCoeff: {}""".format(self.order,self.coeff)
		return str1

	def __call__(self, x, y):
		if self.order == 1:
			X = x.flatten()
			Y = y.flatten()
			return np.dot(np.c_[X, Y, np.ones(X.shape)], self.coeff).reshape(x.shape)
		elif self.order == 2:
			X = x.flatten()
			Y = y.flatten()
			return np.dot(np.c_[np.ones(X.shape), X, Y, X*Y, X**2, Y**2], self.coeff).reshape(x.shape)

