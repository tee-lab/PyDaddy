import warnings

import numpy as np
import numpy.linalg
import scipy.linalg
import matplotlib.pyplot as plt
import shutil
import os
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import ridge_regression


class Metrics:
	"""
	Helper/utility module

	:meta private:
	"""
	def __init__(self, **kwargs):
		"""
		Utility function module
		"""
		self.__dict__.update(kwargs)

	def _rms(self, x):
		"""
		Calculates root mean square error of x

		Parameters
		----------
		x : array
			input 

		Returns
		-------
		rms : float
			rms error
		"""
		x = np.array(x)
		return np.sqrt(np.square(x - x.mean())).mean()
		#return np.nanmean(np.sqrt(np.square(x2 - x1)))

	def _R2(self, data, op, poly, k, adj=False):
		"""
		R-square value between the predicted and expected values

		Parameters
		----------
		data : array
			depended variable values, expected values, data
		op : array
			independent variable values
		poly : numpy.poly1d
			numpy polynomial fitted object
		k : int
			degree of the polynomial poly
		adj : bool
			if True, use R2-adjusted method instead of R2

		Returns
		-------
		R2 : float
			R2 or R2-adjusted depending upon 'adj' value
		"""
		if adj:
			return self._R2_adj(data, op, poly, k)
		return 1 - (
			np.nanmean(np.square(data - poly(op)))
			/ np.nanmean(np.square(data - np.nanmean(data)))
		)

	def _R2_adj(self, data, op, poly, k):
		"""
		Get R-squared adjusted parameter between data and fitted polynomial

		Parameters
		----------
		data : array
			depended variable values, expected values, data
		op : array
			independent variable for which the data is defined
		poly : numpy.poly1d
			numpy polynomial fitted object
		k : int
			degree of polynomial

		Returns
		-------
		R2-adjusted : folat
			R2 adjusted parameter between data and fitted polynomial
		"""
		r2 = 1 - (
			np.nanmean(np.square(data - poly(op)))
			/ np.nanmean(np.square(data - np.nanmean(data)))
		)
		n = len(op)
		return 1 - (((1 - r2) * (n - 1)) / (n - k - 1))

	def _fit_poly(self, x, y, deg):
		"""
		Fits polynomial of degree `deg`

		Parameters
		----------
		x : array
			independent variable
		y_: array
			depended variable
		deg : int
			degree of the polynomial

		Returns
		-------
		poly : numpy.poly1d
			polynomial object
		x : array
			values of x for where y in defined

		Notes
		-----
		The nan values in the input x and y (if any) will be ignored.
		"""
		nan_idx = np.argwhere(np.isnan(y))
		x_ = np.delete(x, nan_idx)
		y_ = np.delete(y, nan_idx)
		z = np.polyfit(x_, y_, deg)
		return np.poly1d(z), x_

	def _fit_poly_sparse(self, x, y, deg, threshold=0.05, alpha=0, weights=None):
		""" Fit a polynomial using sparse regression using STLSQ (Sequentially thresholded least-squares)
		Parameters:
			x, y: (np.array) Independent and dependent variables
			deg: (int) Maximum degree of the polynomial
			threshold: (float) Threshold for sparse fit.
		"""

		nan_idx = np.argwhere(np.isnan(y))
		x_ = np.delete(x, nan_idx)
		y_ = np.delete(y, nan_idx)
		weights = np.delete(weights, nan_idx)

		maxiter = deg

		dictionary = np.zeros((x_.shape[0], deg + 1))
		for d in range(deg + 1):
			dictionary[:, d] = x_ ** d

		coeffs = np.zeros(deg + 1)
		keep = np.ones_like(coeffs, dtype=np.bool)
		for it in range(maxiter):
			if np.sum(keep) == 0:
				warnings.warn('Sparsity threshold is too big, eliminated all parameters.')
				break
			# coeffs_, _, _, _ = np.linalg.lstsq(dictionary[:, keep], y_)
			coeffs_ = ridge_regression(dictionary[:, keep], y_, alpha=alpha, sample_weight=weights)
			coeffs[keep] = coeffs_
			keep = (np.abs(coeffs) > threshold)
			coeffs[~keep] = 0

		return np.poly1d(np.flipud(coeffs)), x_

	def _nan_helper(self, x):
		"""
		Helper function used to handle missing data

		Parameters
		----------
		x : array
			data

		Returns
		-------
		callable function
		"""
		return np.isnan(x), lambda z: z.nonzero()[0]

	def _interpolate_missing(self, y, copy=True):
		"""
		Interpolate missing data

		Parameters
		----------
		y : array
			data with missing (nan) values
		copy : bool, optional(default=True)
			if True makes a copy of the input array object

		Returns
		-------
		y : array
			interpolated data
		"""
		if copy:
			k = y.copy()
		else:
			k = y
		nans, x = self._nan_helper(k)
		k[nans] = np.interp(x(nans), x(~nans), k[~nans])
		return k

	def _kl_divergence(self, p, q):
		"""
		Calculates KL divergence between two probablity distrubitions p and q

		Parameters
		----------
		p : array
			distrubution p
		q : array
			distrubution q

		Returns
		-------
		kl_divergence : float
			kl divergence between p and q
		"""
		k = p * np.log(np.abs(((p + 1e-100) / (q + 1e-100))))
		# k[np.where(np.isnan(k))] = 0
		return np.nansum(k)

	def _divergence(self, a, b):
		"""
		Get the divergence between two timeseries data,
		the divergence returned here is defined as follows:
		divergence = 0.5*(KL_divergence(p,q) + KL_divergence(q,p))

		The probablity density of a and b input timeseries is calculated
		before finding the divergence.

		Parameters
		----------
		a : array
			observed timeseries data
		b : array
			simulated timeseries data
		Returns
		-------
		divergence : float
		"""
		a, bins_a = np.histogram(a, bins=100, density=True)
		b, bins_b = np.histogram(b, bins=bins_a, density=True)
		return jensenshannon(a,b)
		#a_b = np.sum(np.where((a != 0)&(b != 0), a * np.log(a / b), 0))
		#b_a = np.sum(np.where((a != 0)&(b != 0), b * np.log(b / a), 0))
		#return (a_b + b_a)/2

	def _fit_plane(self, x, y, z, order=2):
		"""
		Fits n-th order plane to data in the form z = f(x,y)
		where f(x,y) the best fit equation of plane for the data
		computed using least square method.

		Args
		----
		x : 2D array
			order parameter x
		y : 2D array
			order parameter y
		z : 2D array
			derrived drift or diffusion data
		order : int
			order of the 2D plane to fit

		Returns
		-------
		pydaddy.metrics.Plane
			A callable object takes in x and y as inputs and
			returns z = f(x,y), where f(x,y) is the fitted function
			of the plane.
		"""
		x = x[~np.isnan(z)]
		y = y[~np.isnan(z)]
		z = z[~np.isnan(z)]

		A = np.c_[np.ones(x.shape[0]), x.flatten(), y.flatten()]
		n = order

		for k in range(n+1):
			A = np.column_stack((A, x**(n-k)*y**k))

		try:
			C, _, _, _ = scipy.linalg.lstsq(A, z)
		except Exception as e:
			if e.__class__.__name__ == 'ValueError':
				print("Unable to fit plane, incompatable parameters")
			elif e.__class__.__name__ == 'LinAlgError':
				print("Unable to fit plane, computation doesnot converge")
			else:
				print("Unable to fit plane, {}".format(e.__class__.__name__))
			return None
		return Plane(coefficients=C, order=order)

	def _make_directory(self, p, i=1):
		"""
		Recursively create directorie for a given path

		Parameters
		----------
		path : str
			destination path

		Returns
		-------
		path : str
			path of created directory, same as input path.
		"""
		if type(p) != list:
			p = p.split("/")
		if p[0] == '':
			p[0] = '/'
		if i > len(p):
			return os.path.join(*p)
		else:
			try:
				os.mkdir(os.path.join(*p[0:i]))
			except (FileExistsError, FileNotFoundError):
				pass
		return self._make_directory(p, i=i + 1)

	def _get_data_range(self, x):
		"""
		Get range of the values in x, (min(x), max(x)), rounded to 3 decimal places.
		"""
		return (round(np.nanmin(x), 3),round(np.nanmax(x), 3))

	def _remove_nan(self, x, y):
		"""
		Removes NaN's by deleting the indices where both `x` and `y` have NaN's

		Parameters
		----------
		x : array
			first input
		y : array
			second input

		Returns
		-------
		array
			x, y - with all nan's removed

		"""
		nan_idx = np.where(np.isnan(x)) and np.where(np.isnan(y))
		return np.array([np.delete(x, nan_idx), np.delete(y, nan_idx)])

	#def _isValidSliderRange(self, r):
		"""
		Checks if the given range for slider is a valid range

		Parameters
		----------
		r : list, tuple
			range of order parameter

		Returns
		-------
		bool
			True if valid, else False

		"""
	#	if r is None or r == 'default':
	#		return True
	#	if isinstance(r, (list, tuple)) and len(r) == 3 and (np.array(r) >= 1).all():
	#		return True
	#	return False

	def _is_valid_slider_timescale_list(self, slider_list):
		"""
		Checks if the given slider timescale lists contains valid entries

		Parameters
		----------
		slider_list : list, tuple
			timescales to include in the slider

		Returns
		-------
		bool
			True if all values are valid, else False 
		"""
		if slider_list is None:
			return False
		if (
			isinstance(slider_list, (list, tuple, range))
			and (np.array(slider_list) >= 1).all()
		):
			return True
		return False

	#def _get_slider_timescales(self, slider_range, slider_scale_list):
		"""
		Times scales to generate the drift and diffusion plot slider

		Parameters
		----------
		slider_range : list, tuple
			range for the slider

		slider_scale_list : list, tuple
			timescales to include in the slider

		Returns
		-------
		list
			sorted list of the timescales to include in the slider

		Notes
		-----
		All dublicate values in the list (if any) will be removed
		"""
		#t_list = []
		#default_range = (1, np.ceil(self.autocorrelation_time) * 2, 8)
		#if self._isValidSliderTimesSaleList(slider_scale_list):
		#	t_list = slider_scale_list

		#if self._isValidSliderRange(slider_range):
		#	if slider_range is None:
		#		slider_start, slider_stop, n_step = 0, 0, 0
		#	elif slider_range == 'default':
		#		slider_start, slider_stop, n_step = default_range
		#	else:
		#		slider_start, slider_stop, n_step = slider_range
		#else:
		#	slider_start, slider_stop, n_step = default_range
		#	self.slider_range = (slider_start, slider_stop, n_step)
		#return sorted(set(map(int, np.linspace(slider_start, slider_stop, n_step))))
		#sreturn sorted(set(map(int, np.concatenate((np.linspace(slider_start, slider_stop, n_step), t_list)))).union(set([self.dt, self.Dt])))

	def _closest_time_scale(self, time_scale, slider):
		"""
		Gives closest matching time scale avaiable from the slider keys.
		"""
		timescale = list(slider.keys())
		i = np.abs(np.array(timescale) - time_scale).argmin()
		return timescale[i]

	def _get_data_from_slider(self, drift_time_scale=None, diff_time_scale=None):
		"""
		Get drift and diffusion data from slider data dictionary, if key not valid, returns the data corresponding to closest matching one.
		"""
		if self.vector:
			if drift_time_scale is None:
					drift_x, drift_y = self._data_avgdriftX,self._data_avgdriftY
			if diff_time_scale is None:
				diff_x, diff_y, diff_xy, diff_yx = self._data_avgdiffX,	self._data_avgdiffY, self._data_avgdiffXY, self._data_avgdiffYX

			if drift_time_scale is not None:
				if drift_time_scale not in self._drift_slider.keys():
					print("\n{} not in list:\n{}".format(drift_time_scale, self._drift_slider.keys()))
					drift_time_scale = self._closest_time_scale(drift_time_scale, self._drift_slider)
					print("Choosing {}; (closest matching timescale from the avaiable ones)".format(drift_time_scale))
				drift_x, drift_y = self._drift_slider[drift_time_scale][0],self._drift_slider[drift_time_scale][1]

			if diff_time_scale is not None:
				if diff_time_scale not in self._diff_slider.keys():
					print("\n{} not in list:\n{}".format(diff_time_scale, self._diff_slider.keys()))
					diff_time_scale = self._closest_time_scale(diff_time_scale, self._diff_slider)
					print("Choosing {}; (closest matching timescale from the avaiable ones)".format(diff_time_scale))
				diff_x, diff_y = self._diff_slider[diff_time_scale][0],	self._diff_slider[diff_time_scale][1]
				diff_xy, diff_yx = self._cross_diff_slider[diff_time_scale][0],	self._cross_diff_slider[diff_time_scale][1]

			return drift_x, drift_y, diff_x, diff_y, diff_xy, diff_yx
		else:
			if drift_time_scale is None:
				drift = self._data_avgdrift

			if diff_time_scale is None:
				diff = self._data_avgdiff

			if drift_time_scale is not None:
				if drift_time_scale not in self._drift_slider.keys():
					print("\n{} not in list:\n{}".format(drift_time_scale, self._drift_slider.keys()))
					drift_time_scale = self._closest_time_scale(drift_time_scale, self._drift_slider)
					print("Choosing {}; (closest matching timescale from the avaiable ones)".format(drift_time_scale))
				drift = self._drift_slider[drift_time_scale][0]

			if diff_time_scale is not None:
				if diff_time_scale not in self._diff_slider.keys():
					print("\n{} not in list:\n{}".format(diff_time_scale, self._diff_slider.keys()))
					diff_time_scale = self._closest_time_scale(diff_time_scale, self._diff_slider)
					print("Choosing {}; (closest matching timescale from the avaiable ones)".format(diff_time_scale))
				diff = self._diff_slider[diff_time_scale][0]

			return drift, diff

	def _get_num_points(self, drift_time_scale, diff_time_scale):
		if self.vector:
			raise NotImplementedError('_get_num_points() is not implemented for vector data.')

		if drift_time_scale is None:
			drift_num = self._data_drift_num
		else:
			if drift_time_scale not in self._drift_slider.keys():
				print("\n{} not in list:\n{}".format(drift_time_scale, self._drift_slider.keys()))
				drift_time_scale = self._closest_time_scale(drift_time_scale, self._drift_slider)
				print("Choosing {}; (closest matching timescale from the avaiable ones)".format(drift_time_scale))
			drift_num = self._data_drift_nums[drift_time_scale]


		if diff_time_scale is None:
			diff_num = self._data_diff_num
		else:
			if diff_time_scale not in self._diff_slider.keys():
				print("\n{} not in list:\n{}".format(diff_time_scale, self._diff_slider.keys()))
				diff_time_scale = self._closest_time_scale(diff_time_scale, self._diff_slider)
				print("Choosing {}; (closest matching timescale from the avaiable ones)".format(diff_time_scale))
			diff_num = self._data_diff_nums[diff_time_scale]

		return np.array(drift_num), np.array(diff_num)

	def _stack_slider_data(self, d, slider_data, index):
		"""
		Stack data from slider dictionary, corresponding to the given index, into columns of numpy array.
		"""
		for i in slider_data:
			d = np.column_stack((d, slider_data[i][index].flatten()))
		return d

	def _csv_header(self, prefix, file_name):
		"""
		Generate headers for CSV file.
		"""
		headers = "x,"
		if self.vector:
			headers = "x,y,"
		if 'drift' in file_name:
			timescales = list(self._drift_slider)
		else:
			timescales = list(self._diff_slider)
		for i in timescales:
			headers = headers + "{}-{},".format(prefix, i)
		return headers

	def _get_stacked_data(self):
		"""
		Get a dictionary of all (op_x, op_y, driftX, driftY, diffX, diffY) slider data stacked into numpy arrays.
		"""
		data_dict = dict()
		if self.vector:
			x, y = np.meshgrid(self._data_op_x, self._data_op_y)
			data = np.vstack((x.flatten(), y.flatten())).T
			data_dict["drift_x"] = self._stack_slider_data(
				data.copy(), self._drift_slider, index=0
			)
			data_dict["drift_y"] = self._stack_slider_data(
				data.copy(), self._drift_slider, index=1
			)
			data_dict["diffusion_x"] = self._stack_slider_data(
				data.copy(), self._diff_slider, index=0
			)
			data_dict["diffusion_y"] = self._stack_slider_data(
				data.copy(), self._diff_slider, index=1
			)
			data_dict["diffusion_xy"] = self._stack_slider_data(
				data.copy(), self._cross_diff_slider, index=0)
			data_dict["diffusion_yx"] = self._stack_slider_data(
				data.copy(), self._cross_diff_slider, index=1)
		else:
			data = self._data_op
			data_dict["drift"] = self._stack_slider_data(
				data.copy(), self._drift_slider, index=0
			)
			data_dict["diffusion"] = self._stack_slider_data(
				data.copy(), self._diff_slider, index=0
			)
		return data_dict

	def _save_csv(self, dir_path, file_name, data, fmt="%.4f", add_headers=True):
		"""
		Save data to CSV file.
		"""
		if not file_name.endswith(".csv"):
			file_name = file_name + ".csv"
		savepath = os.path.join(dir_path, file_name)
		prefix = "Dt" if "drift" in file_name else "dt"
		headers = self._csv_header(prefix, file_name) if add_headers else ""
		np.savetxt(savepath, data, fmt=fmt, header=headers, delimiter=",", comments="")
		return None

	def _combined_data_dict(self):
		"""
		Get all drift and diffusion data in dictionary format.
		"""
		combined_data = dict()
		if self.vector:
			k = ["x", "y"]
			k_ = ["xy", "yx"]
			combined_data["x"] = self._data_op_x
			combined_data["y"] = self._data_op_y
			for i in self._drift_slider:
				for j in range(2):
					drift_key = "drift_{}_{}".format(k[j], i)
					combined_data[drift_key] = self._drift_slider[i][j]
			for i in self._diff_slider:
				for j in range(2):
					diff_key = "diffusion_{}_{}".format(k[j], i)
					cross_diff_key = "diffusion_{}_{}".format(k_[j], i)
					combined_data[diff_key] = self._diff_slider[i][j]
					combined_data[cross_diff_key] = self._cross_diff_slider[i][j]

		else:
			combined_data["x"] = self._data_op
			for i in self._drift_slider:
				drift_key = "drift_{}".format(i)
				combined_data[drift_key] = self._drift_slider[i][0]

			for i in self._diff_slider:
				diff_key = "diffusion_{}".format(i)
				combined_data[diff_key] = self._diff_slider[i][0]
		return combined_data

	def _zip_dir(self, dir_path):
		"""
		Make ZIP file of the exported result.
		"""
		file_name = os.path.dirname(dir_path)
		return shutil.make_archive(dir_path, "zip", dir_path)

	def _isnotebook(self):
		try:
			shell = get_ipython().__class__.__name__
			#print(shell)
			if shell == 'ZMQInteractiveShell':
				return True   # Jupyter notebook or qtconsole
			elif shell == 'TerminalInteractiveShell':
				return False  # Terminal running IPython
			else:
				return False  # Other type (?)
		except NameError:
			return False      # Probably standard Python interpreter


"""
class Plane:
	""
	Create first or second order plane surfaces.
	
	:meta private:
	""
	def __init__(self, coefficients, order):
		self.coeff = coefficients
		self.order = order

	def expr(self):
		a = sympy.Symbol('x')
		b = sympy.Symbol('y')
		c = self.coeff
		n = self.order
		#expr = "{} + {}*x + {}*y".format(c[0], c[1], c[2])
		expr = c[0] + a*c[1] + b*c[2]
		for k in range(n+1):
			expr = expr + c[3+k]*a**(n-k)*b**k
			#expr = expr + " + {}*x**{}*y**{}".format(c[3-k], n-k, k)
		return expr

	def __repr__(self):
		return str(self.expr())

	def __call__(self, x, y):
		X = x.flatten()
		Y = y.flatten()
		a = np.c_[np.ones(X.shape[0]), X, Y]
		n = self.order
		for k in range(n+1):
			a = np.column_stack((a, X**(n-k)*Y**k))
		return np.dot(a, self.coeff).reshape(x.shape)
"""