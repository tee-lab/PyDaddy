import numpy as np
import scipy.linalg
import shutil
import os

class metrics:
	def __init__(self, **kwargs):
		"""
		Utility function module
		"""
		self.__dict__.update(kwargs)

	def _rms(self, x1, x2):
		"""
		Calculates root mean square error

		Input params:
		--------------
		x1 : numpy.ndarray
		x2 : numpy.ndarray
		
		returns:
		--------------
		rms : float
		"""
		return np.nanmean(np.sqrt(np.square(x2 - x1)))

	def _R2(self, data, op, poly, k, adj=False):
		"""
		R-square value between the predicted and expected values

		Input params:
		--------------
		data : numpy.ndarray
			depended variable data of expected values
		op : numpy.ndarray
			independent variable values
		poly : numpy.poly1d
			the fit polynomial
		k : int
			degree of polynomial fit
		adj : bool
			if True, use R2-adjusted method instead of R2

		returns:
		--------------
		R2 : float
			R2 or R2-adjusted depending upon 'adj' value
		"""
		if adj: return self._R2_adj(data, op, poly, k)
		return 1 - (np.nanmean(np.square(data - poly(op))) /
					np.nanmean(np.square(data - np.nanmean(data))))

	def _R2_adj(self, data, op, poly, k):
		"""
		R-squared adjusted method
		"""
		r2 = 1 - (np.nanmean(np.square(data - poly(op))) /
				  np.nanmean(np.square(data - np.nanmean(data))))
		n = len(op)
		return 1 - (((1 - r2) * (n - 1)) / (n - k - 1))

	def _fit_poly(self, x, y, deg):
		"""
		Fits polynomial f(x) = y

		Input params:
		--------------
		x  : numpy.ndarray
			independent variable
		y : numpy.ndarray
			depended variable
		deg : int
			degree of the polynomial

		returns:
		-------------
		poly : np.poly1d
			polynomial object
		x : numpy.ndarray
			values of x for where y in defined
		"""
		nan_idx = np.argwhere(np.isnan(y))
		x_ = np.delete(x, nan_idx)
		y_ = np.delete(y, nan_idx)
		z = np.polyfit(x_, y_, deg)
		return np.poly1d(z), x_

	def _nan_helper(self, x):
		"""
		Helper function used to handle missing data

		Input params:
		-------------
		x : numpy.ndarray
			data

		returns:
		-------------
		np.isnan(x) : numpy.array
			array of bools, true if x is numpy.nan else false
		z : callable function

		"""
		return np.isnan(x), lambda z: z.nonzero()[0]

	def _interpolate_missing(self, y):
		"""
		Interpolate missing data

		Input params:
		-------------
		y : numpy.array
			array with missing (nan) values

		returns:
		-------------
		y : numpy.array
			missing values are interpolated
		"""
		nans, x = self._nan_helper(y)
		y[nans] = np.interp(x(nans), x(~nans), y[~nans])
		return y

	def _kl_divergence(self, p, q):
		"""
		Calculates KL divergence between two probablity distrubitions

		Input params:
		-------------
		p : np.array
			distrubution p
		q : np.array
			distrubution q

		returns:
		-------------
		kl_divergence : float
			kl divergence between p and q

		"""
		k = p * np.log(np.abs(((p + 1e-100) / (q + 1e-100))))
		#k[np.where(np.isnan(k))] = 0
		return np.sum(k)

	def _fit_plane(self,
				   x,
				   y,
				   z,
				   order=2,
				   inc_x=0.1,
				   inc_y=0.1,
				   range_x=(-1, 1),
				   range_y=(-1, 1)):
		"""
		Fits first order or second order plane to the surface data points

		Input params:
		-------------
		x : numpy.ndarray
			x values of meshgrid
		y : numpy.ndarray
			y values of meshgrid
		z : numpy.ndarray
			data points, function of x and y [z = f(x,y)]
		order = 2 : int
			1 or 2, order of the plane to fit
		inc_x = 0.1 : float
			increment in x
		inc_y = 0.1 : float
			increment in y
		range_x = (-1,1) : tuple
			range of x
		range_y = (-1,1) : tuple
			range of y

		returns:
		-----------
			plane : pyFish.metrics.Plane
				plane object
		"""
		x = x[~np.isnan(z)]
		y = y[~np.isnan(z)]
		z = z[~np.isnan(z)]
		data = np.array(list(zip(x, y, z)))

		x_, y_ = np.meshgrid(np.arange(range_x[0], range_x[-1], inc_x),
							 np.arange(range_y[0], range_y[-1], inc_y))
		X = x_.flatten()
		Y = y_.flatten()

		if order == 1:
			# best-fit linear plane
			A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
			C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
			return Plane(coefficients=C, order=order)
		else:
			order = 2
			# best-fit quadratic curve
			A = np.c_[np.ones(data.shape[0]), data[:, :2],
					  np.prod(data[:, :2], axis=1), data[:, :2]**2]
			C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
			return Plane(coefficients=C, order=order)

	def _make_directory(self, p, i=1):
		"""
		Recursively create directories in given path

		Input params:
		--------------
		path : str
			destination path

		returns:
		-------------
		path : str
		"""
		if type(p) != list: p = p.split('/')
		if i > len(p):
			return os.path.join(*p)
		else:
			try:
				os.mkdir(os.path.join(*p[0:i]))
			except FileExistsError:
				pass
		return self._make_directory(p, i=i + 1)

	def _remove_nan(self, Mx, My):
		"""
		Removes NaN's from data
		"""
		nan_idx = (np.where(np.isnan(Mx)) and np.where(np.isnan(My)))
		return np.array([np.delete(Mx, nan_idx), np.delete(My, nan_idx)])

	def _isValidSliderRange(self, r):
		"""
		Checks if the given range for slider is a valid range
		"""
		if r is None:
			return False
		if isinstance(r, (list, tuple)) and len(r) == 3 and (np.array(r) >= 1).all():
			return True
		print('\n[Warning] : Entered slider range is not in valid format. Using default range.\nValid format <(slider_start, slider_stop, n_steps)>\nAll values must be >= 1\n')
		return False

	def _isValidSliderTimesSaleList(self, slider_list):
		"""
		Checks if the given slider timescale lists contains valid entries
		"""
		if slider_list is None:
			return False
		if isinstance(slider_list, (list, tuple)) and (np.array(slider_list) >=1).all():
			return True
		print('\n[Warning] : Given slider timescale list is not valid\nUsing default range')
		return False

	def _get_slider_timescales(self, slider_range, slider_scale_list):
		"""
		Times scales to generate the drift and diffusion plot slider
		"""
		if self._isValidSliderTimesSaleList(slider_scale_list):
			self.slider_range = None
			return sorted(list(slider_scale_list))

		if self._isValidSliderRange(slider_range):
			slider_start, slider_stop, n_step = slider_range
		else:
			slider_start = 1
			slider_stop = np.ceil(self.autocorrelation_time)*2
			n_step = 8
		self.slider_range = (slider_start, slider_stop, n_step)
		return sorted(set(map(int, np.linspace(slider_start, slider_stop, n_step))))

	def _closest_time_scale(self, time_scale):
		"""
		Gives closest matching time scale avaiable from the timescale list.
		"""
		i = np.abs(np.array(self._time_scale_list) - time_scale).argmin()
		return self._time_scale_list[i]

	def _get_data_from_slider(self, time_scale=None):
		"""
		Get drift and diffusion data from slider data dictionary, if key not valid, returns the data corresponding to closest matching one.
		"""
		if self.vector:
			if time_scale is None:
				return self._data_avgdriftX, self._data_avgdriftY, self._data_avgdiffX, self._data_avgdiffY
			if time_scale not in self._time_scale_list:
				print("\n{} not in list:\n{}".format(time_scale, self._time_scale_list))
				time_scale = self._closest_time_scale(time_scale)
				print("Choosing {}; (closest matching timescale from the avaiable ones)".format(time_scale))
			return self._drift_slider[time_scale][0],self._drift_slider[time_scale][1], self._diff_slider[time_scale][0], self._diff_slider[time_scale][1]
		else:
			if time_scale is None:
				return self._data_avgdrift, self._data_avgdiff
			if time_scale not in self._time_scale_list:
				print("\n{} not in list:\n{}".format(time_scale, self._time_scale_list))
				time_scale = self._closest_time_scale(time_scale)
				print("Choosing {}; (closest matching timescale from the avaiable ones)".format(time_scale))
			return self._drift_slider[time_scale][0], self._diff_slider[time_scale][0]

	def _stack_slider_data(self, d, slider_data, index):
		"""
		Stack data from slider dictionary, corresponding to the given index, into columns of numpy array.
		"""
		for i in slider_data:
			d = np.column_stack((d, slider_data[i][index].flatten()))
		return d

	def _csv_header(self, prefix):
		"""
		Generate headers for CSV file.
		"""
		headers = "x,"
		if self.vector:
			headers = "x,y,"
		for i in self._drift_slider:
			headers = headers + "{} {},".format(prefix, i)
		return headers

	def _get_stacked_data(self):
		"""
		Get a dictionary of all (op_x, op_y, driftX, driftY, diffX, diffY) slider data stacked into numpy arrays.
		"""
		data_dict = dict()
		if self.vector:
			x, y = np.meshgrid(self._data_op_x, self._data_op_y)
			data = np.vstack((x.flatten(), y.flatten())).T
			data_dict['drift_x'] = self._stack_slider_data(data.copy(), self._drift_slider, index=0)
			data_dict['drift_y'] = self._stack_slider_data(data.copy(), self._drift_slider, index=1)
			data_dict['diffusion_x'] = self._stack_slider_data(data.copy(), self._diff_slider, index=0)
			data_dict['diffusion_y'] = self._stack_slider_data(data.copy(), self._diff_slider, index=1)
		else:
			data = self._data_op
			data_dict['drift'] = self._stack_slider_data(data.copy(), self._drift_slider, index=0)
			data_dict['diffusion'] = self._stack_slider_data(data.copy(), self._diff_slider, index=0)
		return data_dict

	def _save_csv(self, dir_path, file_name, data, fmt='%.4f', add_headers=True):
		"""
		Save data to CSV file.
		"""
		if not file_name.endswith('.csv'):
			file_name = file_name + '.csv'
		savepath = os.path.join(dir_path, file_name)
		prefix = 'Dt' if 'drift' in file_name else 'dt'
		headers = self._csv_header(prefix) if add_headers else ''
		np.savetxt(savepath, data, fmt=fmt, header=headers, delimiter=',', comments="")
		return None

	def _combined_data_dict(self):
		"""
		Get all drift and diffusion data in dictionary format.
		"""
		combined_data = dict()
		if self.vector:
			k = ['x', 'y']
			combined_data['x'] = self._data_op_x
			combined_data['y'] = self._data_op_y
			for i in self._drift_slider:
				for j in range(2):
					drift_key = 'drift_{}_{}'.format(k[j], i)
					diff_key = 'diffusion_{}_{}'.format(k[j], i)
					combined_data[drift_key] = self._drift_slider[i][j]
					combined_data[diff_key] = self._diff_slider[i][j]
		else:
			combined_data['x'] = self._data_op
			for i in self._drift_slider:
				drift_key = 'drift_{}'.format(i)
				diff_key = 'diffusion_{}'.format(i)
				combined_data[drift_key] = self._drift_slider[i][0]
				combined_data[diff_key] = self._diff_slider[i][0]
		return combined_data

	def _zip_dir(self, dir_path):
		"""
		Make ZIP file of the exported result.
		"""
		file_name = os.path.dirname(dir_path)
		return shutil.make_archive(dir_path, 'zip', dir_path)




class Plane:
	def __init__(self, coefficients, order):
		self.coeff = coefficients
		self.order = order

	def __str__(self):
		str1 = """2D plane\nOrder: {}\nCoeff: {}""".format(
			self.order, self.coeff)
		return str1

	def __call__(self, x, y):
		if self.order == 1:
			X = x.flatten()
			Y = y.flatten()
			return np.dot(np.c_[X, Y, np.ones(X.shape)],
						  self.coeff).reshape(x.shape)
		elif self.order == 2:
			X = x.flatten()
			Y = y.flatten()
			return np.dot(np.c_[np.ones(X.shape), X, Y, X * Y, X**2, Y**2],
						  self.coeff).reshape(x.shape)
