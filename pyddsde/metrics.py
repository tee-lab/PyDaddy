import numpy as np
import scipy.linalg
import shutil
import os


class metrics:
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
        return np.sum(k)

    def _fit_plane(
        self, x, y, z, order=2, inc_x=0.1, inc_y=0.1, range_x=(-1, 1), range_y=(-1, 1)
    ):
        """
        Fits first order or second order plane to the surface data points

        Parameters
        ----------
        x : array_like
            x values of meshgrid
        y : array_like
            y values of meshgrid
        z : array_like
            data points, function of x and y [z = f(x,y)]
        order : int, optional(default=2)
            1 or 2, order of the plane to fit
        inc_x : float, optional(default=0.1)
            increment in x
        inc_y : float, optional(default=0.1)
            increment in y
        range_x : tuple, optional(default=(-1,1))
            range of x
        range_y : tuple, optional(default=(-1,1))
            range of y

        Returns
        -------
        plane : pyddsde.metrics.Plane
            plane object

        Notes
        -----
        This is an experimental implementation of plane fitting function.
        Hence it may not work will all types of data.
        """
        x = x[~np.isnan(z)]
        y = y[~np.isnan(z)]
        z = z[~np.isnan(z)]
        data = np.array(list(zip(x, y, z)))

        x_, y_ = np.meshgrid(
            np.arange(range_x[0], range_x[-1], inc_x),
            np.arange(range_y[0], range_y[-1], inc_y),
        )
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
            A = np.c_[
                np.ones(data.shape[0]),
                data[:, :2],
                np.prod(data[:, :2], axis=1),
                data[:, :2] ** 2,
            ]
            C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
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
        if i > len(p):
            return os.path.join(*p)
        else:
            try:
                os.mkdir(os.path.join(*p[0:i]))
            except FileExistsError:
                pass
        return self._make_directory(p, i=i + 1)

    def _get_data_range(self, x):
        """
        Get range of the values in x, (min(x), max(x)), rounded to 3 decimal places.
        """
        return (round(min(x), 3),round(max(x), 3))

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

    def _isValidSliderRange(self, r):
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
        if r is None:
            return False
        if isinstance(r, (list, tuple)) and len(r) == 3 and (np.array(r) >= 1).all():
            return True
        return False

    def _isValidSliderTimesSaleList(self, slider_list):
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
            isinstance(slider_list, (list, tuple))
            and (np.array(slider_list) >= 1).all()
        ):
            return True
        return False

    def _get_slider_timescales(self, slider_range, slider_scale_list):
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
        t_list = []
        if self._isValidSliderTimesSaleList(slider_scale_list):
            t_list = slider_scale_list

        if self._isValidSliderRange(slider_range):
            slider_start, slider_stop, n_step = slider_range
        else:
            if len(t_list):
                return sorted(set(map(int, t_list)))
            slider_start = 1
            slider_stop = np.ceil(self.autocorrelation_time) * 2
            n_step = 8
        self.slider_range = (slider_start, slider_stop, n_step)
        #return sorted(set(map(int, np.linspace(slider_start, slider_stop, n_step))))
        return sorted(set(map(int, np.concatenate((np.linspace(slider_start, slider_stop, n_step), t_list)))).union(set([self.dt])))

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
                return (
                    self._data_avgdriftX,
                    self._data_avgdriftY,
                    self._data_avgdiffX,
                    self._data_avgdiffY,
                )
            if time_scale not in self._time_scale_list:
                print("\n{} not in list:\n{}".format(time_scale, self._time_scale_list))
                time_scale = self._closest_time_scale(time_scale)
                print(
                    "Choosing {}; (closest matching timescale from the avaiable ones)".format(
                        time_scale
                    )
                )
            return (
                self._drift_slider[time_scale][0],
                self._drift_slider[time_scale][1],
                self._diff_slider[time_scale][0],
                self._diff_slider[time_scale][1],
            )
        else:
            if time_scale is None:
                return self._data_avgdrift, self._data_avgdiff
            if time_scale not in self._time_scale_list:
                print("\n{} not in list:\n{}".format(time_scale, self._time_scale_list))
                time_scale = self._closest_time_scale(time_scale)
                print(
                    "Choosing {}; (closest matching timescale from the avaiable ones)".format(
                        time_scale
                    )
                )
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
        headers = self._csv_header(prefix) if add_headers else ""
        np.savetxt(savepath, data, fmt=fmt, header=headers, delimiter=",", comments="")
        return None

    def _combined_data_dict(self):
        """
        Get all drift and diffusion data in dictionary format.
        """
        combined_data = dict()
        if self.vector:
            k = ["x", "y"]
            combined_data["x"] = self._data_op_x
            combined_data["y"] = self._data_op_y
            for i in self._drift_slider:
                for j in range(2):
                    drift_key = "drift_{}_{}".format(k[j], i)
                    diff_key = "diffusion_{}_{}".format(k[j], i)
                    combined_data[drift_key] = self._drift_slider[i][j]
                    combined_data[diff_key] = self._diff_slider[i][j]
        else:
            combined_data["x"] = self._data_op
            for i in self._drift_slider:
                drift_key = "drift_{}".format(i)
                diff_key = "diffusion_{}".format(i)
                combined_data[drift_key] = self._drift_slider[i][0]
                combined_data[diff_key] = self._diff_slider[i][0]
        return combined_data

    def _zip_dir(self, dir_path):
        """
        Make ZIP file of the exported result.
        """
        file_name = os.path.dirname(dir_path)
        return shutil.make_archive(dir_path, "zip", dir_path)


class Plane:
    """
    Create first or second order plane surfaces.
    
    :meta private:
    """
    def __init__(self, coefficients, order):
        self.coeff = coefficients
        self.order = order

    def __str__(self):
        str1 = """2D plane\nOrder: {}\nCoeff: {}""".format(self.order, self.coeff)
        return str1

    def __call__(self, x, y):
        if self.order == 1:
            X = x.flatten()
            Y = y.flatten()
            return np.dot(np.c_[X, Y, np.ones(X.shape)], self.coeff).reshape(x.shape)
        elif self.order == 2:
            X = x.flatten()
            Y = y.flatten()
            return np.dot(
                np.c_[np.ones(X.shape), X, Y, X * Y, X ** 2, Y ** 2], self.coeff
            ).reshape(x.shape)
