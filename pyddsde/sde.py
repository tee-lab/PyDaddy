import numpy as np

class SDE:
    """
    A class to form a basic SDE from data

    Args
    ----
    X : array_like
            time series data
    t_int : float
            time step in time series
    dt : int
            analysis time step
    inc : float
            max increment for binning thae data

    Returns
    -------
    diff : array_like
            diffusion in time series
    drift : array_like
            drift in time series
    avgdiff : array_like
            avarage diffusion
    avgdrift : array_like
            average drift

    :meta private:
    """

    def __init__(self, **kwargs):
        """
        pass

        .. document private functions
        """
        self.__dict__.update(kwargs)

    def _drift(self, X, t_int, dt):
        r"""
        Get Drift coeffecient vector of data.

        Args
        ----
        X : array_like
                Time Series data
        t_int : float
                time difference betwen consecutive observations
        dt : float
                drift calculation timescale

        Returns
        -------
        diff : array_like
            Diffusion in time series

        Notes
        -----
        Drift is calculated as follows

        .. math::
            drift = \frac{x(i+dt)-x(i)}{tint * dt}
        """

        # return np.array([b - a for a, b in zip(X, X[dt:])]) / (t_int * dt)
        return (X[dt:] - X[:-dt]) / (t_int * dt)

    def _residual(self, X, t_int, dt, delta_t=1):
        """
        Get the residual.
        """
        p = len(X) - max(delta_t, dt)
        drift = self._drift(X, t_int, dt)[:p]
        res = (X[delta_t:] - X[:-delta_t])[:p]
        return res - drift*(t_int*delta_t)

    def _diffusion(self, X, t_int, delta_t=1):
        """
        Get Diffusion coefficient vector of data

        Parameters
        ----------
        X : array_like
            time series data
        t_int : float
            time step in time series
        delta_t : int
            diffusion calculation timescale

        Returns
        --------
        diff : array.
            Diffusion
        """

        # return np.square(np.array([b - a for a, b in zip(X, X[delta_t:])])) / (t_int * delta_t)
        return np.square(X[delta_t:] - X[:-delta_t]) / (t_int * delta_t)

    def _diffusion_xy(self, x, y, t_int, delta_t):
        """
        Get cross-correlation coefficients between x and y arrays.

        Args
        ----
        x : numpy.array
                x data
        y : numpy.array
                y data
        t_int : float
                time difference betwen consecutive observations
        delta_t : diffusion calculation timescale

        Returns
        -------
        diffusion_xy : numpy.array
                cross-correlation coefficients between x and y data
        """
        return np.array(
            [(b - a) * (d - c) for a, b, c, d in zip(x, x[delta_t:], y, y[delta_t:])]
        ) / (delta_t * t_int)

    def _isValidRange(self, r):
        """
        Checks if the specified range of order parameter is valid range

        Args
        ----
        r : tuple, list
                range of order parameter

        Returns
        -------
        bool
                True if valid, False if not.
        """
        return isinstance(r, (list, tuple)) and len(r) == 2

    def _order_parameter(self, X, inc, r):
        """
        Get order parameter array for a given range and increments

        If range is None or not valid, order parameter array will be generated considering
        maxmium and mimimum limits of the data as the range

        Args
        ----
        X : numpy.array
                data
        inc : float
                step increments in order parameter
        r : tuple, list
                range of the order parameter

        Returns
        -------
        tuple
                first element will be the order parameter array
                second element is the rangen used
        """
        if r is None:
            r = (min(X), max(X))
        if not self._isValidRange(r):
            print(
                "Warning : given order parameter range is not in valid (typle or list of length 2) format\nUsing range of data"
            )
            r = (min(X), max(X))
        return np.arange(min(r), max(r)+inc, inc), r

    def _drift_and_diffusion(self, X, t_int, dt, delta_t, inc):
        """
        Get drift and diffusion coefficients for a given timeseries data

        Args
        ----
        X : numpy.array
                time series data
        t_int : float
                time difference betwen consecutive observations
        dt : int
                timescale to calculate drift
        delta_t : int
                timescale to claculate diffusion
        inc : float
                step increments in order parameter

        Returns
        -------
        diff : array
            diffusion of the data
        drift : array.
            drift, of the data
        avgdiff : array
            average diffusion
        avgdrift : array
            average drift 
        op : array
            order parameter
        """
        op, self.op_range = self._order_parameter(X, inc, self.op_range)
        avgdiff, avgdrift = [], []
        drift = self._drift(X, t_int, dt)
        diff = self._diffusion(X, t_int, delta_t=delta_t)
        X = X[0 : -max(dt, delta_t)]
        for b in op:
            i = np.where(np.logical_and(X < (b + inc), X >= b))[0]
            avgdiff.append(diff[i].mean())
            avgdrift.append(drift[i].mean())
        return diff, drift, np.array(avgdiff), np.array(avgdrift), op

    def _vector_drift_diff(self, x, y, inc_x, inc_y, t_int, dt, delta_t):
        """
        Get average binned drift and diffusion coefficients for given x and y data

        Args
        ----
        x : array_like
            timeseries x data
        y : array_like
            timesereis y data
        inc_x : float
            step increment of order parameter for x
        inc_y : float
            step increment of order parameter for y
        dt : int
            timescale to calculate drift
        delta_t : int
            timescale to calculate diffusion

        Returns
        -------
        list
            [avgdriftX, avgdriftY, avgdiffX, avgdiffY, avgdiffXY, op_x, op_y]
        """

        op_x, self.op_x_range = self._order_parameter(x, inc_x, self.op_x_range)
        op_y, self.op_y_range = self._order_parameter(y, inc_y, self.op_y_range)

        driftX = self._drift(x, t_int, dt)
        driftY = self._drift(y, t_int, dt)

        diffusionX = self._diffusion(x, t_int, delta_t)
        diffusionY = self._diffusion(y, t_int, delta_t)

        diffusionXY = self._diffusion_xy(x, y, t_int, delta_t)

        avgdriftX = np.zeros((len(op_x), len(op_y)))
        avgdriftY = np.zeros((len(op_x), len(op_y)))
        avgdiffX = np.zeros((len(op_x), len(op_y)))
        avgdiffY = np.zeros((len(op_x), len(op_y)))
        avgdiffXY = np.zeros((len(op_x), len(op_y)))

        m = 0
        x_, y_ = x[0 : -max(dt, delta_t)], y[0 : -max(dt, delta_t)]
        for bin_x in op_y:
            n = 0
            for bin_y in op_x:
                i = np.where(
                    np.logical_and(
                        np.logical_and(x_ < (bin_x + inc_x), x_ >= bin_x),
                        np.logical_and(y_ < (bin_y + inc_y), y_ >= bin_y),
                    )
                )[0]
                avgdriftX[n, m] = np.nanmean(driftX[i])
                avgdriftY[n, m] = np.nanmean(driftY[i])
                avgdiffX[n, m] = np.nanmean(diffusionX[i])
                avgdiffY[n, m] = np.nanmean(diffusionY[i])
                avgdiffXY[n, m] = np.nanmean(diffusionXY[i])
                n = n + 1
            m = m + 1
        return [avgdriftX, avgdriftY, avgdiffX, avgdiffY, avgdiffXY, op_x, op_y]

    def __call__(self, X, t_int, dt, delta_t=1, inc=0.01, **kwargs):
        """
        Calcualtes drift, diffusion, average drift and avarage difussion.

        Parameters
        ----------
        X : list, array_like
                time series data
        t_int :float
                time step in time series
        dt : float
                analysis time step
        inc = 0.01 : float
                max increment for binning thae data

        returns
        -------
        diff : array_like
                diffusion in time series
        drift : array_like
                drift in time series
        avgdiff : array_like
                avarage diffusion
        avgdrift : array_like
                avaerage drift
        """
        self.__dict__.update(kwargs)
        return self._drift_and_diffusion(X, t_int, dt, inc)
