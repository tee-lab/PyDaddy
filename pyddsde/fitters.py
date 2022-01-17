""" Code for fitting polynomials to data. """

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ridge_regression
from sklearn.model_selection import KFold


class Poly2D:
    """ A rudimentary 2D polynomial class. Returns polynomial objects that can be called or pretty-printed. """

    def __init__(self, coeffs, degree):
        assert len(coeffs) == (degree + 1) * (degree + 2) / 2, \
            f'For degree {degree}, number of coefficients mut be {(degree + 1) * (degree + 2) / 2}'
        self.coeffs = np.array(coeffs)
        self.degree = degree

    def __call__(self, x, y):
        # x = np.array(x)
        # if x.ndim == 1:
        #     x, y = x[0], x[1]
        # elif x.ndim == 2:
        #     x, y = x[:, 0], x[:, 1]
        # else:
        #     raise ValueError('Incompatible dimensions for Poly2D call. Poly2D objects can be called with a tuple, or a 1D or 2D numpy array.')

        terms = np.array([(x ** n) * (y ** m)
                          for m in range(self.degree + 1)
                          for n in range(self.degree - m + 1)])
        terms_with_coeffs = self.coeffs * np.moveaxis(terms, 0, -1)
        return terms_with_coeffs.sum(axis=-1)

    def __array__(self):
        return self.coeffs

    def __str__(self):
        def term(m, n):
            if m == 0: xterm = ''
            elif m == 1: xterm = 'x'
            else: xterm = f'x^{m}'

            if n == 0: yterm = ''
            elif n == 1: yterm = 'y'
            else: yterm = f'y^{n}'

            return xterm + yterm

        terms = [term(n, m) for m in range(self.degree + 1) for n in range(self.degree - m + 1)]
        terms_with_coeffs = [f'{c}{t}' for (c, t) in zip(self.coeffs, terms) if c != 0]
        if terms_with_coeffs:
            return ' + '.join(terms_with_coeffs)
        else:
            return '0'


class PolyFitBase:
    """ Fits polynomial to estimated drift and diffusion functions with sparse regression. """

    def __init__(self, threshold=0, max_degree=5, alpha=0, library=None):
        """ Initialize the PolyFit object with appropriate regression parameters.
        Parameters:
            threshold: Sparsity threshold (can be optimized using model_selection())
            max_degree: Maximum degree for polynomial fits
            alpha: Regularization parameter for ridge regression
            library: Library of candidate functions (optional). Should be a list of callables.
        """
        self.max_degree = max_degree
        self.threshold = threshold
        self.alpha = alpha
        self.library = library

    def fit(self, x, y, weights=None):
        """ Fit a polynomial using sparse regression using STLSQ (Sequentially thresholded least-squares)
        Parameters:
            x (np.array or list): Independent variable. Could either be an array (for 1D case) or
                a list of two arrays (for 2D case).
            y (np.array): Dependent variable
            weights (np.array): Sample weights for regression.
                If None (default), simple unweighted ridge regression will be performed.
        Returns:
            np.poly1d object for 1D case, Poly2D object for 2D case.
        """

        if self.library:
            dictionary = np.vstack([f(x) for f in self.library]).T
            coeffs = np.zeros(len(self.library))
            keep = np.ones_like(coeffs, dtype=np.bool)
            ispoly = False
        else:  # Default polynomial dictionary
            dictionary = self._get_poly_dictionary(x)
            coeffs = self._get_coeffs()
            keep = np.ones_like(coeffs, dtype=np.bool)
            ispoly = True

        maxiter = dictionary.shape[1]

        for it in range(maxiter):
            if np.sum(keep) == 0:
                warnings.warn('Sparsity threshold is too big, eliminated all parameters.')
                break
            coeffs_ = ridge_regression(dictionary[:, keep], y, alpha=self.alpha, sample_weight=weights)
            # print(f'coeffs: {coeffs_}')
            coeffs[keep] = coeffs_
            keep = (np.abs(coeffs) > self.threshold)
            coeffs[~keep] = 0
        if ispoly:
            return self._get_callable_poly(coeffs)
        else:
            return coeffs

    def fit_ssr(self, x, y, weights=None, plot=True, model_selection='cv'):
        """ Fit a polynomial using sparse regression using STLSQ (Sequentially thresholded least-squares)
        Parameters:
            x (np.array or list): Independent variable. Could either be an array (for 1D case) or
                a list of two arrays (for 2D case).
            y (np.array): Dependent variable
            weights (np.array): Sample weights for regression.
                If None (default), simple unweighted ridge regression will be performed.
            model_selection: ('bic' or 'cv') Method for model-selection
        Returns:
            np.poly1d object for 1D case, Poly2D object for 2D case.
        """

        assert model_selection in ['bic', 'cv'], "Parameter 'model_selection' should be 'bic' or 'cv'."

        if self.library:
            dictionary = np.vstack([f(x) for f in self.library]).T
            coeffs = np.zeros(len(self.library))
            keep = np.ones_like(coeffs, dtype=np.bool)
            ispoly = False
        else:  # Default polynomial dictionary
            dictionary = self._get_poly_dictionary(x)
            coeffs = self._get_coeffs()
            keep = np.ones_like(coeffs, dtype=np.bool)
            ispoly = True

        metrics = []
        best_metric = np.inf
        for it in range(len(coeffs)):
            coeffs_ = ridge_regression(dictionary[:, keep], y, alpha=self.alpha, sample_weight=weights)
            coeffs[keep] = coeffs_
            if model_selection == 'cv':
                metric = self._get_cv_error_2(keep, x, y, dictionary, folds=5)
            else:
                metric = self._get_bic(coeffs, x, y)
            print(f'Iteration: {it}, Best Metric: {best_metric}, Metric: {metric}, Coeffs: {coeffs}')
            metrics.append(metric)

            if metric <= best_metric + 0.05:
                best_model = coeffs.copy()
            if metric <= best_metric:
                best_metric = metric

            keep[np.abs(coeffs) == np.min(np.abs(coeffs_))] = False
            coeffs[~keep] = 0

            # print(f'Best BIC: {best_bic}, Best coeffs: {best_model}')

        if plot:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(metrics)
            ylabel = {'cv': 'CV Error', 'bic': 'BIC'}
            ax.set(xlabel='Iteration', ylabel=ylabel[model_selection])
            plt.show()

        if ispoly:
            return self._get_callable_poly(best_model)
        else:
            return best_model  # FIXME Return callable function?

    def model_selection(self, thresholds, x, y, weights=None, method='cv', plot=False):
        """ Automatically choose the best threshold using BIC.
        Parameters:
            thresholds: List of thresholds to search over.
            x, y: Data to be used for parameter tuning.
            weights: (Optional) weights for fitting.
            method: {'bic', 'cv'} The metric used for model selection
            plot: If true, plot the model selection curves
        """

        assert method in ['bic', 'cv'], "Parameter 'model_selection' should be 'bic' or 'cv'."
        metric_name = {'bic': 'BIC', 'cv': 'CV Error'}

        # print('Finding best threshold for polynomial fit ...')
        # best_thresh = 0
        # best_metric = np.inf

        metrics = []
        nparams = []
        for thresh in thresholds:
            self.threshold = thresh
            p = self.fit(x, y, weights)
            if method == 'bic':
                metric = self._get_bic(p, x, y)
            else:
                metric = self._get_cv_error(x, y, folds=5)

            metrics.append(metric)
            nparams.append(np.count_nonzero(p))
            # print(f'poly: {p}')
            # print(f'degree = {degree}, threshold: {thresh}, BIC: {bic}')
            # print(f'threshold: {thresh:.4f}, {metric_name[method]}: {metric:.4f}, poly: {p}')
            # if metric <= best_metric:
            #     best_metric = metric
            #     best_thresh = thresh

        metrics = np.array(metrics)
        errordelta = metrics[1:] - metrics[:-1]
        best_thresh = thresholds[:-1][np.argmax(errordelta)]
        if plot:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].plot(thresholds, metrics, '.-')
            ax[0].set(xlabel='Sparsity Threshold', ylabel=metric_name[method])

            metrics = np.array(metrics)
            errordelta = metrics[1:] - metrics[:-1]
            ax[1].plot(thresholds[:-1], errordelta, '.-')
            ax[1].set(xlabel='Sparsity Threshold', ylabel=f'Change in {metric_name[method]}')

            ax[2].plot(thresholds, nparams, '.-')
            ax[2].set(xlabel='Sparsity Threshold', ylabel='Nonzero Coefficients')
            plt.tight_layout()
            plt.show()
        # print(f'Model selection complete. Chosen threshold = {best_thresh}')
        self.threshold = best_thresh

    def tune_and_fit(self, x, y, thresholds=None, steps=20):
        """
        Args:
            x, y: Data to fit
            thresholds: List of thresholds to try, will be automatically chosen if None
            steps: When auto-choosing thesholds, the number of steps to take in the threshold range.
        """

        if thresholds is None:
            self.threshold = 0
            p = np.array(self.fit(x, y))
            thresholds = np.linspace(0, np.max(np.abs(p)), steps, endpoint=False)

        self.model_selection(thresholds=thresholds, x=x, y=y, plot=False)
        return self.fit(x, y)

    def _get_cv_error_2(self, keep, x, y, dictionary, folds):
        """
        Parameters:
            keep: Bit-vector denoting which coefficients to keep
        """
        kf = KFold(n_splits=folds, shuffle=False)
        cv_errors = []
        coeffs = np.zeros_like(keep)
        for train, test in kf.split(x, y):
            # Fit on train data
            coeffs_ = ridge_regression(dictionary[train][:, keep], y[train], alpha=self.alpha)
            coeffs[keep] = coeffs_
            coeffs[~keep] = 0

            # Evaluate on held-out test-data
            mse = np.mean((y[test] - self._evaluate(coeffs, x[test])) ** 2)
            cv_errors.append(mse)

        return np.mean(cv_errors)

    def _get_cv_error(self, x, y, folds):
        kf = KFold(n_splits=folds, shuffle=False)
        cv_errors = []
        for train, test in kf.split(x, y):
            p = self.fit(x[train], y[train])
            mse = np.mean((y[test] - self._evaluate(p, x[test])) ** 2)
            cv_errors.append(mse)

        return np.mean(cv_errors)

    def _get_bic(self, p, x, y):
        """ Compute the BIC for a fitted polynomial with given data x, y. """

        dof = np.count_nonzero(p)  # Degrees of freedom
        n_samples = len(y)
        mse = np.mean((y - self._evaluate(p, x)) ** 2)#np.mean(y ** 2)  # Normalized mean-squared error
        # bic = np.log(n_samples) * dof + n_samples * np.log(mse / n_samples)
        # print(f'dof: {dof}, n_samples: {n_samples}, mse: {mse}, bic: {bic}')
        bic = 2 * dof + n_samples * np.log(mse / n_samples)
        return bic

    def _get_poly_dictionary(self, x):
        raise NotImplementedError

    def _get_callable_poly(self, coeffs):
        raise NotImplementedError

    def _get_coeffs(self):
        raise NotImplementedError

    # def _get_cv_splits(self, x, y):
    #     raise NotImplementedError

    def _evaluate(self, c, x):
        if self.library:  # Fitting with custom library
            # In this case, c is an array of coefficients.
            dictionary = np.vstack([f(x) for f in self.library]).T
            return np.sum(c * dictionary, axis=1)
        else:  # Fitting with default polynomial library
            # In this case, c is a callable polynomial.
            # c = self._get_callable_poly(c)
            return self._evaluate_poly(c, x)

    def _evaluate_poly(self, c, x):
        return NotImplementedError


class PolyFit1D(PolyFitBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_poly_dictionary(self, x):
        return np.array([x ** d for d in range(self.max_degree + 1)]).T

    def _get_callable_poly(self, coeffs):
        """ Construct a callable polynomial from a given coefficient array. """
        return np.poly1d(np.flipud(coeffs))

    def _get_coeffs(self):
        return np.zeros(self.max_degree + 1)

    def _evaluate_poly(self, c, x):
        return c(x)


class PolyFit2D(PolyFitBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_poly_dictionary(self, x):
        x, y = x[:, 0], x[:, 1]
        return np.array([(x ** n) * (y ** m)
                         for m in range(self.max_degree + 1)
                         for n in range(self.max_degree - m + 1)]).T

    def _get_callable_poly(self, coeffs):
        return Poly2D(coeffs, self.max_degree)

    def _get_coeffs(self):
        return np.zeros(int((self.max_degree + 1) * (self.max_degree + 2) / 2))

    def _evaluate_poly(self, c, x):
        return c(x[:, 0], x[:, 1])
