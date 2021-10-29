""" Code for fitting polynomials to data. """

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ridge_regression


class Poly2D:
    """ A rudimentary 2D polynomial class. Returns polynomial objects that can be called or pretty-printed. """

    def __init__(self, coeffs, degree):
        assert len(coeffs) == (degree + 1) * (degree + 2) / 2, \
            f'For degree {degree}, number of coefficients mut be {(degree + 1) * (degree + 2) / 2}'
        self.coeffs = np.array(coeffs)
        self.degree = degree

    def __call__(self, x):
        x, y = x
        terms = np.array([(x ** n) * (y ** m)
                          for m in range(self.degree + 1)
                          for n in range(self.degree - m + 1)])
        try:
            # The following statement gives a ValueError (broadcast dimensions error) if
            #  being called with np.arrays as arguments. In this case, we need an appropriate
            #  reshaping -- see except clause.
            terms_with_coeffs = self.coeffs * terms
        except ValueError:
            terms_with_coeffs = self.coeffs[:, None] * terms

        return terms_with_coeffs.sum(axis=0)

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

        maxiter = self.max_degree

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

    def model_selection(self, thresholds, x, y, weights=None, plot=False):
        """ Automatically choose the best threshold using BIC.
        Parameters:
            thresholds: List of thresholds to search over.
            x, y: Data to be used for parameter tuning.
            weights: (Optional) weights for fitting.
        """

        print('Finding best threshold for polynomial fit ...')
        best_thresh = 0
        best_bic = np.inf

        bics = []
        nparams = []
        for thresh in thresholds:
            self.threshold = thresh
            p = self.fit(x, y, weights)
            bic = self._get_bic(p, x, y)
            bics.append(bic)
            nparams.append(np.count_nonzero(p))
            # print(f'poly: {p}')
            # print(f'degree = {degree}, threshold: {thresh}, BIC: {bic}')
            if bic <= best_bic:
                best_bic = bic
                best_thresh = thresh
        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(16, 7))
            ax[0].plot(thresholds, bics)
            ax[0].set(xlabel='Sparsity Threshold', ylabel='BIC')
            ax[1].plot(thresholds, nparams)
            ax[1].set(xlabel='Sparsity Threshold', ylabel='Nonzero Coefficients')
            plt.show()
        print(f'Model selection complete. Chosen threshold = {best_thresh}')
        self.threshold = best_thresh

    def _get_bic(self, p, x, y):
        """ Compute the BIC for a fitted polynomial with given data x, y. """

        dof = np.count_nonzero(p)  # Degrees of freedom
        n_samples = len(y)
        mse = np.mean((y - self._evaluate(p, x)) ** 2) / np.var(y)  # Normalized mean-squared error
        bic = np.log(n_samples) * dof + n_samples * np.log(mse)
        # print(f'dof: {dof}, n_samples: {n_samples}, mse: {mse}, bic: {bic}')
        # bic = 2 * dof + n_samples * np.log(mse)
        return bic

    def _get_poly_dictionary(self, x):
        raise NotImplementedError

    def _get_callable_poly(self, coeffs):
        raise NotImplementedError

    def _get_coeffs(self):
        raise NotImplementedError

    def _evaluate(self, c, x):
        if self.library:  # Fitting with custom library
            # In this case, c is an array of coefficients.
            dictionary = np.vstack([f(x) for f in self.library]).T
            return np.sum(c * dictionary, axis=1)
        else:  # Fitting with default polynomial library
            # In this case, c is a callable polynomial.
            return c(x)


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


class PolyFit2D(PolyFitBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_poly_dictionary(self, x):
        x, y = x
        return np.array([(x ** n) * (y ** m)
                         for m in range(self.max_degree + 1)
                         for n in range(self.max_degree - m + 1)]).T

    def _get_callable_poly(self, coeffs):
        return Poly2D(coeffs, self.max_degree)

    def _get_coeffs(self):
        return np.zeros(int((self.max_degree + 1) * (self.max_degree + 2) / 2))
