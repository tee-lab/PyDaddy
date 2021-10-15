""" Code for fitting polynomials to data. """

import warnings
import numpy as np
from sklearn.linear_model import ridge_regression


class PolyFit:
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
            x (np.array): Independent variable
            y (np.array): Dependent variable
            weights (np.array): Sample weights for regression.
                If None (default), simple unweighted ridge regression will be performed.
        """

        nan_idx = np.argwhere(np.isnan(y))
        x = np.delete(x, nan_idx)
        y = np.delete(y, nan_idx)
        if weights is not None:
            weights = np.delete(weights, nan_idx)

        maxiter = self.max_degree

        if self.library:
            dictionary = np.array([f(x) for f in self.library])
        else:  # Default polynomial dictionary
            dictionary = np.array([x ** d for d in range(self.max_degree + 1)]).T

        coeffs = np.zeros(self.max_degree + 1)
        keep = np.ones_like(coeffs, dtype=np.bool)
        for it in range(maxiter):
            if np.sum(keep) == 0:
                warnings.warn('Sparsity threshold is too big, eliminated all parameters.')
                break
            # coeffs_, _, _, _ = np.linalg.lstsq(dictionary[:, keep], y_)
            coeffs_ = ridge_regression(dictionary[:, keep], y, alpha=self.alpha, sample_weight=weights)
            coeffs[keep] = coeffs_
            keep = (np.abs(coeffs) > self.threshold)
            coeffs[~keep] = 0

        return np.poly1d(np.flipud(coeffs))

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

    @staticmethod
    def _get_bic(p, x, y):
        """ Compute the BIC for a fitted polynomial with given data x, y. """

        dof = np.count_nonzero(p)  # Degrees of freedom
        n_samples = len(x)  # Number of samples
        mse = np.mean((y - p(x)) ** 2) / np.var(y)  # Normalized mean-squared error
        bic = np.log(n_samples) * dof + n_samples * np.log(mse)
        # bic = 2 * dof + n_samples * np.log(mse)
        return bic
