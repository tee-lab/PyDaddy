import unittest

import numpy as np
from scipy.io import loadmat
import pydaddy
from pydaddy.fitters import Poly1D, Poly2D


# FIXME: Ensure test-case functions are stable.
class TestSimulateVector(unittest.TestCase):

    def setUp(self):
        data, t = pydaddy.load_sample_dataset('model-data-vector-ternary')
        self.ddsde = pydaddy.Characterize(data, t=1, Dt=1, bins=20, show_summary=False)

    def test_corner_cases(self):
        with self.assertRaises(AssertionError,
                               msg='Should have raised AssertionError when all functions missing.'):
            self.ddsde.simulate(t_int=0.12, timepoints=1000)

        self.ddsde._ddsde.F1 = Poly2D(degree=2, coeffs=[0, 0, 1, 3, 0, 0])
        self.ddsde._ddsde.G11 = Poly2D(degree=2, coeffs=[0, 0, 1, -3, 0, 0])

        with self.assertRaises(AssertionError,
                               msg='Should have raised AssertionError when some functions missing.'):
            self.ddsde.simulate(t_int=0.12, timepoints=1000)

        self.ddsde._ddsde.G12 = Poly2D(degree=2, coeffs=[0, 0, 1, 0, 0, 0])

        with self.assertRaises(AssertionError,
                               msg='Should have raised AssertionError when some functions missing.'):
            self.ddsde.simulate(t_int=0.12, timepoints=1000)

        print('ddsde.simulate() vector corner-case tests passed.')

    def test_simulate(self):
        self.ddsde._ddsde.F1 = Poly2D(degree=2, coeffs=[0, 0, 1, 3, 0, 0])
        self.ddsde._ddsde.F2 = Poly2D(degree=2, coeffs=[0, 0, 1, 3, 0, 0])
        self.ddsde._ddsde.G11 = Poly2D(degree=2, coeffs=[0, 0, 1, 3, 0, 0])
        self.ddsde._ddsde.G22 = Poly2D(degree=2, coeffs=[0, 0, 1, 3, 0, 0])
        self.ddsde._ddsde.G12 = Poly2D(degree=2, coeffs=[0, 0, 0, 0, 0, 0])

        self.ddsde.simulate(t_int=0.12, timepoints=1000)

        print('ddsde.simulate() vector simulation test passed.')


class TestSimulateScalar(unittest.TestCase):

    def setUp(self):
        data, t = pydaddy.load_sample_dataset('model-data-scalar-ternary')
        self.ddsde = pydaddy.Characterize(data, t=1, Dt=1, bins=20, show_summary=False)

    def test_corner_cases(self):
        with self.assertRaises(AssertionError,
                               msg='Should have raised AssertionError when all functions missing.'):
            self.ddsde.simulate(t_int=0.12, timepoints=1000)

        self.ddsde._ddsde.F = Poly1D(degree=3, coeffs=[1, 0, 0, 1])

        with self.assertRaises(AssertionError,
                               msg='Should have raised AssertionError when F is missing.'):
            self.ddsde.simulate(t_int=0.12, timepoints=1000)

        print('ddsde.simulate() scalar corner-case tests passed.')

    def test_simulate(self):
        self.ddsde._ddsde.F = Poly1D(degree=3, coeffs=[1, 0, 0, 1])
        self.ddsde._ddsde.G = Poly1D(degree=4, coeffs=[0, 0, 1, 0, 1])

        self.ddsde.simulate(t_int=0.12, timepoints=1000)
        print('ddsde.simulate() scalar simulation test passed.')


if __name__ == '__main__':
    unittest.main()

