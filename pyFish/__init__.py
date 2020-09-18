# -*- coding: utf-8 -*-
# flake8: noqa: F401
# noreorder

"""
pyFish : a package to analyse timeseries data using stocastic differential equations
"""

__title__ = "pyFish"
__author__ = "Ashwin Karichannavar, (e-mail: ashwinkk.23@gmail.com)"
__license__ = ""
__copyright__ = ""

from pyFish.sde import SDE
from pyFish.analysis import underlying_noise
from pyFish.analysis import AutoCorrelation
from pyFish.analysis import gaussian_test
from pyFish.preprocessing import preprocessing
from pyFish.metrics import metrics
from pyFish.output import output
from pyFish.__main__ import Characterize