# -*- coding: utf-8 -*-
# flake8: noqa: F401
# noreorder
"""
pyFish : a package to analyse timeseries data using stocastic differential equations
Copyright (C) 2020 Theoritical Evolution and Ecology Lab (TEE Lab), IISc, Bengaluru

This program is free software: you can redistribute it and/or modify it under the terms 
of the GNU General Public License as published by the Free Software Foundation, either 
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. 
If not, see <https://www.gnu.org/licenses/>.
"""

__title__ = "pyFish"
__author__ = "Ashwin Karichannavar, (e-mail: ashwinkk.23@gmail.com)"
__license__ = "GNU General Public License v3.0"
__copyright__ = "Copyright (C) 2020 Theoritical Evolution and Ecology Lab (TEE Lab), IISc, Bengaluru"
__version__ = '1.17'

from pyFish.sde import SDE
from pyFish.analysis import underlying_noise
from pyFish.analysis import AutoCorrelation
from pyFish.analysis import gaussian_test
from pyFish.preprocessing import preprocessing
from pyFish.metrics import metrics
from pyFish.output import output
from pyFish.visualize import visualize
from pyFish.__main__ import Characterize

def isnotebook():
	try:
		shell = get_ipython().__class__.__name__
		print(shell)
		if shell == 'ZMQInteractiveShell':
			return True   # Jupyter notebook or qtconsole
		elif shell == 'TerminalInteractiveShell':
			return False  # Terminal running IPython
		else:
			return False  # Other type (?)
	except NameError:
		return False      # Probably standard Python interpreter

if not isnotebook():
	import matplotlib
	matplotlib.use('Qt5Agg')
	matplotlib.rcParams['font.size'] = 18
	#matplotlib.style.use('seaborn')