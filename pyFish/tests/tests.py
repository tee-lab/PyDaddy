#!/usr/bin/env python
# coding: utf-8

import pyFish
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_sample_data(data_path):
	"""
	Load the sample distrubuted data

	data
	├── fish_data
	│   └── ectropus.csv
	└── model_data
		├── scalar
		│   ├── pairwise.csv
		│   └── ternary.csv
		└── vector
			├── pairwise.csv
			└── ternary.csv


	Each data file in pairwise, ternary and extras have two columns;
	first column is the timeseries data x, and the second one is the time stamp

	vector_data.csv also has two columns but contains the vector data x1 and x2 with missing time stamp. Use t_int=0.12.
	"""
	stream = pkg_resources.resource_stream('pyFish', data_path)
	try:
		return np.loadtxt(stream, delimiter=',')
	except:
		return np.loadtxt(stream)


def scalar_test(data_path='data/model_data/scalar/ternary.csv', show=False):
	data = load_sample_data(data_path)
	X = data[:, 0]
	t = data[:, 1]
	t_int = t[-1] / len(t)

	# # Analyse
	out = pyFish.Characterize(data=[X], t=t)

	drift, diff, avgdrift, avgdiff, op = out.data()
	len(diff), len(drift), len(avgdrift), len(avgdiff), len(op)

	# # View parameters
	out.parameters(save=True)

	# # Visualize Output
	out.visualize(show=show, save=True)

	# # Diagnostics graphs
	out.diagnostic(show=show, save=True)

	# # Noise Characterstics
	out.noise_characterstics(show=show, save=True)

	# #Save data
	out.save_data()

	return out


def vector_test(data_path='data/model_data/vector/ternary.csv', show=False):
	data = load_sample_data(data_path)
	vel_x = data[:, 0]
	vel_y = data[:, 1]
	tint = 0.12

	# # Initialize object with parameters
	out = pyFish.Characterize([vel_x, vel_y], t=0.12)

	# # Analyse
	out.data()

	# # View parameters
	out.parameters(save=True)

	# # Visualize Output
	out.visualize(show=show, save=True)

	# # Diagnostics graphs
	out.diagnostic(show=show, save=True)

	# # Noise analysis
	out.noise_characterstics(show=show, save=True)

	# # Save data
	out.save_data()

	return out
