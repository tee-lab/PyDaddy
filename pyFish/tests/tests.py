#!/usr/bin/env python
# coding: utf-8

# # Import Modules

# In[1]:


import pyFish
import pkg_resources
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


	# # Initialize object with parameters
	# ### Default parameter values:
	# 			dt='auto', 
	# 			delta_t =1,
	# 			t_int=None, 
	# 			t_lag=1000, 
	# 			inc=0.01, 
	# 			inc_x=0.1, 
	# 			inc_y=0.1,
	# 			max_order=10,
	# 			fft = True,
	# 			drift_order = None,
	# 			diff_order = None,
	# 			order_metric = "R2_adj",
	# 			simple_method = True,
	# 			n_trials = 1,


def load_data(data_path):
	stream = pkg_resources.resource_stream('pyFish', data_path)
	return np.loadtxt(stream, delimiter=',')

def scalar_test(data_path='data/pairwise/N30.csv'):
	data = load_data(data_path)
	X = data[:,0]
	t = data[:,1]
	t_int = t[-1]/len(t)


	# # Analyse
	out = pyFish.Characterize(data=[X],t=t)

	drift, diff, avgdrift, avgdiff, op = out.data()
	len(diff), len(drift),len(avgdrift), len(avgdiff), len(op)

	# # View parameters
	out.parameters(save=True)

	# # Visualize Output
	out.visualize(show=True, save=True)

	# # Diagnostics graphs
	out.diagnostic(show=True, save=True)

	# # Noise Characterstics
	out.noise_characterstics(show=True, save=True)

	# #Save data
	out.save_data()



def vector_test(data_path='data/vector/vector_data.csv'):
	data = load_data(data_path)
	vel_x = data[:,0]
	vel_y = data[:,1]
	tint = 0.12

	# # Initialize object with parameters
	out = pyFish.Characterize([vel_x, vel_y], t=None, t_int=tint)

	# # Analyse
	out.data()

	# # View parameters
	out.parameters(save=True)

	# # Visualize Output
	out.visualize(show=True, save=True)

	# # 2D Slice
	out.slices_2d(show=True, save=True)

	# # Diagnostics graphs
	out.diagnostic(show=True, save=True)

	# # Noise analysis
	out.noise_characterstics(show=True, save=True)

	# # Save data
	out.save_data()

