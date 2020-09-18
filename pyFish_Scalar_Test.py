#!/usr/bin/env python
# coding: utf-8

# # Import Modules

# In[1]:


import pyFish
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Load Sample data
# > Comment second line for pairwise data
# 
# > Uncomment 3rd line for vegetation data (Shuaib's data)

# In[2]:


data = np.loadtxt('data/pairwise/TimeSeries/TimeSeries_pairwise_N_50_.csv', delimiter=',')
data = np.loadtxt('data/ternary/TimeSeries/TimeSeries_ternary_N_50_.csv', delimiter=',')
#data = np.loadtxt('data/extras/data.txt')
X = data[:,0]
t = data[:,1]
t_int = t[-1]/len(t)


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

# # Analyse

# In[3]:


out = pyFish.Characterize(data=[X],t=t)
out


# In[4]:


drift, diff, avgdrift, avgdiff, op = out.data()
len(diff), len(drift),len(avgdrift), len(avgdiff), len(op)


# # View parameters

# In[5]:


out.parameters(save=True)


# # Visualize Output

# In[6]:


out.visualize(show=False, save=True)


# # Diagnostics graphs

# In[7]:


out.diagnostic(show=False, save=True)


# # Noise Characterstics

# In[8]:


out.noise_characterstics(show=False, save=True)


# In[9]:


out.save_data()


# In[ ]:




