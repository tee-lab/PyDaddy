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

# In[2]:


#data = np.loadtxt('data/pairwise/TimeSeries/TimeSeries_pairwise_N_50_.csv', delimiter=',')
data = np.loadtxt('data/ternary/TimeSeries/TimeSeries_ternary_N_50_.csv', delimiter=',')
X = data[:,0]
t = data[:,1]
t_int = t[-1]/len(t)


# # Initialize object with parameters

# In[ ]:


sde = pyFish.Characterize(inc=0.01, t_lag=1000, simple_method=True, max_order=10, delta_t=1)


# # Analyse

# In[4]:


out = sde([X],t,dt='auto')
out


# In[5]:


drift, diff, avgdrift, avgdiff, op = out.data()
len(diff), len(drift),len(avgdrift), len(avgdiff), len(op)


# # View parameters

# In[6]:


params = out.parameters()
print(params)


# # Visualize Output

# In[7]:


out.visualize()


# # Diagnostics graphs

# In[8]:


out.diagnostic()


# In[ ]:

# # Noise Characterstics
out.noise_characterstics()



