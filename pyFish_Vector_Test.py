#!/usr/bin/env python
# coding: utf-8

# # Import Modules

# In[17]:


import pyFish
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Load Sample data
# > Comment second line for pairwise data

# In[18]:


data = np.loadtxt('data/sample_vel_x_vel_y.csv', delimiter=',')
vel_x = data[:,0]
vel_y = data[:,1]
tint = 0.12


# # Initialize object with parameters

# In[19]:


sde = pyFish.Characterize(inc=0.01, inc_x=0.1, inc_y=0.1,t_lag=1000, simple_method=True, max_order=10, delta_t=1)


# # Analyse

# In[21]:


out = sde([vel_x, vel_y],t=None,dt='auto',t_int=tint)
out


# In[7]:


out.data()


# # View parameters

# In[9]:


print(out.parameters())


# # Visualize Output

# In[22]:


out.visualize()


# # Diagnostics graphs

# In[23]:


out.diagnostic()


# In[ ]:

# # Noise Characterstics
out.noise_characterstics()




