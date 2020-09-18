#!/usr/bin/env python
# coding: utf-8

# # Import Modules

# In[1]:


import pyFish
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# # Load Sample data

# In[3]:


data = np.loadtxt('data/sample_vel_x_vel_y.csv', delimiter=',')
vel_x = data[:,0]
vel_y = data[:,1]
tint = 0.12


# # Initialize object with parameters

# In[4]:


out = pyFish.Characterize([vel_x, vel_y], t=None, t_int=tint)


# # Analyse

# In[5]:


out.data()


# # View parameters

# In[6]:


out.parameters(save=True)


# # Visualize Output

# In[7]:


out.visualize(show=False, save=True)


# # 2D Slice

# In[8]:


out.slices_2d(show=False, save=True)


# # Diagnostics graphs

# In[9]:


out.diagnostic(show=False, save=True)


# In[10]:


out.noise_characterstics(show=False, save=True)


# In[11]:


out.save_data()


# In[ ]:




