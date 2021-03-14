#!/usr/bin/env python
# coding: utf-8

# The script shows the effect of different coarsening methods on a toy example.
# 
# The code accompanies paper [Graph reduction with spectral and cut guarantees](http://www.jmlr.org/papers/volume20/18-680/18-680.pdf) by Andreas Loukas published at JMLR/2019 ([bibtex](http://www.jmlr.org/papers/v20/18-680.bib)).
# 
# This work was kindly supported by the Swiss National Science Foundation (grant number PZ00P2 179981).
# 
# 15 May 2020
# 
# [Andreas Loukas](https://andreasloukas.blog)
# 
# [![DOI](https://zenodo.org/badge/175851068.svg)](https://zenodo.org/badge/latestdoi/175851068)
# 
# Released under the Apache license 2.0

# In[1]:


get_ipython().system('pip install networkx')


# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import networkx as nx
import pygsp as gsp
gsp.plotting.BACKEND = 'matplotlib'


# In[3]:


from graph_coarsening.coarsening_utils import *
import graph_coarsening.graph_utils
import graph_coarsening.graph_lib 


# Load the graph

# In[4]:


N = 600 # number of nodes


# In[5]:


G = graph_coarsening.graph_lib.real(N, 'yeast')


# Coarsen it with different methods

# In[6]:


r       = 0.6 # coarsening ratio 
methods = ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 
           'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron']


# In[7]:


for method in methods: 

    C, Gc, Call, Gall = coarsen(G, r=r, method=method)
    plot_coarsening(Gall, Call, title=method, size=2);
    


# In[ ]:




