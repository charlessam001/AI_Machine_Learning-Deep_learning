#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


# In[2]:


x, y = make_regression(n_samples=100, n_features=1, noise=10)


# In[3]:


x.shape


# In[6]:


y = y.reshape(100, 1)


# In[7]:


y.shape


# In[8]:


plt.scatter(x, y)


# In[9]:


gx = np.hstack((x, np.ones(x.shape)))


# In[11]:


theta = np.random.randn(2, 1)


# In[12]:


theta


# In[13]:


theta.shape


# In[36]:


def model(gx, theta):
    return gx.dot(theta)


# In[37]:


plt.scatter(x, y)
plt.plot(x, model(gx, theta), c="r")


# In[38]:


def cost_f(gx, y, theta):
    m = len(y)
    return 1/(m*2) * np.sum((model(gx, theta) - y)**2)


# In[39]:


cost_f(gx, y, theta)


# In[40]:


def grad(gx, y, theta):
    m = len(y)
    return 1/m * gx.T.dot(model(gx, theta) - y)


# In[41]:


grad(gx, y, theta)


# In[42]:


def grad_d(gx, y, theta, l_rate, n_itr):
    for i in range(0, n_itr) :
        theta = theta - l_rate * grad(gx, y, theta)
        return theta


# In[53]:


theta_final = grad_d(gx, y, theta, l_rate=1.1, n_itr=1000)


# In[54]:


theta_final


# In[55]:


prediction = model(gx, theta_final)
plt.scatter(x, y)
plt.plot(x, prediction, c='r')


# In[ ]:




