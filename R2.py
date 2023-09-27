#!/usr/bin/env python
# coding: utf-8

# In[2]:



import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sys
from IPython.display import display
import math

def Nonlinear1(v,r,i,l, t):
    didt = (v-r*i)/l
    return(didt)
#Define the derivative function
i0 = np.array([0]) # initial state at t = 0

v=10
r=50
l=100


t0 = 0 # initial time
tf = 20 # final time

n = 10 # Number of points at which output will be evaluated

t = np.linspace(t0, tf, n)


result = integrate.solve_ivp(fun=lambda t, y: Nonlinear1(v, r, y, l,t), t_span=(t0, tf), y0=i0, method="RK45", t_eval=t)
t_span = (t0, tf), # Initial and final times
y0 =  i0 # Initial state
method = "RK23", # Integration method
t_eval = t # Time points for result to be reported


def calc(t):
    return ((v/r) * (1 - math.exp((-r * t) / l)))

value = [calc(t) for t in t]


        # Read the solution and time from the array returned by Scipy
y = result.y[0]
t = result.t

#plot the solution
plt.plot(t,value, label=f'Solution', linewidth=3, alpha=1)
plt.plot(t,y, label=f'Approximation', linewidth=2.5, alpha=1, linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('Current (amp)')

plt.legend()
plt.savefig("R2_10.svg")
plt.show()


# In[31]:


y


# In[4]:


def calc(t):
    return(0.2*(1-(math.exp(-(50*t)/100))))


value = [calc(t) for t in t]


# In[21]:


math.exp(-1)

