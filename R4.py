#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
def DampedPendulum(t, y, b=0.2, omega0=1):
    x,v = y
    dxdt = v
    dvdt = -b*v-(omega0**2)*x
    dydt = np.array([dxdt, dvdt])
    return dydt
# Define the two derivatives
# note that dydt is now an array that holds both of the derivatives
x0 = 0 # initial position
v0 = 1 # initial velocity
y0 = (x0,v0) # initial state
t0 = 0 # initial time
tf = 25*np.pi # final time
n = 1001 # Number of points at which output will be evaluated
# Note: this does not mean the integrator will take only n steps
# Scipy will take more steps if required to control the error in the solution
t = np.linspace(t0, tf, n) # Points at which output will be evaluated
result = integrate.solve_ivp(fun = DampedPendulum, # The function defining the derivative
t_span = (t0, tf), # Initial and final times
y0 = y0, # Initial state
method = "RK45", # Integration method
t_eval = t) # Time points for result to be defined at
# Read the solution and time from the result arrat returned by Scipy
x,v = result.y
t = result.t
# plot position ad velocity as a function of time.
plt.plot(t,x, label=r"$x(t)$")
plt.plot(t,v, label=r"$v(t)$")
plt.legend(loc=1)

plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.savefig('r4.svg', bbox_inches='tight')


plt.show()


# In[21]:


plt.plot(v,x,'k')
plt.axis('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$v$")
plt.savefig('r4_2.svg', bbox_inches='tight')

