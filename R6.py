#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math
def DampedPendulum(t, y, b=0.1, omega0=1, A=1, omegad=0.9):
    x,v = y
    dxdt = v
    dvdt = -b*v-(omega0**2)*x-A*math.sin(omegad*t)
    dydt = np.array([dxdt, dvdt])
    return dydt
# Define the two derivatives
# note that dydt is now an array that holds both of the derivatives
x0 = 0 # initial position
v0 = 1 # initial velocity
y0 = (x0,v0) # initial state
t0 = 0 # initial time
tf = 10*np.pi # final time
n = 1001 # Number of points at which output will be evaluated
b=0.1
omega0=1
# Note: this does not mean the integrator will take only n steps
# Scipy will take more steps if required to control the error in the solution
t = np.linspace(t0, tf, n) # Points at which output will be evaluated
result = integrate.solve_ivp(fun = lambda t, y, : DampedPendulum(t, y, b, omega0), # The function defining the derivative
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


plt.savefig('r6_01.svg', bbox_inches='tight')
plt.show()


# In[3]:


plt.plot(v,x, 'k')
plt.axis('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$v$")
plt.savefig('r6_1phase.svg', bbox_inches='tight')


# In[24]:





# In[7]:


#Loop through list of three driving frequencies (100%, 90%, 50% of omega0)
def DampedPendulum(t, y, b=0.1, omega0=1, A=1, omegad=1):
    x,v = y
    dxdt = v
    dvdt = -b*v-(omega0**2)*x-A*math.sin(omegad*t)
    dydt = np.array([dxdt, dvdt])
    return dydt

omega0=1
    
for i in (1*omega0, 0.9*omega0, 0.5*omega0):
    # Define the anonymous function, including the changing omegad
    lfun = lambda t, y, omegad=i, omega0=omega0, b=0.1, A=1: DampedPendulum(t, y, b, omega0, A, omegad)

    result = integrate.solve_ivp(fun = lfun,
    t_span = (0, tf),
    y0 = y0 ,
    method = "RK45",
    t_eval = t )
    # Store result of this run in variables t, x, v
    t = result.t
    x,v = result.y
    # Plot the result x(t) for this run, lable it with omegad as well
    plt.plot(t, x, label='$x(t): \omega_d =${}'.format(i))
# End of loop, continue with next omegad
# Out of the loop
# Save and show plot

plt.legend() # Make the plot labels visible
plt.savefig('Oscillator-driven-multi.svg', bbox_inches='tight')
plt.show()

