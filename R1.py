#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
def Nonlinear1(a,b,t,y):
    dydt = -a*y**3 + b*np.sin(t)
    return(dydt)
#Define the derivative function
y0 = np.array([0]) # initial state at t = 0

a_values = [1, 9]  
b_values = [2, 4]  

t0 = 0 # initial time
tf = 20 # final time

n = 101 # Number of points at which output will be evaluated
# Note: this does not mean the integrator will take only n steps SciPy
# will control this to control the error in the solution
t = np.linspace(t0, tf, n) #linearly spaced time intervals
#Call the RK integrator and return the solution in the array "result"

for a in a_values:
    for b in b_values:
        result = integrate.solve_ivp(fun=lambda t, y: Nonlinear1(a, b, t, y), t_span=(t0, tf), y0=y0) # The function defining the derivative
        t_span = (t0, tf), # Initial and final times
        y0 = y0  # Initial state
        method = "RK45", # Integration method
        t_eval = t # Time points for result to be reported
        # Read the solution and time from the array returned by Scipy
        y = result.y[0]
        t = result.t

#plot the solution
        plt.plot(t,y, label=f'a={a}, b={b}')
    

plt.xlabel('Time (s)')
plt.ylabel('x')

plt.legend()
plt.savefig("R1.svg")
plt.show()

