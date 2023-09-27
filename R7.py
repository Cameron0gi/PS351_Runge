#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math
def DampedPendulum(t, y, b=0.2, omega0=1, A=1, omegad=0.9):
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
tf = 25*np.pi # final time
n = 1001 # Number of points at which output will be evaluated
b=0.15
omega0=.5
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
plt.xlabel('Time (s)')
plt.ylabel('Position')


plt.savefig('r6__0.svg', bbox_inches='tight')

plt.show()


# In[24]:





# In[6]:


plt.plot(v,x, 'k')
plt.axis('equal')
plt.xlabel(r"$x$")
plt.ylabel(r"$v$")


# In[5]:


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
    lfun = lambda t, y, omegad=i, omega0=omega0, b=0.2, A=1: DampedPendulum(t, y, b, omega0, A, omegad)

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
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend() # Make the plot labels visible

plt.savefig('Oscillator-driven-multi.svg', bbox_inches='tight')
plt.show()


# In[ ]:





# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math


# Set your parameters
omega0 = 1
tf = 10.0  # Final time
n = 1000   # Number of time points
y0 = [0.0, 0.0]  # Initial conditions [angle, angular velocity]

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

# Create an array of driving frequencies ranging from 0 to 2*omega_0
drivFreq = np.linspace(0, 2 * omega0, 100)

# Create a list of damping coefficients (b values)
b = 0.1

# Create a list to store amplitudes for each b value
amplitudes_list = []
t = np.linspace(0.8*tf, tf, n)

amplitudes = []  # Create an empty list to store amplitudes for this b value
    # Loop through each driving frequency in drivFreq
for omegad in drivFreq:
        # Define the anonymous function, including the changing omegad and b
    lfun = lambda t, y, omegad=omegad, omega0=omega0, b=b, A=1: DampedPendulum(t, y, b, omega0, A, omegad)

    result = integrate.solve_ivp(fun=lfun, t_span=(0, tf), y0=y0, method="RK45", t_eval=t)
        
        # Store result of this run in variables t_result, x, v
    t_result = result.t
    v, x = result.y
        
        # Calculate and append the amplitude (peak-to-peak)
    amplitudes.append((max(x) - min(x)) / 2)
    
    # Append the amplitudes for this b value to the list
plt.plot(drivFreq, amplitudes, label=f'b = {b}')

plt.xlabel("Driving Frequency (Omegad)")
plt.ylabel("Amplitude")
plt.savefig('r7.svg')
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math


# Set your parameters
omega0 = 1
tf = 10.0  # Final time
n = 1000   # Number of time points
y0 = [0.0, 0.0]  # Initial conditions [angle, angular velocity]

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

# Create an array of driving frequencies ranging from 0 to 2*omega_0
drivFreq = np.linspace(0, 2 * omega0, 100)

# Create a list of damping coefficients (b values)
b_values = [1.0, 0.5, 0.4, 0.2, 0.1]

# Create a list to store amplitudes for each b value
amplitudes_list = []
t = np.linspace(0.8*tf, tf, n)

# Loop through each damping coefficient (b)
for b in b_values:
    amplitudes = []  # Create an empty list to store amplitudes for this b value
    # Loop through each driving frequency in drivFreq
    for omegad in drivFreq:
        # Define the anonymous function, including the changing omegad and b
        lfun = lambda t, y, omegad=omegad, omega0=omega0, b=b, A=1: DampedPendulum(t, y, b, omega0, A, omegad)

        result = integrate.solve_ivp(fun=lfun, t_span=(0, tf), y0=y0, method="RK45", t_eval=t)
        
        # Store result of this run in variables t_result, x, v
        t_result = result.t
        v, x = result.y
        
        # Calculate and append the amplitude (peak-to-peak)
        amplitudes.append((max(x) - min(x)) / 2)

    plt.plot(drivFreq, amplitudes, label=f'b = {b}')

plt.xlabel("Driving Frequency (Omegad)")
plt.ylabel("Amplitude")
plt.legend()
plt.savefig('r8.svg')
plt.show()


# In[8]:




