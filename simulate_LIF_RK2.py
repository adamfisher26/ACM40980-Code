# Single leaky integrate-and-fire neuron
# Exploring the effect of noise on firing rate
# 15-05-2023

import numpy as np
import matplotlib.pyplot as plt

# Set seed
np.random.seed(101)

# Numerical parameters
T = 500             # Final time in ms
dt = 0.05          # time step in ms
M = int(T / dt)     # number of time steps
t = np.linspace(0, T, M + 1)

# Model parameters
Cm = 1
gL = 0.1
vL = -70
vth = -49.2
vreset = -66.9
I = 0

# Gaussian noise
sigma = 0
sigma_bis_V = np.sqrt(dt) * sigma * np.sqrt(2. / Cm)

# Initialise variable
v = np.zeros(M + 1)
v[0] = vreset

# Set up spike time vector
spike_times = np.zeros(M)
spikeCounter = 0

def dxdt(x):

    # Voltage update
    dv =  ( gL * ( vL - x ) + I ) / Cm
    
    return dv


# Loop over time steps
# 2nd order Rumge-Kutta method
for i in range(M):
    
    k1 = dt * dxdt(v[i])                           
    k2 = dt * dxdt(v[i] + k1*dt/2)
    
    # integrate LIF voltage equations
    v[i + 1] = v[i] +  k2 +  sigma_bis_V * np.random.randn()
    
    # determine if neuron reached threshold
    if v[i + 1] > vth:
        
        # Find spike time (linear interpolation)
        tspike = t[i] + dt * (vth - v[i]) / (v[i+1] - v[i])
        
        # RK2 step from spike time to end of time step
        dt_from_spike =  dt * (1 - (vth - v[i]) / (v[i+1] - v[i]))
        k1_from_spike = dt_from_spike * dxdt(v[i])
        k2_from_spike = dt_from_spike * dxdt(v[i] + k1_from_spike*dt_from_spike/2)
        
        # integrate voltage equation from reset
        v[i + 1] = vreset + k2_from_spike
            
        # put spike times into the matrix
        spike_times[spikeCounter] = tspike
        spikeCounter = spikeCounter + 1
            
                  
# Plot voltages vs time
plt.figure()   
plt.plot(t, v)
plt.xlabel('Time (ms)')
plt.ylabel('Voltage')

# Remove zeros after final spike times
spike_times = spike_times[:spikeCounter]

# Create raster plot - each ddot is a spike time
plt.figure()  
plt.scatter(spike_times,np.ones(spike_times.shape))
plt.xlim([0,T])
plt.xlabel('Time (ms)')
plt.ylabel('Neuron')


    
    
    