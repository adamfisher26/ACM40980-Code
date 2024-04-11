# Network of leaky integrate-and-fire neurons
# Exploring the role of gap junctions
# 26-08-2022

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import sparse

# Set seed
np.random.seed(101)

# Numerical parameters
N = 1             # Number of neurons
T = 500            # Final time in ms
dt = 0.05          # time step in ms
M = int(T / dt)     # number of time steps
t = np.linspace(0, T, M + 1)

# Model parameters
Cm = 1
gL = 0.1
vL = -70
vE = 0
tauE = 1
vth = -50
vreset = -67.3

# Poisson noise
fnu = 20       # combined firing = r*f
r =  100       # firing rate for external spikes per neuron (Hz)
f =  fnu/r     # strength of each spike/input
external_spikes = sparse.csr_matrix(stats.poisson.rvs(r/1000*dt, size=((M + 1, N))))

# Initialise variables
v = np.zeros((M + 1, N))
v[0, :] = stats.norm.rvs(loc=-60, scale=2, size=(1, N))
s = np.zeros((M + 1, N))


# Set up spike time vector
spike_times = np.zeros((M, 2))
spikeCounter = 0


def dxdt(x):

    # Voltage update
    dv =  dt * ( gL * ( vL - x[0] ) + x[1] * ( vE - x[0] ) ) / Cm 
                       
    # Synaptic conductance upadtes (no spike)                        
    ds = dt * ( -x[1] / tauE )
    
    return [dv, ds]

v = np.zeros((M + 1, N))
v[0, :] = stats.norm.rvs(loc=-60, scale=2, size=(1, N))
s = np.zeros((M + 1, N))

# Set up spike time vector
spike_times = np.zeros((M, 2))
spikeCounter = 0


# Loop over time steps
# 2nd order Rumge-Kutta method
for i in range(M):
    
    k1 = dxdt([v[i, :],s[i, :]])                        
    k2 = dxdt([v[i, :] + k1[0]*dt/2, s[i, :] + k1[1]*dt/2])
    
    # integrate LIF volatge equations
    v[i + 1, :] = v[i, :] +  k2[0] #+  sigma_bis_V * np.random.randn(N)
    
    # integrate synaptic equations
    s[i + 1, :] = s[i, :] + k2[1] +  f*external_spikes[i, :]/tauE


    # Determine which, if any, neurons has reached threshold
    spike = np.where(v[i + 1, :] > vth)
    
    # if at least one neuron reached threshold
    if len(spike[0]) > 0:
        
        # loop over those neurons that spike
        for spikeInd in spike[0]:           
            
            # Find spike time (linear interpolation)
            tspike = t[i] + dt * (vth - v[i, spikeInd]) / (v[i+1, spikeInd] - v[i, spikeInd])

            # put spike times into the matrix
            spike_times[spikeCounter,:] = [tspike, spikeInd]
            spikeCounter = spikeCounter + 1
            
            # Reset voltage of neurons that spiked
            v[i + 1, spikeInd] = vreset

spike_times = spike_times[:spikeCounter,:]



# Plot voltages vs time
plt.figure()   
plt.plot(t, v)

# Plot synaptic conductances
plt.figure()  
plt.plot(t, s)


# Remove zeros after final spike times
spike_times = spike_times[:spikeCounter,:]

# Create raster plot
plt.figure()  
plt.scatter(spike_times[:,0],spike_times[:,1])


# Compute ISIs
ISI = spike_times[1:,0] - spike_times[:-1,0]

plt.figure()  
plt.hist(ISI,bins=20,color=(0,0.5,1))
plt.xlabel('ISI')
plt.title('r = '+str(r)+', '+str(spikeCounter)+' spikes')

    

    
