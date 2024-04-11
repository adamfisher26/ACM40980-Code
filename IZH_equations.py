import numpy as np

def IS_RK(y,order,C,I,vr,vt,k,a,b,k_u):
    '''
    Algorithm of the evolution of the Izhikevich model, returning the numpy array dydt
    '''

    dvdt = (k * (y[0] - vr) * (y[0] - vt) - k_u * y[1] + I ) / C
    dudt = a * (b*(y[0] - vr) - y[1])

    dydt = np.array([dvdt,dudt])

    return dydt


# Parameter values
C = 1
vr = -70
vt = -45
k = 0.004
a = 0.1
b = 8
c = -65
d = 100
vpeak = 25
k_u = 0.002
