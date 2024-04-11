import numpy as np

def ML_RK(y, order, psi, V1, V2, V3, V4, gna, gk, gl, Ena, Ek, El, C, I):
    '''
    Algorithm of the evolution of the Morris-Lecar model, returning the numpy array dydt
    '''

    minf = 0.5 * (1 + np.tanh( ((y[0] - V1 )/ V2)))
    Iion = gna * minf * (y[0] - Ena) + gk * y[1] * (y[0] - Ek) + gl * (y[0] - El)

    dvdt = ( - Iion  + I ) / C

    winf = 0.5 * (1 + np.tanh( (y[0] - V3) / V4))
    dwdt = psi * (winf - y[1])*np.cosh( (y[0] - V3) / (2 * V4))

    dydt = np.array([dvdt,dwdt])

    return dydt

# Parameter values
psi = 0.4
V1 = -1.2
V2 = 20
V3 = 2
V4 = 19
gk = 2
gna = 2
gl = 0.12
Ek = -90
Ena = 70
El = -72.2262
C = 1
