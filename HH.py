def HH_RK(y,gna,gk,gl,Ena,Ek,El,C,I):
    '''
    Algorithm that calculates the changes in v, m, h, and n as per the HH model equations. It returns an np.array containing these changes
    '''
    vt = -58 
    Ina = gna * y[2]**3 * y[3] * (y[0] - Ena)
    Ik = gk * y[1]**4 * (y[0]- Ek)
    
    dvdt = (-Ina -Ik - gl * (y[0] - El) + I ) / C 

    dmdt = am(y[0],vt) * (1-y[2]) - bm(y[0],vt) * y[2]
    dhdt = ah(y[0],vt) * (1-y[3]) - bh(y[0],vt) * y[3]
    dndt = an(y[0],vt) * (1-y[1]) - bn(y[0],vt) * y[1]

    dydt = [dvdt,dndt,dmdt,dhdt]

    return dydt

def an(v,vt):
    return -0.032 * (v-vt-15) / (np.exp(-(v-vt-15)/5)-1)

def bn(v,vt):
    return 0.5* np.exp(-(v-vt-10)/40)

def am(v,vt):

    return -0.32 * (v-vt-13) / (np.exp(-(v-vt-13)/4)-1)

def bm(v,vt):
    return 0.28 * (v-vt-40) / (np.exp((v-vt-40)/5) -1)

def ah(v,vt):
    return 0.128*np.exp(-(v-vt-17)/18)

def bh(v,vt):
    return 4 / (1+np.exp(-(v-vt-40)/5))

# Parameter values
gna=30
gk=5
gl=0.1
Ena=30
Ek=-90
El=-70
C=1
