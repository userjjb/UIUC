import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
L = 0.8

def sq_wave(xn): #1 from 0.25 to 0.75, 0 elsewhere
    return np.all([xn>=0.25,xn<=0.75],axis=0)*1
def gaussian(xn):
    return np.exp(-100*(xn-0.5)**2)
def sine(xn,k):
    return np.sin(2*np.pi*k*xn)

#Forward upwind, of the form: u_k,l+1 = u_k - lmbda*(u_k - u_k-1)
res = []
for p in range(5,6):
    ne = 2**p
    #third "diag" is the periodic BC for the right elem wrapping to the left
    A = diags([1-L,L,L],[0,-1,ne],shape=(ne+1,ne+1))
    xn = np.linspace(0,1,ne+1)
    
    u0 = sine(xn,2) #Save for later comparison to calc L2 norm
    u = u0.copy()
    for t in range(int(ne/L)+1):
        u = A@u
    res.append(np.sqrt(np.sum((u0-u)**2)/ne)) #discrete L2 norm
plt.plot(xn,u0)
plt.plot(xn,u*1.7)

#ne = 2**np.arange(6,15)
#plt.figure(figsize=(8,6))
#plt.loglog(ne,res,'.')
#m, b = np.polyfit(np.log10(ne), np.log10(res), 1)
#plt.loglog(ne, 10**(m*np.log10(ne) + b), '-')
#plt.xlabel('Number of grid points')
#plt.ylabel('Discrete L2 norm of error')