import numpy as np
import scipy.integrate as spint
import scipy.sparse as sparse

saved = []
t = np.linspace(0, 2, 32) #Time levels
for p in range(2,7):
    n = 2**p #Number of nodes (inner and BCs)
    
    A = sparse.diags([[1],[-2],[1]],[-1,0,1],(n-2,n-2)) #'A' is our spatial diff stencil
    c = (n-1)**2 #1/dx^2
    
    def wave(Y, t, c):
        y, dy = np.reshape(Y,(2,n)) #odeint expects only 1 vector, unstack
        dYdt = [dy,np.r_[0, c*A*y[1:-1], 0]] #use spatial discretization to calc y''
        dYdt = np.reshape(dYdt,2*n) #restack
        return dYdt
    
    x = np.linspace(0,1,n) #Node locations
    Y0 = np.array([np.sin(np.pi*x), 0*x])
    Y0 = Y0.reshape(2*n)
    
    sol = spint.odeint(wave, Y0, t, args=(c,))
    saved.append(max(abs(Y0[0:n]-sol[-1,0:n]))) #Compute max abs error

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

xm,tm = np.meshgrid(x,t)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xm, tm, sol[:,:n], rstride=1, cstride=1, cmap=cm.viridis, linewidth=0)
plt.title('Space-time plot of wave')
plt.xlabel('Space')
plt.ylabel('Time')
plt.show()

plt.loglog(1/(2.**np.arange(2,7)-1),saved)
plt.title('Log-log plot of spatial convergence rate')
plt.xlabel('$\Delta x$')
plt.ylabel('Max Absolute Error')