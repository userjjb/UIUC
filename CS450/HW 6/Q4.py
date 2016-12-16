import numpy as np
import scipy.integrate as spint
import matplotlib.pyplot as plt
   
def orbit(Y, t, e):
    x, y, vx, vy = Y
    dYdt = [vx,vy,-x/np.sqrt(x**2+y**2)**3,-y/np.sqrt(x**2+y**2)**3]
    return dYdt

def boil_plt(x,y,xlabel,ylabel,title,aspect='auto'):
    plt.figure()
    plt.plot(x,y)
    plt.axis(aspect)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

for e in [0, 0.5, 0.9]:
    Y0 = [1-e, 0, 0, np.sqrt((1+e)/(1-e))]
    t = np.linspace(0, 6*np.pi, 1001)
    
    sol = spint.odeint(orbit, Y0, t, args=(e,))
    
    boil_plt(sol[:,0],sol[:,1],'x','y','y vs x, e = '+str(e),aspect='equal')
    boil_plt(t,sol[:,0],'t','x','x vs t, e = '+str(e))
    boil_plt(t,sol[:,1],'t','y','y vs t, e = '+str(e))

ConsE = (sol[:,2]**2+sol[:,3]**2)/2 - 1/np.sqrt(sol[:,0]**2+sol[:,1]**2)
ConsAM = sol[:,0]*sol[:,3] - sol[:,1]*sol[:,2]

boil_plt(t,ConsE-ConsE[0],'t','Absolute Change in Energy','Conservation of Energy, e = '+str(e))
boil_plt(t,ConsAM-ConsAM[0],'t','Absolute Change in Angular Momentum','Conservation of Angular Momentum, e = '+str(e))