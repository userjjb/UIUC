import numpy as np
import scipy.integrate as spint
import matplotlib.pyplot as plt
   
def KermMck(y, t, c, d):
    y1, y2, y3 = y
    dydt = [-c*y1*y2, c*y1*y2 - d*y2, d*y2]
    return dydt

c = 1
d = 5
y0 = [95, 5 ,0]
t = np.linspace(0, 1, 101)

sol = spint.odeint(KermMck, y0, t, args=(c, d))

plt.plot(t,sol[:,0], label='Susceptibles')
plt.plot(t,sol[:,1], label='Infectives in Circulation')
plt.plot(t,sol[:,2], label='Infectives Removed')
plt.xlabel("Time")
plt.ylabel("Percent")
plt.title('Kermack-McKendrick epidemic model')
plt.legend(loc=5)

y1 = sol[-1,:]