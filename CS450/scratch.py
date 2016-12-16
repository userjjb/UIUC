import numpy as np
import numpy.linalg as la
from scipy import optimize

t = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
y = [6.8, 3, 1.5, 0.75, 0.48, 0.25, 0.2, 0.15]
def f(t, x1, x2,):
    return x1*np.exp(x2*t)

nl_x, pcov = optimize.curve_fit(f, t, y)

l_x= la.lstsq(np.vstack([np.ones(len(t)),t]).T, np.log(y))[0]
l_x[0] = np.exp(l_x[0])

plt.plot(t,y,'o')
xx=np.linspace(0.5,4,100)
yy=f(xx,nl_x[0],nl_x[1])
plt.plot(xx,yy,'r')
yyy=f(xx,l_x[0],l_x[1])
plt.plot(xx,yyy,'--k')