import numpy as np
import pylab as plt
import scipy.integrate as spint

def f(y, x, yp, xp):
    return 1/np.sqrt((xp-x)**2+(yp-y)**2)
    
def one(x):
    return 1
    
def negone(x):
    return -1

survey = np.linspace(2,10,5)
results = np.zeros([len(survey),len(survey)])

for xi in range(len(survey)):
    for yi in range(xi, len(survey)):
        def g(y,x):
            return f(y,x,survey[yi],survey[xi])
            
        res,abserr = spint.dblquad(f,-1,1,negone,one,args=(survey[yi],survey[xi]))
        results[xi,yi] = res
        results[yi,xi] = res #geometric symmetry

plt.imshow(np.flipud(results),cmap='viridis',extent=[2,10,2,10])
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Electric Potential')
plt.colorbar(label='Electric Potential Magnitude')
pot_10_10 = res