import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,2,100)
def gamma(R):
    if 0:
        result = 1/R
    else:
        result = 1-0.5*(R**2-1)+(3/8)*(R**2-1)**2
    return result

y = 1/x

a = 1
ys = y - (1/a)*gamma(x/a)

plt.figure(figsize=(8,6))
plt.plot(x,y,'b',label='Original unsmoothed')
plt.plot(x,ys,'r',label='Smoothed, ready for interpolation')
plt.xlabel('Non-dimensional distance')
plt.ylabel('Relative strength')
plt.legend()