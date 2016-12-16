import scipy as sp
import numpy as np
import numpy.linalg as la
import scipy.interpolate as interp
import matplotlib.pyplot as plt

pop=np.array(\
[76212168,
92228496,
106021537,
123202624,
132164569,
151325798,
179323175,
203302031,
226542199])

n = len(pop)
t = np.linspace(1900,1980,n)
tI = np.arange(1900,1981)
#1
p = np.arange(n).reshape(n,1)
mat1 = (t**p).T
mat2 = ((t-1900)**p).T
mat3 = ((t-1940)**p).T
mat4 = (((t-1940)/40)**p).T
#2
cond1 = la.cond(mat1)
cond2 = la.cond(mat2)
cond3 = la.cond(mat3)
cond4 = la.cond(mat4) #best
#3
coeffs = la.solve(mat4,pop)

tIm = (np.arange(1900,1981)-1940)/40 #map onto [-1,1]
pI = coeffs[-1]
for i in range(n-1,0,-1):
    pI = coeffs[i-1] + tIm*pI #Horner's

plt.plot(t,pop,'o',markerfacecolor='none',label='Census Data')
plt.plot(tI,pI,'b',label='Interpolation')
plt.xlabel("Year")
plt.ylabel("US Population")
plt.title('An Interpolation of US Census Data')
#4
MHCI = interp.PchipInterpolator(t,pop)
plt.plot(tI,MHCI(tI),'r',label='Monotone Hermite Cubic')
#5
CS = interp.UnivariateSpline(t,pop)
plt.plot(tI,CS(tI),'--g',label='Cubic Spline')
plt.legend(loc=2)

pIe = coeffs[-1]
for i in range(n-1,0,-1):
    pIe = coeffs[i-1] + 5/4*pIe #5/4 is the mapped value of 1990

exact = 248709873
err_polynomial = abs(pIe-exact)/exact
err_hermite = abs(MHCI(1990)-exact)/exact
err_spline = abs(CS(1990)-exact)/exact
#6
coeffsA = la.solve(mat4,1e6*np.round(pop/1e6))
err_coeffs = abs((coeffsA-coeffs)/coeffs)