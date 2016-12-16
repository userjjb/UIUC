import scipy as sp
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def f(x):
    return 4/(1+x**2)

Mid=np.array([]); Trap=np.array([]); Simp=np.array([]); Romb=np.array([]);
for i in range(25):#max iters
    h = 1/(2**i)
    #Midpoint
    Mx = (np.linspace(0,1,1/h+1)+h/2)[:-1]
    Mid = np.hstack([Mid,(h*np.sum(f(Mx)))])
    #Trapezoid
    Tx = np.linspace(0,1,1/h+1)[1:-1]
    Trap = np.hstack([Trap,(h*(f(0)/2 + np.sum(f(Tx)) + f(1)/2))])
    if Trap[i]<1.5*np.spacing(np.pi):
        break

for i in range(10):#max iters
    #Simpson's
    Simp = np.hstack([Simp,((2/3)*Mid[i] + Trap[i]/3)])
#Romberg
T = np.zeros([len(Trap),len(Trap)])
T[:,0] = Trap
for i in range(1,len(Trap)):
    for j in range(1,i+1):
        T[i,j] = T[i,j-1]+(T[i,j-1]-T[i-1,j-1])/(2**(j+1)-1)
    Romb = np.hstack([Romb,T[i,i]])
    if abs(Romb[-1]-np.pi)<(1.5*np.spacing(np.pi)):
        break
    
plt.loglog(1/(2**np.arange(len(Mid))), abs(Mid-np.pi)/np.pi,'r',label='Midpoint')
plt.loglog(1/(2**np.arange(len(Trap))), abs(Trap-np.pi)/np.pi,'g',label='Trapezoid')
plt.loglog(1/(2**np.arange(len(Simp))), abs(Simp-np.pi)/np.pi,'b',label='Simpson')
plt.loglog(1/(2**np.arange(1,len(Romb)+1)), abs(Romb-np.pi)/np.pi,'k',label='Romberg')

plt.xlabel("h")
plt.ylabel("Relative Error")
plt.title('Convergence of Various Quadrature Methods')
plt.legend(loc=2)

print('Error stops improving for computed value of pi because eventually machine precision is reached. Some methods appear not to reach ~4e16, but that is because the next steps relative error is 0, which is not displayed on a log plot.')

#Monte Carlo
MC = np.zeros(6)
for i in range(6):
    MC[i] = abs(np.mean(f(np.random.rand(10**i)))-np.pi)/np.pi

plt.figure()    
plt.loglog(10**np.arange(6), MC,'k')
plt.xlabel("n, number of eval points")
plt.ylabel("Relative Error")
plt.title('Convergence of Monte Carlo Quadrature')