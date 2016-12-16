import numpy as np
import scipy.sparse as sparse
import scipy.optimize as spopt
import matplotlib.pyplot as plt

saved1 = []
tries = [1,3,7,15]
for n in tries:
    A = sparse.diags([[1],[-2],[1]],[-1,0,1],(n+2,n+2))
    A = A.toarray()[1:-1,:]
    c = (n+1)**2
    
    t = np.linspace(0,1,n+2)
    t = t[1:-1]
    
    def F(x):
        return c*A@np.r_[0,x,1] - 10*x**3 - 3*x - t**2
    
    plt.plot(np.r_[0,t,1],np.r_[0,spopt.broyden1(F,t),1],'-o',markersize=3, markerfacecolor='None',label='n='+str(n))

plt.legend(loc=2)
plt.xlabel('t')
plt.ylabel('u')
plt.title('Finite Difference solution of non-linear BVP')