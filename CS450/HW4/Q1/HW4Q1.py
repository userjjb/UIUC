import numpy as np
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

def f1(x):
    return (x[0]+3)*(x[1]**3-7)+18
def f2(x):
    return np.sin(x[1]*np.exp(x[0])-1)
    
def f(x):
    return np.array([f1(x),f2(x)])
#----------------------------------------    
def J11(x):
    return x[1]**3-7
def J12(x):
    return 3*(3 + x[0])*x[1]**2
def J21(x):
    return np.exp(x[0])*x[1]*np.cos(1 - np.exp(x[0])*x[1])
def J22(x):
    return np.exp(x[0])*np.cos(1 - np.exp(x[0])*x[1])

def J(x):
    return np.array([[J11(x),J12(x)],[J21(x),J22(x)]])
    
x0 = np.array([-0.5,1.4])
x02 = x0.copy()
xs = np.array([0,1])

err = []
tol = 2*np.spacing(max(xs))
maxit = 100
for it in range (0, maxit):
    err.append(la.norm(x0-xs))
    if max(abs(xs-x0))<tol:
        break
    x0 += la.solve(J(x0),-f(x0))

B = J(x02)
err2 = []
tol2 = 2*np.spacing(max(xs))
maxit2 = 100
for it2 in range (0, maxit2):
    err2.append(la.norm(x02-xs))
    if max(abs(xs-x02))<tol2:
        break
    s = la.solve(B,-f(x02))
    x02 += s
    B += np.outer(f(x02),s)/s@s

plt.semilogy(err,"-r",label='Newton')
plt.hold(True)
plt.semilogy(err2,"-b",label='Broyden')
plt.xlabel("iterations")
plt.ylabel("L2 norm error")
plt.legend()
plt.title('Convergence of Newton vs Broyden Methods for a system of NL EQs')

print("Newton:",it)
print("Broyden:",it2)