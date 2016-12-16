import scipy as sp
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
import stencil
from timeit import default_timer as timer

def hnorm(r):
    """define ||r||_h = h ||r||_2"""
    n = len(r)
    h = 1.0 / (n+1)
    hrnorm = h * np.linalg.norm(r)
    return hrnorm
    
def relaxGS(A, u, f, nu):
    n = A.shape[0]
    unew = u.copy()
    DE = sparse.tril(A, 0).tocsc()
    
    for i in range(nu):
        unew += sla.spsolve(DE, f - A * unew, permc_spec='NATURAL')

    return unew

#For some reason this seems to suffer from numerical inaccuracies    
#def relaxWJ(A, u, f, nu):
#    n = np.sqrt(A.shape[0])
#    unew = u.copy()
#    Dinv = 1.0 / (4.0 * ((n+1)**2))
#    omega = 2.0 / 3.0
#    
#    for steps in range(nu):
#        unew += omega * Dinv * (f-A*unew)
#    
#    return unew

def relaxWJ(A, u, f, nu):
    n = A.shape[0]
    unew = u.copy()
    D = sparse.diags(A.diagonal(),format="csc")
    omega = 2.0 / 3.0
    
    for i in range(nu):
        unew += sla.spsolve(D/omega, f - A * unew, permc_spec='NATURAL')

    return unew

def interpolation1d(nc, nf):
    d = np.repeat([[1, 2, 1]], nc, axis=0).T
    I = np.zeros((3,nc))
    for i in range(nc):
        I[:,i] = [2*i, 2*i+1, 2*i+2]
    J = np.repeat([np.arange(nc)], 3, axis=0)
    P = sparse.coo_matrix(
        (d.ravel(), (I.ravel(), J.ravel()))
        ).tocsr()
    return 0.5 * P

def create_operator(n, sten):
    """
    Create a 2D operator from a stencil.
    """
    A = stencil.stencil_grid(sten, (n, n), format='csr')
    return A
    
def twolevel(A, P, A1, u0, f0, nu):
    u0 = relaxWJ(A, u0, f0, nu) # pre-smooth
    f1 = P.T * (f0 - A * u0)  # restrict

    u1 = sla.spsolve(A1, f1)  # coarse solve

    u0 = u0 + P * u1          # interpolate
    u0 = relaxWJ(A, u0, f0, nu) # post-smooth
    return u0
#---------------------------    
def vcycle(kmax, kmin, A, u, f, nu):
    ulist = [None for k in range(kmax+1)]
    flist = [None for k in range(kmax+1)]
    Alist = [None for k in range(kmax+1)]
    Plist = [None for k in range(kmax+1)]

    print('grid: ', end=' ')
    # down cycle
    for k in range(kmax, kmin, -1):
        print(k, end=' ')
        u = relaxGS(A, u, f, nu)
        ulist[k] = u
        flist[k] = f
        Alist[k] = A
        
        P1d = interpolation1d(2**(k-1) - 1, 2**k - 1)
        P = sparse.kron(P1d, P1d).tocsr() #Bilinear interpolation
        Plist[k] = P
        
        f = P.T * (f - A * u) #Restrict
        u = np.zeros(f.shape)
        A = P.T * A * P 

    # coarsest grid
    print(kmin, end=' ')
    flist[kmin] = f
    ulist[kmin] = sla.spsolve(A, f)

    # up cycle
    for k in range(kmin+1, kmax+1, 1):
        print(k, end=' ')
        u = ulist[k]
        f = flist[k]
        P = Plist[k]
        A = Alist[k]
        uc = ulist[k-1]
        u += P * uc
        u = relaxGS(A, u, f, nu)
    print('.')
    return u

# Problem setup
k = 10
n = 2**k - 1
nc = 2**(k-1) - 1
sten = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
A = (n+1)**2 * create_operator(n, sten)
f = np.zeros(n*n)
u = np.random.rand(n*n)
usave = u #reuse same random u for direct comparison

# Multigrid cycling
elapsed = []
resid = []
for nu in range(1,11):
    print(nu)
    u = usave
    res = [hnorm(f - A * u)]
    start = timer()
    while(timer()-start<30):
        u = vcycle(k, 2, A, u, f, nu)
        res.append(hnorm(f - A * u))

    elapsed.append(timer()-start)
    resid.append(res)

#Unhelpful plot of residual vs iteration
plt.figure(figsize=(8,6))
for i in range(0,10):
    plt.semilogy(range(len(resid[i])),resid[i],label=i+1)
plt.xlabel("iterations")
plt.ylabel("residual")
plt.legend()
plt.show()

#Timescaled plot of residual vs clock time
plt.figure(figsize=(8,6))
for i in range(0,10):
    plt.semilogy(np.linspace(0,elapsed[i],len(resid[i])),resid[i],label=i+1)
plt.xlabel("wall clock elapsed (s)")
plt.ylabel("residual")
plt.legend()
plt.show()

#Plot of residual after 5 iterations
plt.figure(figsize=(8,6))
plt.semilogy(range(1,11),[item[-1] for item in resid])
plt.xlabel("Number of smoothing passes (kmax=10)")
plt.ylabel("final residual after 30 seconds")

#Plot of residual after 5 iterations
plt.figure(figsize=(8,6))
plt.plot(range(1,11),np.array([item[3] for item in resid])/np.array([item[2] for item in resid]))
plt.xlabel("Numbering of smoothing passes (kmax=10)")
plt.ylabel("convergence factor at iteration 4")