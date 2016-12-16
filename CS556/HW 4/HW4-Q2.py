import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import random_sparse_mat_generator as rs

#Calculates ILU0 factorization of a matrix A, returning L and U which can be used for preconditioning
def ILU0(A): #based on ikj ordering, Alg 10.4 from Saad
    n = A.shape[1]
    A2 = A.copy()
    (I,J,V) = sparse.find(A2)
    for i in range(1,n): #this many nested for loops isn't great for performance, but is more readable than the vectorized version probably
        krange = J[(I==i)*(J<i)] #Find nonzero pattern
        if np.size(krange): #If nonzero pattern exists
            for k in np.nditer(krange):
                A2[i,k] /= A2[k,k]
                jrange = J[(I==i)*(J>k)]
                if np.size(jrange):
                    for j in np.nditer(jrange):
                        A2[i,j] -= A2[i,k]*A2[k,j]
    L = sparse.tril(A2)
    L.setdiag(1)
    U = sparse.triu(A2)
    return L,U

#callback counter as described by "ali_m" on SO:
#(http://stackoverflow.com/questions/33512081/getting-the-number-of-iterations-of-scipys-gmres-iterative-method)
class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.res = [] #Save residuals for later use -JJB
    def __call__(self, rk=None):
        self.niter += 1
        self.res.append(rk) #Save residuals for later use -JJB
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))
n = 6
n = trial[ite]
density = 0.1

#Random sparse matrix + I
A = rs.sprand(n**2, n**2, density)+sparse.eye(n**2)

#2D Poisson Stencil
c = sparse.diags([-1, 4, -1],[-1, 0, 1],shape=(n,n))
C = sparse.block_diag([c]*n,format="csr")
C.setdiag(-1,-n)
C.setdiag(-1,n)

#--Compare GMRES to ILU0 preconditioned GMRES for 2d Poisson
print("2D Poisson")
L,U = ILU0(C)
#Construct a linear operator that avoids the need to directly calculate the inverse of the preconditioner
M_x = lambda x: spla.spsolve(L*U, x)
M = spla.LinearOperator((n**2, n**2), M_x)

x0 = np.random.rand(n**2)
b = np.zeros(n**2)

counter_C = gmres_counter()
x_C,info_C = spla.gmres(C,b,x0=x0,tol=1e-8,callback=counter_C)
print("2D Poisson-PC")
counter_Cpre = gmres_counter()
x_Cpre,info_Cpre = spla.gmres(C,b,x0=x0,tol=1e-8,M=M,callback=counter_Cpre)

#--Compare GMRES to ILU0 preconditioned GMRES for random matrix example
print("Random")
L,U = ILU0(A)
M_x = lambda x: spla.spsolve(L*U, x)
M = spla.LinearOperator((n**2, n**2), M_x)

#use same b as for matrix C

counter_A = gmres_counter()
x_A,info_A = spla.gmres(A,b,x0=x0,tol=1e-8,callback=counter_A)
print("Random-PC")
counter_Apre = gmres_counter()
x_Apre,info_Apre = spla.gmres(A,b,x0=x0,tol=1e-8,M=M,callback=counter_Apre)

plt.figure(figsize=(8,8))
plt.semilogy(counter_A.res, label='Random')
plt.hold(True)
plt.semilogy(counter_Apre.res, label='Random-PC')
plt.xlabel("iterations")
plt.ylabel("residual")
plt.legend()

plt.figure(figsize=(8,8))
plt.semilogy(counter_C.res, label='Poisson')
plt.hold(True)
plt.semilogy(counter_Cpre.res, label='Poisson-PC')
plt.xlabel("iterations")
plt.ylabel("residual")
plt.legend()