"""
Test the convergence of a 100x100 anisotropic diffusion equation
"""
import numpy
import scipy
import matplotlib.pyplot as plt

from pyamg.gallery import stencil_grid
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.strength import classical_strength_of_connection
from pyamg.classical.classical import ruge_stuben_solver

from pyamg.gallery.convergence_tools import print_cycle_history #pulled in from the pyamg-examples repo and updated for python3

if __name__ == '__main__':
    n = 100
    nx = n
    ny = n

    # Rotated Anisotropic Diffusion
    stencil = diffusion_stencil_2d(type='FE',epsilon=0.001,theta=scipy.pi/3)
    A = stencil_grid(stencil, (nx,ny), format='csr')

    saved = []
    work = []
    for test in numpy.arange(0,1,0.01):
        test = 0.36
        numpy.random.seed(625)
        x = scipy.rand(A.shape[0])
        b = A*scipy.rand(A.shape[0])
        ml = ruge_stuben_solver(A, strength=('classical', {'theta': test}), CF='CLJP', max_levels=40, max_coarse=5, keep=True)
    
        resvec = []
        x = ml.solve(b, x0=x, maxiter=20, tol=1e-14, residuals=resvec)
        saved.append(resvec[-1])
        work.append(-ml.cycle_complexity() / scipy.log10((resvec[-1]/resvec[0])**(1.0/len(resvec))))
            if numpy.mod(test,0.05)<0.01:
                print(test)
    
    print_cycle_history(resvec, ml, verbose=True, plotting=False)
        
#plt.figure(figsize=(8,6))
#plt.semilogy(resvec)
#plt.xlabel("iterations")
#plt.ylabel("residual")

plt.figure(figsize=(8,6))
plt.semilogy(numpy.arange(0,1,0.01),saved)
plt.xlabel("theta (strength)")
plt.ylabel("residual after 20 iterations")

plt.figure(figsize=(8,6))
plt.plot(numpy.arange(0,1,0.01),work)
plt.xlabel("theta (strength)")
plt.ylabel("work (relative to matvec) at 20th iteration")