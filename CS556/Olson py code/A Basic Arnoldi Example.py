import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import scipy.linalg as sla
get_ipython().magic('matplotlib inline')

from arnoldi import arnoldi

A = sla.toeplitz([2, -1, 0, 0])
print(A)

n = 20
Arb = sparse.diags([1], [-1], shape=(n,n), format='csr')
Arb[0,:]=1
print(Arb)


#r0 = np.ones((4,))
r0 = np.random.rand(4)
V, H = arnoldi(A, 3, v0=r0)

print(V)
print(H)

print(np.sort(sla.eigvals(H).real))
print(np.sort(sla.eigvals(A).real))