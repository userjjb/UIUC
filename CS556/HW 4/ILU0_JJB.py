import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse as sparse

def ILU0(A):
    n = A.shape[1]
    A2 = A.copy()
    (I,J,V) = sparse.find(A2)
    for i in range(1,n):
        krange = J[(I==i)*(J<i)] #Find nonzero pattern
        if np.size(krange):
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