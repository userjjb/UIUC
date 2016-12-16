import numpy as np
from scipy import rand as srand
from scipy.sparse import csr_matrix

# Excerpted from pyamg

def _rand_sparse(m, n, density, format='csr'):
    """Helper function for sprand, sprandn"""

    nnz = max(min(int(m*n*density), m*n), 0)

    row = np.random.random_integers(low=0, high=m-1, size=nnz)
    col = np.random.random_integers(low=0, high=n-1, size=nnz)
    data = np.ones(nnz, dtype=float)

    # duplicate (i,j) entries will be summed together
    return csr_matrix((data, (row, col)), shape=(m, n))


def sprand(m, n, density, format='csr'):
    """Returns a random sparse matrix.

    Parameters
    ----------
    m, n : int
        shape of the result
    density : float
        target a matrix with nnz(A) = m*n*density, 0<=density<=1
    format : string
        sparse matrix format to return, e.g. 'csr', 'coo', etc.

    Returns
    -------
    A : sparse matrix
        m x n sparse matrix

    Examples
    --------
    >>> A = sprand(5,5,3/5.0)

    """
    m, n = int(m), int(n)

    # get sparsity pattern
    A = _rand_sparse(m, n, density, format='csr')

    # replace data with random values
    A.data = srand(A.nnz)

    return A.asformat(format)

