n = 100
A2 = sparse.diags([1], [-1], shape=(n,n), format='csr')
A2[0,:]=1