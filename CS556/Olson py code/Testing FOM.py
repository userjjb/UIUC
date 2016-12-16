
# coding: utf-8

# In[1]:

from fom import fom
from onedprojection import onedprojection
import matplotlib.pyplot as plt
from scipy import sparse
import numpy as np
get_ipython().magic('matplotlib inline')


# In[ ]:

its = 99

n = 100
A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n,n), format='csr')
b = np.zeros((n,))

b = np.zeros((n,))
x0 = np.random.rand(n)
res = []
err = []
res2 = []
err2 = []

x = onedprojection(A, b, x0=x0, tol=1e-8, maxiter=its, residuals=res, method='SD', errs=err)

x2 = fom(A, b, x0=x0, maxiter=its, residuals=res2, errs=err2)


# In[ ]:

plt.figure(figsize=(8,8))
plt.semilogy(res, label='SD residuals')
plt.hold(True)
plt.semilogy(err, label='SD errors')
plt.semilogy(res2, label='FOM residuals')
plt.semilogy(err2, label='FOM errors')
plt.xlabel("iteration")
plt.ylabel("||.||")
plt.legend()


# In[9]:

its = 60

n = 200
d = np.linspace(0.1,1.0,n)
d[0] = 0.0001
d[1] = 0.009
d[2] = 0.008
d[3] = 0.0007
A = sparse.spdiags(d,[0],n,n).tocsr()

b = np.zeros((n,))
x0 = np.random.rand(n)
res = []
err = []
res2 = []
err2 = []

x = onedprojection(A,b,x0=x0,tol=1e-8,maxiter=its,residuals=res,method='SD',errs=err)

x2 = fom(A,b,x0=x0,maxiter=its,residuals=res2,errs=err2)


# In[10]:

plt.figure(figsize=(8,8))
plt.semilogy(res, label='SD residuals')
plt.hold(True)
plt.semilogy(err, label='SD errors')
plt.semilogy(res2, label='FOM residuals')
plt.semilogy(err2, label='FOM errors')
plt.xlabel("iteration")
plt.ylabel("||.||")
plt.legend()


# In[ ]:




# In[ ]:



