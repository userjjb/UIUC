import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def BCG(A,b,x0,tol,maxit,snoop):
    res=[]
    err=[]
    resp=[]
    r = b-A@x0
    rs = r.copy()
    #rs = np.random.rand(r.shape[0])
    p=r.copy()
    ps=rs.copy()
    x1=x0.copy()
    for i in range(0,maxit): #Follows Saad's Alg 7.3
        a = (r@rs)/np.dot(A@p,ps)
        x1 += a*p
        rold=r.copy()
        r -= a*A@p
        res.append(la.norm(r))
        resp.append(r[snoop])
        err.append(la.norm(x1-np.ones(x1.shape)))
        if la.norm(r)<tol:
            break
        rsold=rs.copy()
        rs -= a*A.T@ps
        bt = (r@rs)/(rold@rsold)
        p = r + bt*p
        ps = rs + bt*ps
    return x1,res,err,resp

n = 200
#callback counter as described by "ali_m" on SO:
#(http://stackoverflow.com/questions/33512081/getting-the-number-of-iterations-of-scipys-gmres-iterative-method)
class cg_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.resk = [] #Save residuals for later use -JJB
        self.errk = []
    def __call__(self, sk=None):
        self.niter += 1
        temp = la.norm(A*sk-b)
        self.resk.append(temp) #Save residuals for later use -JJB
        self.errk.append(la.norm(sk-np.ones(n)))
        if self._disp:
            1#print('iter %3i\trk = %s' % (self.niter, str(temp)))
            
D = np.linspace(1,1000, n)-30
#D[-1] = 1050
A = sparse.diags(D)
b = A@np.ones(n)

x0 = np.random.rand(n)

snoop = 0
x_bcg, res, err, resp = BCG(A,b,x0,1e-8,300,snoop)
cnt1 = cg_counter()
x_cg, info = spla.cg(A, b, x0, 1e-12, 300, callback=cnt1)

plt.figure(figsize=(9,6))
plt.semilogy(res,"-b",label='BCG k=50',linewidth=2.0)
plt.hold(True)
plt.semilogy(err,"--b")
plt.semilogy(cnt1.resk,"-r",label='CG',linewidth=2.0)
plt.semilogy(cnt1.errk,"--r")
plt.xlabel("iterations")
plt.ylabel("log-residual (-)   log-error (--)")
plt.legend()

plt.figure(figsize=(9,6))
plt.semilogy(np.abs(resp),"-b",label='Residual Component '+str(snoop))
plt.xlabel("iterations")
plt.ylabel("log-residual")
plt.legend()