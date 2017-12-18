import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

V = np.loadtxt('mesh.v')
E = np.loadtxt('mesh.e', dtype=int)
import refine_mesh
V, E = refine_mesh.refine2dtri(V, E)
#V, E = refine_mesh.refine2dtri(V, E)
nv = V.shape[0]
ne = E.shape[0]
X, Y = V[:, 0], V[:, 1]

ID = np.kron(np.arange(0, ne), np.ones((3,)))
G = sparse.coo_matrix((np.ones((ne*3,)), (E.ravel(), ID,)))
E2E = G.T * G
E2Ec = E2E.copy()
V2V = G * G.T
V2V = V2V.todia()
E2E.data = 0.0 * E2E.data + 1.0
nbrs = np.array(E2E.sum(axis=0)).ravel()

def kappa(x, y):
    if (x**2 + y**2)**0.5 <= 0.25:
        return 1.0
    else:
        return 1

def f(x, y):
    if 1:
        return -np.sin(np.pi*x)*np.sin(np.pi*y)
    else:
        return -np.sin(np.pi*x)*np.sin(np.pi*y)
    
def g(x):
    return 0#10 * (1 - x**2)

AA = np.zeros((ne, 9))
IA = np.zeros((ne, 9))
JA = np.zeros((ne, 9))
bb = np.zeros((ne, 3))
ib = np.zeros((ne, 3))
jb = np.zeros((ne, 3))

from timeit import default_timer as timer
start = timer()
for ei in range(0, ne):
    # Step 1
    K = E[ei, :]
    x0, y0 = X[K[0]], Y[K[0]]
    x1, y1 = X[K[1]], Y[K[1]]
    x2, y2 = X[K[2]], Y[K[2]]

    # Step 2
    J = np.array([[x1 - x0, x2 - x0],
                  [y1 - y0, y2 - y0]])
    invJ = la.inv(J.T)
    detJ = la.det(J)

    # Step 3
    dbasis = np.array([[-1, 1, 0],
                       [-1, 0, 1]])

    # Step 4
    dphi = invJ.dot(dbasis)

    # Step 5
    Aelem = kappa(X[K].mean(), Y[K].mean()) *\
        (detJ / 2.0) * (dphi.T).dot(dphi)

    # Step 6
    belem = f(X[K].mean(), Y[K].mean()) *\
        (detJ / 6.0) * np.ones((3,))

    # Step 7
    AA[ei, :] = Aelem.ravel()
    IA[ei, :] = [K[0], K[0], K[0], K[1], K[1], K[1], K[2], K[2], K[2]]
    JA[ei, :] = [K[0], K[1], K[2], K[0], K[1], K[2], K[0], K[1], K[2]]
    bb[ei, :] = belem.ravel()
    ib[ei, :] = [K[0], K[1], K[2]]
    jb[ei, :] = 0
end = timer()
print(end - start)
    
A = sparse.coo_matrix((AA.ravel(), (IA.ravel(), JA.ravel())))
A = A.tocsr()
A = A.tocoo()
b = sparse.coo_matrix((bb.ravel(), (ib.ravel(), jb.ravel())))
b = b.tocsr()
b = np.array(b.todense()).ravel()

tol = 1e-12
Dflag = np.logical_or.reduce((abs(X+1) < tol,
                              abs(X-1) < tol,
                              abs(Y+1) < tol,
                              abs(Y-1) < tol))
gflag = abs(Y+1) < tol
ID = np.where(Dflag)[0]
Ig = np.where(gflag)[0]

u0 = np.zeros((nv,))
u0[Ig] = g(X[Ig])

b = b - A * u0

for k in range(0, len(A.data)):
    i = A.row[k]
    j = A.col[k]
    if Dflag[i] or Dflag[j]:
        if i == j:
            A.data[k] = 1.0
        else:
            A.data[k] = 0.0

b[ID] = 0.0

A = A.tocsr()
u = sla.spsolve(A, b)

u = u + u0

#fig = plt.figure(figsize=(8,8))
#ax = plt.gca(projection='3d')
#ax.plot_trisurf(X, Y, u, triangles=E, cmap=plt.cm.jet, linewidth=0.2)
#plt.show()

vertx = X[E[:,0:3]]
verty = Y[E[:,0:3]]
x0=vertx[:,0]
x1=vertx[:,1]
x2=vertx[:,2]
y2=verty[:,2]
y1=verty[:,1]
y0=verty[:,0]
J = np.array([[x1 - x0, x2 - x0],
              [y1 - y0, y2 - y0]])
detJm = np.zeros(J.shape[2])
for i in range(J.shape[2]):
    detJm[i] = la.det(J[:,:,i])

exact = 1/(2*np.pi**2)
tt=exact*f(X[E],Y[E])-u[E]
ttt=(detJm/6)*np.sum(tt**2,axis=1)
print(np.sqrt(np.sum(ttt)))
#print(-np.polyfit(np.log10([688,2752,11008]),np.log10([0.000539488245637,0.000134234103075,3.34377599782e-05]),1)[0])