import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

na = np.newaxis

V = np.loadtxt('mesh.v')
E = np.loadtxt('mesh.e', dtype=int)
import refine_mesh
#V, E = refine_mesh.refine2dtri(V, E)
#V, E = refine_mesh.refine2dtri(V, E)
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

# Calculate edge midpoints of elements
Xm = (X[E] + np.roll(X[E],-1, axis=1)) / 2
Ym = (Y[E] + np.roll(Y[E],-1, axis=1)) / 2

tol = 1e-12

# Loop through each element/edge and assign a global number if needed
# Find matching sister edge from candidates in E2Ec and assign same number
# Add new global vertices to V
Vm = np.zeros((ne*3,2)) # big enough to contain all midpoint nodes
count = int(0)
Es = int(np.max(E))
E = np.hstack([E,np.zeros(E.shape, dtype=int)])
for i in range(ne):
    for j in range(3):
        if not(E[i,j+3]):
            Vm[count,:] = [Xm[i,j],Ym[i,j]]
            count += 1
            E[i,j+3] = Es + count
            # number edge for adjacent elem that shares midpoint's edge
            adjel = E2Ec[i,:].indices[E2Ec[i,:].data==2]
            eln,edn = np.where((abs(Xm[i,j]-Xm[adjel]) < tol) & (abs(Ym[i,j]-Ym[adjel]) < tol))
            E[adjel[eln], edn+3] = Es + count
V = np.vstack([V,Vm[:count,]])
nv = V.shape[0]
X, Y = V[:, 0], V[:, 1]

# kappa and f now calculate their values at the edge midpoints when fed the triangle
def kappa(x, y):
    xi = (x + np.roll(x,-1))/2
    yi = (y + np.roll(y,-1))/2
    res = np.ones(3)
    res[(xi**2 + yi**2)**0.5 > 0.25] = 1
    return res[:, na, na]

def fq(x, y, J):
    xi = x.copy()
    yi = y.copy()
    xi[0],yi[0] = J@[2/3,1/6]+[x[0],y[0]]
    xi[1],yi[1] = J@[1/6,2/3]+[x[0],y[0]]
    xi[2],yi[2] = J@[1/6,1/6]+[x[0],y[0]]
    return -np.sin(np.pi*xi)*np.sin(np.pi*yi)
    
def fv(xi, yi):
    return -np.sin(np.pi*xi)*np.sin(np.pi*yi)
    
def fm(x, y):
    xi = (x + np.roll(x,-1))/2
    yi = (y + np.roll(y,-1))/2
#    res = np.zeros(3)
#    res[(xi**2 + yi**2)**0.5 <= 0.25] = 25
    #return res
    return -np.sin(np.pi*xi)*np.sin(np.pi*yi)
    
def g(x):
    return 0 #10 * (1 - x**2)

# Quadratic basis functions
def basis2(x,y):
    return np.array([(1-x-y)*(1-2*x-2*y), x*(2*x-1), y*(2*y-1), 4*x*(1-x-y), 4*x*y, 
                     4*y*(1-x-y)])

# Step 3
# With a quadratic basis, the gradient is no longer constant, we need to calulate it
# at the quadrature points
def dbasis2(x,y):
    return np.array([[4*x+4*y-3, 4*x-1, 0, -8*x-4*y+4, 4*y, -4*y],
                     [4*x+4*y-3, 0, 4*y-1, -4*x, 4*x, -4*x-8*y+4]])

# Quadrature nodes, xi, at midpoints of unit triangle edges
# Evaluate the quadratic basis and gradient of basis at quad points
dbasis2E = [dbasis2(0.5,0), dbasis2(0.5,0.5), dbasis2(0,0.5)]
basis2E = np.array([basis2(0.5,0), basis2(0.5,0.5), basis2(0,0.5)])#([basis2(2/3,1/6), basis2(1/6,2/3), basis2(1/6,1/6)])

AA = np.zeros((ne, 36))
IA = np.zeros((ne, 36))
JA = np.zeros((ne, 36))
bb = np.zeros((ne, 6))
ib = np.zeros((ne, 6))
jb = np.zeros((ne, 6))

from timeit import default_timer as timer
start = timer()
for ei in range(0, ne):
    # Step 1
    K = E[ei, :]
    vertx = X[K[0:3]]
    verty = Y[K[0:3]]
    [x0, x1, x2] = vertx
    [y0, y1, y2] = verty

    # Step 2
    J = np.array([[x1 - x0, x2 - x0],
                  [y1 - y0, y2 - y0]])
    invJ = la.inv(J.T)
    detJ = la.det(J)
    # Step 3 (already done above)
    # Step 4
    dphi = np.swapaxes(invJ.dot(dbasis2E), 0, 1)

    # Step 5
    Aelem = (detJ/6) * np.sum(kappa(vertx,verty) * np.swapaxes(dphi,1,2)@dphi, axis=0)

    # Step 6
    belem = (detJ/6) * fm(vertx, verty) @ basis2E

    # Step 7
    AA[ei, :] = Aelem.ravel()
    IA[ei, :] = np.repeat(K, 6)
    JA[ei, :] = np.tile(K,(6,1)).ravel()
    bb[ei, :] = belem.ravel()
    ib[ei, :] = K
    jb[ei, :] = 0
end = timer()
print(end - start)      
    
A = sparse.coo_matrix((AA.ravel(), (IA.ravel(), JA.ravel())))
A = A.tocsr()
A = A.tocoo()
b = sparse.coo_matrix((bb.ravel(), (ib.ravel(), jb.ravel())))
b = b.tocsr()
b = np.array(b.todense()).ravel()

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
#ax.plot_trisurf(X, Y, u, cmap=plt.cm.jet, linewidth=0.2)
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

tt=0.050643*fm(X[E[:,0:2]],Y[E[:,0:2]])-u[E[:,3:6]]
ttt=(detJm/4)*np.sum(tt**2,axis=1)
print(np.sqrt(np.sum(ttt)))