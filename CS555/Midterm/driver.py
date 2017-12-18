import mesh_neu
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import vtk_writer
import refine_mesh


def checkorientation(V, E):
    A = np.ones((E.shape[0],3,3))
    A[:,:,1] = V[E,0]
    A[:,:,2] = V[E,1]
    return np.linalg.det(A)

# read mesh
V, E = mesh_neu.read_neu('neweire.neu')
# refine mesh
V, E = refine_mesh.refine2dtri(V, E)

sgn = checkorientation(V, E)
I = np.where(sgn<0)[0]
E1 = E[I,1]
E2 = E[I,2]
E[I,2] = E1
E[I,1] = E2

ne = E.shape[0]
nv = V.shape[0]
X = V[:,0]
Y = V[:,1]

na = np.newaxis
L = np.diff(np.r_['1',V[E], V[E[:,na,0]]],axis=1).T # components of side length
nlength = np.sqrt(np.sum(L**2,axis=0)) # side length
nx = L[1,:,:]
ny = -L[0,:,:]
nx = nx / nlength
ny = ny / nlength
h = sgn[na,:]/nlength # sgn = 2*area
hinv = 1.0 / h
center = np.sum(V[E],axis=1)/3

# construct vertex to vertex graph
ID = np.kron(np.arange(0, ne), np.ones((3,)))
G = sparse.coo_matrix((np.ones((ne*3,)), (E.ravel(), ID,)))
E2E = G.T * G
V2V = G * G.T

Enbrs = -np.ones((ne,3), dtype=int)
for i in range(ne):
    vi = E[i, :]
    nbrids = np.where(E2E[i, :].data == 2)[0]
    nbrs = E2E[i, :].indices[nbrids]
    # for each nbr, find the face it goes with
    for j in nbrs:
        vj = E[j, :]
        # edge 0
        if (vi[0] in vj) and (vi[1] in vj):
            Enbrs[i, 0] = j
        # edge 1
        if (vi[1] in vj) and (vi[2] in vj):
            Enbrs[i, 1] = j
        # edge 2
        if (vi[2] in vj) and (vi[0] in vj):
            Enbrs[i, 2] = j

xx = np.sum(V[E,0],axis=1)/3
yy = np.sum(V[E,1],axis=1)/3
# set initial values
u = np.zeros(ne)
v = np.zeros(ne)
#p = np.prod(np.cos(np.pi * np.sum(V[E],axis=1)/3),axis=1) # analytical
p = np.exp(-((xx-200)**2 + (yy+400)**2)/10000) # pulse

p[np.where(np.abs(p)<1e-15)[0]] = 0.0  # trim small values for Paraview
vtk_writer.write_basic_mesh(V, E, cdata=p, mesh_type='tri')

mapR = Enbrs.T
# find boundary elements and set mapR
ids = np.where(mapR.ravel()==-1)[0]
r, c = np.where(mapR==-1)
mapR = mapR.ravel()
mapR[ids] = c
mapR = mapR.reshape((3,ne))
# set boundary
mapB = ids.copy()
vmapB = c
# set mapL to be "this"
mapL = np.outer(np.ones((3,), dtype=int), np.arange(0,ne, dtype=int))

# set the time step
dt = 0.25 * h.min()
t = 0
# set the number of time steps
nstep=int(450/dt)-1
errlog = np.linspace(0.032,3.2,100)
un = []
pn = []
pnr = []
i = 0
logger = False
for tstep in np.arange(nstep):
    
    print("tstep %d of %d" % (tstep,nstep))
    uL = u[mapL]
    uR = u[mapR]
    vL = v[mapL]
    vR = v[mapR]
    pL = p[mapL]
    pR = p[mapR]
    
    # set the boundary conditions
    shp = uR.shape
    uR = uR.ravel()
    uR[mapB] = -uL.ravel()[mapB]
    uR = uR.reshape(shp)
    
    vR = vR.ravel()
    vR[mapB] = -vL.ravel()[mapB]
    vR = vR.reshape(shp)
    
    pR = pR.ravel()
    pR[mapB] = pL.ravel()[mapB]
    pR = pR.reshape(shp)
    
    # intermediate common flux variable
    s = (nx*(uR-uL) + ny*(vR-vL) - (pR-pL)) / h
    
    # set the update
    u = u + dt * np.sum(nx*s,axis=0)
    v = v + dt * np.sum(ny*s,axis=0)
    p = p + dt * np.sum(-1*s,axis=0)
    t = t+dt
    
    if logger & (abs(t-errlog[i])<1e-10):
        i +=1
        ut = np.sin(np.pi*np.sqrt(2)*(t)) * np.sin(np.pi*xx) * np.cos(np.pi*yy) / np.sqrt(2)
        pt = (np.cos(np.pi*np.sqrt(2)*(t)) * np.prod(np.cos(np.pi * np.sum(V[E],axis=1)/3),axis=1))
        ue = u - ut
        pe = p - pt
        un.append(np.sqrt(np.sum((sgn/2)*ue**2)))
        pn.append(np.sqrt(np.sum((sgn/2)*pe**2)))
        pnr.append(np.sqrt(np.sum((sgn/2)*(pe/pt)**2)))
    
    if (tstep % 1000) == 0:
        print('File written')
        p[np.where(np.abs(p)<1e-15)[0]] = 0.0
        vtk_writer.write_basic_mesh(V, E, cdata=p, mesh_type='tri', fname='p%04d.vtu'%(tstep,))

# Filter out spurious relative error spikes
if logger:
    from scipy import signal
    b, a = signal.butter(1, 0.03)
    y1 = signal.filtfilt(b, a, pnr)
    pnrt = np.minimum(pnr,y1*1.2)
    y2 = signal.filtfilt(b, a, pnrt)
    plt.plot(y1)
    plt.plot(y2)
    plt.plot(pnr)

#How the convergence plot was generated (from each meshes data that was saved between runs)
#plt.plot(-np.polyfit(np.log10([226,904,4096,16384,65536]),np.log10(np.vstack([pf1,pf1b,pf2,pf2b,pf3])),1)[0])