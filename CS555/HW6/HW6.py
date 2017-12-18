import numpy as np
from dolfin import *
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(4, 4)


def boundary(x):
    return abs(x[0]) < DOLFIN_EPS\
        or abs(x[0]-1.0) < DOLFIN_EPS\
        or abs(x[1]) < DOLFIN_EPS\
        or abs(x[1]-1.0) < DOLFIN_EPS


alpha = -1
x0 = 1./3
y0 = 1./3

k = 4
errsH0 = []
errsH1 = []
hs = []
for i in range(0, k):
    mesh = refine(mesh)
    V = FunctionSpace(mesh, "Lagrange", 2)
    f = Expression('pow(pow(x[0]-x0,2) + pow(x[1] -y0,2),alpha)',
                   x0=x0, y0=y0, alpha=alpha, element=V.ufl_element())

    bc = DirichletBC(V, 0.0, boundary)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = f*v*dx

    uh = Function(V)
    solve(a == L, uh, bc)

    meshfine = refine(refine(mesh))
    Vfine = FunctionSpace(meshfine, "Lagrange", 2)
    f = Expression('pow(pow(x[0]-x0,2) + pow(x[1] -y0,2),alpha)',
                   x0=x0, y0=y0, alpha=alpha, element=Vfine.ufl_element())
    bc = DirichletBC(Vfine, 0.0, boundary)
    u = TrialFunction(Vfine)
    v = TestFunction(Vfine)
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx
    ue = Function(Vfine)
    solve(a == L, ue, bc)

    EH0 = errornorm(ue, uh, norm_type='L2')
    EH1 = errornorm(ue, uh, norm_type='H1')
    errsH0.append(EH0)
    errsH1.append(EH1)
    hs.append(mesh.hmax())

errsH0 = np.array(errsH0)
errsH1 = np.array(errsH1)
hs = np.array(hs)
rH0 = np.log(errsH0[1:] / errsH0[0:-1]) / np.log(hs[1:] / hs[0:-1])
rH1 = np.log(errsH1[1:] / errsH1[0:-1]) / np.log(hs[1:] / hs[0:-1])

#polyorder = 2
#plt.loglog(hs,errsH0,'o-',label='$H^0$ error' )
#plt.loglog(hs,errsH1,'o-',label='$H^1$ error')
#plt.loglog(hs,2.5*hs,label='$O(h)$')
#plt.loglog(hs,0.1*hs**2,label='$O(h^2)$')
#
#plt.loglog(hs,0.06*hs**polyorder,label='$O(h^{%d})$'%polyorder)
#z = polyorder + 1
#plt.loglog(hs,0.005*hs**(polyorder+1),label='$O(h^{%d})$'%z)
#
#plt.legend(loc=0)
#plt.xlabel('$h_{max}$')
#plt.ylabel('Error')
#plt.title('$alpha$ = %2f, polynomial order = %d' %(alpha,polyorder))
#plt.savefig('test.png')
print(np.polyfit(np.log10(hs),np.log10(errsH0),1)[0])
print(np.polyfit(np.log10(hs),np.log10(errsH1),1)[0])