from dolfin import *
import numpy as np
import time
t0 = time.time()

def boundaryAll(x):
    p1 = (abs(x[0]+1.0) < DOLFIN_EPS)
    p2 = (abs(x[0]-1.0) < DOLFIN_EPS)
    p3 = (abs(x[1]+1.0) < DOLFIN_EPS)
    p4 = (abs(x[1]-1.0) < DOLFIN_EPS)
    p5 = (abs(x[0]) < DOLFIN_EPS) & ((-1<=x[1])|(x[1]<=0))
    p6 = (abs(x[1]) < DOLFIN_EPS) & ((0<=x[0])|(x[0]<=1))
    return p1 or p2 or p3 or p4 or p5 or p6


def boundaryNorthSouth(x):
    p3 = (abs(x[1]+1) < DOLFIN_EPS)
    p4 = (abs(x[1]-1.0) < DOLFIN_EPS)
    p6 = (abs(x[1]) < DOLFIN_EPS) & ((0<=x[0])|(x[0]<=1))
    return p3 or p4 or p6


def boundaryWestEast(x):
    p1 = (abs(x[0]+1) < DOLFIN_EPS)
    p2 = (abs(x[0]-1.0) < DOLFIN_EPS)
    p5 = (abs(x[0]) < DOLFIN_EPS) & ((-1<=x[1])|(x[1]<=0))
    return p1 or p2 or p5


def LSop(q, u):
    a1 = q - grad(u)
    a2 = -div(q)
    a3 = curl(q)
    return [a1, a2, a3]


# # Pick the mesh and the space
mesh = Mesh('lshape.xml')
mesh = refine(mesh)
mesh = refine(mesh)
mesh = refine(mesh)

pV = 3
Vv = VectorElement('CG', mesh.ufl_cell(), pV)
Vs = FiniteElement('CG', mesh.ufl_cell(), pV)
V = FunctionSpace(mesh, Vv * Vs)
(q, u) = TrialFunctions(V)
(psi, phi) = TestFunctions(V)


# # Set the weak form and boundary conditions
g = Constant(0)

Lu = LSop(q, u)
Lv = LSop(psi, phi)
a = dot(Lu[0], Lv[0])*dx + Lu[1]*Lv[1]*dx + dot(Lu[2], Lv[2])*dx
L = Lv[1]*dx

# boundary conditions:
# u = 0
bc_qx = DirichletBC(V.sub(0).sub(0), g, boundaryNorthSouth)
bc_qy = DirichletBC(V.sub(0).sub(1), g, boundaryWestEast)
bc_u = DirichletBC(V.sub(1),        g, boundaryAll)
bc = [bc_u]#bc_qx, bc_qy, bc_u]


# # Assemble and solve
A = assemble(a)
rhs = assemble(L)
for condition in bc:
    condition.apply(A, rhs)

QU = Function(V)
solve(A, QU.vector(), rhs)

(Q, U) = QU.split()

#plot(U, interactive=True)

class MyF(Expression):
    def __init__(self, element):
        self.element = element

    def eval(self, value, x):
        r = np.sqrt(x[0]**2 + x[1]**2)
        theta = np.arctan2(x[1],x[0])
        if theta < 0.0:
            theta += 2 * np.pi
        value[0] = r**(2.0/3) * np.sin(2 * theta / 3.0)
    
    def value_shape(self):
        return (1,)
    
class Weight(Expression):
    def __init__(self, element):
        self.element = element

    def eval(self, value, x):
        value[0] = np.exp(-(x[0]**2/(2*0.005**2) + x[1]**2/(2*0.005**2)))
    
    def value_shape(self):
        return (1,)

VV = FunctionSpace(mesh, 'Lagrange', 3)
ff = MyF(element=VV.ufl_element())
wgt = Weight(element=VV.ufl_element())
fv = project(ff, VV)
wgtv = project(wgt, VV)

print(errornorm(fv, U, norm_type='L2'))

err = U*wgtv-fv*wgtv
print(assemble(err**2*dx(mesh)))
print(time.time() - t0)
