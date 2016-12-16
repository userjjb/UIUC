#Multilevel summation method simple demonstration code with 2 grid level hierarchy
#Attempts to calculate electric potential for gaussian distribution of charges
import numpy as np
import scipy as sp
import scipy.interpolate as interp
import matplotlib.pyplot as plt

na = np.newaxis

n = 1024 #2**even
nh = int(np.sqrt(n)/2) #number of coarse grid squares on a side
A = 4/nh #cutoff length

px=np.array([]); py=px.copy() #Construct set of random bodies, roughly evenly
for i in np.arange(0,1,1/nh):
    for j in np.arange(0,1,1/nh):
        px = np.r_[px,i+np.random.rand(4)/nh]
        py = np.r_[py,j+np.random.rand(4)/nh]

pr = np.c_[px,py]

q = np.exp(-( (px-0.5)**2/(2*0.15**2) + (py-0.5)**2/(2*0.15**2) )) #Gaussian strength
#q = np.random.rand(n)

h = 16+2j #Coarse grid size
h_x, h_y = np.mgrid[0:1:h, 0:1:h] #coarse grid points
qh = interp.griddata(pr, 4*q, (h_x,h_y), method='cubic', fill_value=0) #interpolated coarse grid charges
Kh = np.sqrt((h_x.ravel()[:,na]-h_x.ravel())**2 + (h_y.ravel()[:,na]-h_y.ravel())**2) #interpolated coarse kernel values
np.fill_diagonal(Kh, 0) #remove self-influence

Uh = Kh@qh.ravel() #Calculate coarse grid potential contribution

#Repeat for coarsest grid
#Note that 2A is large enough to cover whole domain, even coarser grid would be a waste
h2 = 8+2j
h2_x, h2_y = np.mgrid[0:1:h2, 0:1:h2]
qh2 = interp.griddata(pr, 16*q, (h2_x,h2_y), method='cubic', fill_value=0)
Kh2 = np.sqrt((h2_x.ravel()[:,na]-h2_x.ravel())**2 + (h2_y.ravel()[:,na]-h2_y.ravel())**2)

Uh2 = Kh2@qh2.ravel()

Uh2qh = interp.griddata(np.c_[h2_x.ravel(),h2_y.ravel()], Uh2, (h_x,h_y), method='cubic', fill_value=0) #Coarsest grid onto coarse grid

#For simplicity generate full O(n^2) kernel for comparison, then construct MSM K0 from this
K = 1/np.sqrt((px[:,na]-px)**2 +(py[:,na]-py)**2)
np.fill_diagonal(K, 0)

#Select nearby exact kernel values for particle level computations
#In practice this would be very expensive, but for simplicity/demonstration it suffices
K0 = np.zeros([n,n])
for i in range(0,nh*4):
    for j in range(0,nh):
        #xs and ys are the coarse grid squares with A range of each target
        xs = i+(np.arange(-4,5)*nh*4-np.arange(4)[:,na])
        xs = xs[xs>=0]
        xs = xs[xs<n]
        ys = j+np.arange(-16,17)
        ys = ys[ys>=0]
        ys = ys[ys<n]
        xsm,ysm = np.meshgrid(xs,ys)
        K0[i*(nh)+j,xsm.ravel()*3+ysm.ravel()*2] = K[xsm.ravel(),ysm.ravel()]

U = K@q #Exact full pair-pair interactions for comparison
U0 = K0@q
a,b = np.mgrid[0:1:200j, 0:1:200j]
#Exact solution for potential, cap potential display maximums to a "good" value to minimize local clustering from dominating plot
Ug = interp.griddata(pr, np.minimum(U,np.mean(U)+2*np.std(U)), (a,b), method='cubic', fill_value=0)
#Interpolate particle, coarse grid, and coarsest grid particle values to a grid for easier plotting
Ug0 = interp.griddata(pr, U0, (a,b), method='cubic', fill_value=0)
Ug1 = interp.griddata(np.c_[h_x.ravel(),h_y.ravel()], Uh.ravel(), (a,b), method='cubic', fill_value=0)
Ug2 = interp.griddata(np.c_[h_x.ravel(),h_y.ravel()], Uh2qh.ravel(), (a,b), method='cubic', fill_value=0)

#MSM calculated values (scaled by a constant C and offset)
plt.imshow((Ug0/16-Ug1*2-Ug2*4), extent=(0,1,0,1),origin='lower')
plt.show()
#Pairwise exact calculated values
plt.imshow(Ug/1000, extent=(0,1,0,1),origin='lower')
plt.colorbar()
plt.show()

plt.spy(K0) #Sparsity pattern for nearby particle interactions at particle level, "kinda" sparse