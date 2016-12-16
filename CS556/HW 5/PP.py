#Unhelpful plot of residual vs iteration
plt.figure(figsize=(8,6))
for i in range(0,10):
    plt.semilogy(range(len(resid[i])),resid[i],label=i+1)
plt.xlabel("iterations")
plt.ylabel("residual")
plt.legend()
plt.show()

#Timescaled plot of residual vs clock time
plt.figure(figsize=(8,6))
for i in range(0,10):
    plt.semilogy(np.linspace(0,elapsed[i],len(resid[i])),resid[i],label=i+1)
plt.xlabel("wall clock elapsed (s)")
plt.ylabel("residual")
plt.legend()
plt.show()

#Plot of residual after 5 iterations
plt.figure(figsize=(8,6))
plt.semilogy(range(1,11),[item[-1] for item in resid])
plt.xlabel("Number of smoothing passes (kmax=10)")
plt.ylabel("final residual after 30 seconds")

#Plot of residual after 5 iterations
plt.figure(figsize=(8,6))
plt.plot(range(1,11),np.array([item[3] for item in resid])/np.array([item[2] for item in resid]))
plt.xlabel("Numbering of smoothing passes (kmax=10)")
plt.ylabel("convergence factor at iteration 4")