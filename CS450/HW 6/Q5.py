import numpy as np
import matplotlib.pyplot as plt
   
def f(x,c):
    return c*np.exp(-5*x)
    
t = np.linspace(0, 1, 101)
for c in np.arange(0,2,0.2):
    plt.plot(t,f(t,c),'b')