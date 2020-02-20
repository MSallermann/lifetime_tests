import matplotlib.pyplot as plt
import numpy as np 
import math 

# plt.hist(life_times, 30, density=True)
# plt.show()

data = np.loadtxt("data.txt")
plt.axhline(2*0.0752640745962459, ls = "--", color="black", label = "HTST")
plt.errorbar(data[:,0], 1/data[:,1], yerr = data[:,2]/data[:,1]**2, marker = "o", label="dynamics")
plt.ylabel("Rate [1/ps]")
plt.ylim(bottom=0)
plt.xlabel(r"Damping $\alpha$")
plt.legend()
plt.show()