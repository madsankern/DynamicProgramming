####################
## Debugging file ##
####################
# Debugging file where we can run all our solvers without going through the notebook.
# Call any solver you want to check externally here

# Imports and settings
#%load_ext autoreload
#%autoreload 2
#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

import tools
import model
import vfi
import egm
import fd

par = model.setup()

# egm = egm.solve_EGM_2d(par)
sol_vfi = vfi.solve_VFI_2dfull(par)
# fd = fd.solve_fd(par)

# # Plotting
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(1,1,1)

# ax.plot(vfi.a, vfi.c[0,:], linestyle = ':', color = 'red', label = '$y_1$')
# ax.plot(vfi.a, vfi.c[1,:], linestyle = ':', color = 'blue', label = '$y_2$')
# ax.set_xlim([-1,20])
# ax.legend(frameon=True)

# plt.plot()