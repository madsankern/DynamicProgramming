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
from model_1d import model_1d
import model_2
import vfi_2
import egm
import fd

model = model_1d()
model.setup()
model.create_grids()
model.solve_egm()


#par = model_2.setup()

# egm = egm.solve_EGM_2d(par)
#sol_vfi = vfi_2.solve_VFI_2dfull(par)
#sol_vfi = vfi.solve_VFI_2dfull_NELDER(par)
# fd = fd.solve_fd(par)

# # Plotting
# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(1,1,1)

# ax.plot(vfi.a, vfi.c[0,:], linestyle = ':', color = 'red', label = '$y_1$')
# ax.plot(vfi.a, vfi.c[1,:], linestyle = ':', color = 'blue', label = '$y_2$')
# ax.set_xlim([-1,20])
# ax.legend(frameon=True)

# plt.plot()

# #Plot some stuff
# fig = plt.figure(figsize=(14,5))
# ax = fig.add_subplot(1,2,1)
# ax.plot(sol_vfi.a, sol_vfi.c[0,:], linestyle = ':', color = 'red', label = '$y_1$')
# ax.plot(sol_vfi.a, sol_vfi.c[1,:], linestyle = ':', color = 'blue', label = '$y_2$')
# ax.plot(sol_vfi.a[:10], sol_vfi.a[:10], linestyle = '--', color = '0.6') # Check with 45 degree line. Seems correct
# ax.set_xlabel(f"Assets, $a_t$")
# ax.set_ylabel(f"Consumption, $c^\star_t$")
# ax.set_title(f'Policy function')
# ax.set_xlim([-1,20])
# ax.legend(frameon=True)
# plt.show()

