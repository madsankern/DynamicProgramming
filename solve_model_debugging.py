####################
## Debugging file ##
####################
# Debugging file where we can run all our solvers without going through the notebook.
# Call any solver you want to check externally here

# Load packages
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# load user written modules
import tools
import model
import vfi
import egm

# Import paramteters
par = model.setup()

egm = egm.solve_EGM_2d(par)

vfi = vfi.solve_VFI_2d(par)

fd = fd.solve_fd(par)
