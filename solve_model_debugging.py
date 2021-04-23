# load packages
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# load user written modules
import tools
import model
import vfi
import egm

# Setup
par = model.setup()

# # Solve simple model with VFI
# sol_vfi = vfi.solve_VFI(par)

# # Solve simple model EGM
# sol_egm = egm.solve_EGM(par)

# Extended model with Markov switching
sol_egm_2d = egm.solve_EGM_2d(par)

