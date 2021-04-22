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

# # Check iterations with VFI
# print("Using value function iteration required " + str(sol_vfi.it) +" iterations before convergence")

# # Solve simple model EGM
# sol_egm = egm.solve_EGM(par)

# # Check iterations with EGM
# print("Using value function iteration required " + str(sol_egm.it) +" iterations before convergence")

# # Plot policy functions with VFI and EGM
# fig = plt.figure(figsize=(14,5))

# ax = fig.add_subplot(1,2,1)

# ax.plot(sol_egm.a, sol_egm.c, linestyle = ':', color = '0.4')
# ax.plot(sol_vfi.a, sol_vfi.c, linestyle = '-', color = '0.7')
# ax.set_xlabel(f"Assets, $a_t$")
# ax.set_ylabel(f"Consumption, $c^\star_t$")
# ax.set_title(f'Policy function')
# ax.set_xlim([-1,20])

# plt.show()

# Extended model with Markov switching
sol_egm_2d = egm.solve_EGM_2d(par)

# Plot policy functions with EGM for Markov switching model
