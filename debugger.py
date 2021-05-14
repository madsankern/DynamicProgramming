import numpy as np
import matplotlib.pyplot as plt
import tools 
from model_1d import model_1d

# Setup model
model = model_1d()
model.setup()
model.create_grids()

# Solve using NEGM
model.solve_negm_dc()
sol = model.sol_negm
par = model.par

# Plot
plt.plot(par.grid_m,sol.h[0])
plt.scatter(par.grid_m,sol.c[1])
