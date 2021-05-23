import numpy as np
import matplotlib.pyplot as plt
import tools 
from model import model_class as model

# Setup model
model = model()
model.setup()
model.create_grids()

# Solve using NEGM
model.solve_egm()
sol = model.sol_egm
par = model.par

# Plot
plt.plot(par.grid_m,sol.h[0])
plt.scatter(par.grid_m,sol.c[1])
