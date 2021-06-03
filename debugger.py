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
# plt.plot(par.grid_m,sol.h[0])
# plt.scatter(par.grid_m,sol.c[1])

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1,1,1)

ax.plot(model.sol_egm.m[0,:],model.sol_egm.c[0,:], label= r'Not having a house', linestyle = '-', color = '0.4')
ax.plot(model.sol_egm.m[1,:],model.sol_egm.c[1,:], label= r'Having a house', linestyle = '--', color = '0.4')

plt.show()