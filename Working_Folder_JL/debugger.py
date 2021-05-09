# load general packages
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# load modules related to this exercise
import tools 
from model_dc import model_dc

model = model_dc()
model.setup()
model.create_grids()
model.solve()

par = model.par
sol = model.sol

model = model_dc()
model.setup()
model.create_grids()
model.solve()

par = model.par
sol = model.sol

fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1,2,1)

t = 1
state = 0
ax.plot(sol.m[t,state,:],sol.c[t,state,:])
ax.set_xlim(0,15)
ax.set_ylim(0,5)

plt.show()

# Define figures

# Retired
def figure(par,sol,z):
    if z == 1:
        print(f'Retired in t+1')
        ts = [par.T, par.T-1, par.T-2, par.T-3, 1]
    elif z ==0:
        print(f'Working in t+1')
        ts = [par.T-4]
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,2,1)
    for i in ts:
        ax.scatter(sol.m[i-1,z,:],sol.c[i-1,z,:], label=f't = {i}')
    ax.set_xlabel(f"$m_t$")
    ax.set_ylabel(f"$c(m_t,z_{{t+1}} = {z})$")
    ax.set_xlim([0, 5])
    ax.set_ylim([0,3])
    ax.set_title(f'Consumption function')
    plt.legend()


    ax_right = fig.add_subplot(1,2,2)
    for i in ts:
        ax_right.scatter(sol.m[i-1,z,:],sol.v[i-1,z,:], label=f't = {i}')
    ax_right.set_xlabel(f"$m_t$")
    ax_right.set_ylabel(f"$v(m_t,z_{{t+1}} = {z})$")
    ax_right.set_xlim([0, 5])
    ax_right.set_ylim([-20,0])
    ax_right.set_title(f'Value function')
    plt.legend()

    plt.show()