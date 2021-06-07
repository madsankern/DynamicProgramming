import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time
import tools as tools

# Local modules
import utility as util
import LCP_solver

##############
## FD - LCP ##
##############
# Solve the discrete-continuous problem using linear complementarity
# ToDo: add timing

s = 2 # Elasticity of substitution 
r = 0.045 # Rate of return
rho = 0.05 # Discount rate
y = 0.5 # Income
kappa = 0.25 # Housing utility
p0 = 2.0 #buying price
p1 = 1.7 #selling price

Delta = 1000 # Stepsize
delta = 1000 # Distance between iterations - MADS

I = 500 # Gridpoints on asset grid - set to 500 normally
amin = 1e-8 # Minimum assets
amax = 10 # Maximum assets
a = np.linspace(amin,amax,I).transpose() # a grid
da = (amax-amin)/(I-1) # Stepsize for a

maxit = 500 # Max number of iterations
crit = 1e-6 # Stopping criteria
it = 0

# Initialize
dVf = np.zeros((I,2)) 
dVb = np.zeros((I,2))
c = np.zeros((I,2))
Vstar = np.zeros((I,2))
u= np.zeros((I,2))
A = sparse.eye(2*I, format = 'csr') # Transition matrix

# Double grid of assets
aa = np.tile(np.array([a]), (2,1))
aa = aa.transpose() # Write this better
yy = y*np.ones((I,2)) # 2 dimensional income grid

# INITIAL GUESS
v0 = np.power((yy + r*aa), 1-s)/((1-s)*rho) # value of initial guess from HJB, double check
v = v0

## RUN LOOP FROM HERE ##

t0 = time.time()
while (delta >= crit and it < maxit):

    # Redefine
    V = v

    # Forward difference

    dVf[:I-1,:] = (V[1:I,:] - V[0:I-1,:])/da
    dVf[I-1,:] = np.tile(np.power((y + r*amax),-s),(1,2)) #will never be used, but impose state constraint a<=amax just in case

    # Backward difference
    dVb[1:,:] = ((V[1:I,:] - V[0:I-1,:]))/da
    dVb[0,:] = np.tile(np.power((y + r*amin),-s),(1,2)) # state constraint boundary condition


    #consumption and savings with forward difference
    cf = np.power(dVf,-1/s) # Take inverse marginal utility of value function (c foc)
    ssf = yy + r*aa - cf # Savings from forward difference

    #consumption and savings with backward difference
    cb = np.power(dVb, -1/s)
    ssb = yy + r*aa - cb


    #consumption and derivative of value function at steady state
    c0 = yy + r*aa

    # Upwind method makes a choice of forward or backward differences based on the sign of the drift    
    If = ssf > 0 #positive drift --> forward difference
    Ib = ssb < 0 #negative drift --> backward difference
    I0 = (1-If-Ib) #at steady state

    c = cf*If + cb*Ib + c0*I0

    # Use util function for this later
    u[:,0] = np.power(c[:,0], (1-s))/(1-s) # Utility without car
    u[:,1] = np.power(c[:,1], (1-s))/(1-s) + kappa # Utility with car

    # Terms to be put into the A matrix
    X = -np.minimum(ssb,0)/da
    Y = np.maximum(ssf,0)/da + np.minimum(ssb,0)/da
    Z = np.maximum(ssf,0)/da
    Z_helper = np.vstack((np.array([0,0]), Z))

    # Construct A matrix
    A1 = sparse.eye(I, format = 'csr')*0 # Intialize sparse zero mat
    A1 += sparse.spdiags(Y[:,0],0,I,I)
    A1 += sparse.spdiags(X[1:,0],-1,I,I)
    A1 += sparse.spdiags(Z_helper[:I-2,0],1,I,I)

    A2 = sparse.eye(I, format = 'csr')*0 # Intialize sparse zero mat
    A2 += sparse.spdiags(Y[:,1],0,I,I)
    A2 += sparse.spdiags(X[1:,1],-1,I,I)
    A2 += sparse.spdiags(Z_helper[:I-2,1],1,I,I)

    # Stack A1 and A2
    block_helper = sparse.eye(I, format = 'csr')*0
    A_upper = sparse.hstack([A1,block_helper], format = 'csr')
    A_lower = sparse.hstack([block_helper,A2], format = 'csr')
    A = sparse.vstack([A_upper,A_lower], format = 'csr') # This is the transition matrix used for the problem

    # B matrix
    B = (rho + 1/Delta)*sparse.eye(2*I) - A

    # Stack utility vector into one long column vector
    u_stacked = np.hstack(([u[:,0]], [u[:,1]])).transpose()
    V_stacked = np.hstack(([V[:,0]], [V[:,1]])).transpose()

    # Outside option
    i_buy = np.ceil(p0/da) # p0 equals i_buy grid points - the points where one can buy the car
    i_buy = int(i_buy)
    # Value of buying car if currently, don't own car d=0
    Vstar[i_buy:I,0] = V[:I - i_buy,1] # sort of 'upper envelope step'

    # Instead of setting Vstar(1:i_buy)=-Inf, do something smoother - to ensure not buying when cannot afford
    slope = (Vstar[i_buy+1,0]-Vstar[i_buy,0])/da
    Vstar[:i_buy,0] = Vstar[i_buy+1,0] + slope*(a[0:i_buy] - a[i_buy+1])

    # Value of selling car if currently own car, d=1
    i_sell = np.ceil(p1/da) #p1 equals i_sell grid points
    i_sell = int(i_sell)
    Vstar[0:I-i_sell,1] = V[i_sell:I,0]
    Vstar[I-i_sell:,1] = V[I-1,0] #assume p = min(p,amax - a), i.e. selling car cannot take you above amax

    # Stack the two optimal value functions

    Vstar_stacked = np.hstack(([Vstar[:,0]], [Vstar[:,1]])).transpose()

    # Parameters for the LPC solver
    vec = u_stacked + V_stacked/Delta
    q = -vec + B*Vstar_stacked

    # using Yuval Tassa's Newton-based LCP solver, download from http://www.mathworks.com/matlabcentral/fileexchange/20952
    z0 = V_stacked - Vstar_stacked
    l = np.zeros(2*I) + 1e-6 # added small term for numerics
    u_ = np.inf*np.ones(2*I)

    # Seems to work now
    z = LCP_solver.LCP_python(B,q,l,u_,z0,0)

    # LCP_error = np.max(abs(z*(B*z + q)))

    # if LCP_error > 1e-5:
    #      print('LCP not solved, Iteration =')
    #      print(n)
    #      break

    V_stacked = z + Vstar_stacked  # calculate value function
    V = np.hstack([V_stacked[:I], V_stacked[I:2*I]])

    Vchange = V - v
    v = V
    # dist = np.amax(np.absolute(Vchange[:,0]))
    # if dist < crit:
    #     print('Value Function Converged, Iteration = ')
    #     print(it)
    #     break

    it += 1
t1 = time.time()

sol_time = t1 - t0

## SAVE RESULTS ##
sol_c = c[1:-1,:].copy()
sol_m = a[1:-1].copy()

## PLOT RESULTS ##
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

# Consumption
fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(1,1,1)

ax.grid(b=True, which = 'major', linestyle='-', linewidth=0.5, color='0.9')
ax.set_xlim([0.0,10])
ax.set_ylim([0.5,1.0])
ax.set_xlabel(r'Cash on Hand, $m_t$', size=13)
ax.set_ylabel(r'Consumption, $c_t$', size=13)

ax.plot(a[1:-1],c[1:-1,0], label= r'Not having a house', linestyle = '-', color = '0.4')
ax.plot(a[1:-1],c[1:-1,1], label= r'Having a house', linestyle = '--', color = '0.4')

ax.legend(frameon = True, edgecolor = 'k', facecolor = 'white', framealpha=1, fancybox=False, loc = 2)
plt.savefig('figs/fd_dc_policy.pdf')

# Value function
fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(1,1,1)

ax.grid(b=True, which = 'major', linestyle='-', linewidth=0.5, color='0.9')
ax.set_xlim([0.0,10])
ax.set_ylim([-40,-15])
ax.set_xlabel(r'Cash on Hand, $m_t$', size=13)
ax.set_ylabel(r'Consumption, $c_t$', size=13)

ax.plot(a[1:-1],v[1:-1,0], label= r'Not having a house', linestyle = '-', color = '0.4')
ax.plot(a[1:-1],v[1:-1,1], label= r'Having a house', linestyle = '--', color = '0.4')

ax.legend(frameon = True, edgecolor = 'k', facecolor = 'white', framealpha=1, fancybox=False, loc = 2)
plt.savefig('figs/fd_dc_value.pdf')

# Discrete choice
action = (np.around(V,3) == np.around(Vstar,3))

fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(1,1,1)

ax.grid(b=True, which = 'major', linestyle='-', linewidth=0.5, color='0.9')
ax.set_xlim([-1,10])
ax.set_ylim([-0.1,1.4])
ax.set_xlabel(r'Cash on Hand, $m_t$', size=13)
ax.set_ylabel(r'Consumption, $c_t$', size=13)

ax.plot(a[0:-1],action[0:-1,0], label= r'Buying a house', linestyle = '-', color = '0.4')
ax.plot(a[0:-1],action[0:-1,1], label= r'Selling a house', linestyle = '--', color = '0.4')

ax.legend(frameon = True, edgecolor = 'k', facecolor = 'white', framealpha=1, fancybox=False, loc = 2)
plt.savefig('figs/fd_dc_action.pdf')


###########################
## Run on very fine grid ##
###########################

## RUN LOOP FROM HERE ##

I = 6000 # Very fine grid
amin = 1e-8 # Minimum assets
amax = 10 # Maximum assets
a = np.linspace(amin,amax,I).transpose() # a grid
da = (amax-amin)/(I-1) # Stepsize for a

maxit = 10 # Max number of iterations
crit = 1e-6 # Stopping criteria
it = 0

# Initialize
dVf = np.zeros((I,2)) 
dVb = np.zeros((I,2))
c = np.zeros((I,2))
Vstar = np.zeros((I,2))
u = np.zeros((I,2))
A = sparse.eye(2*I, format = 'csr') # Transition matrix

# Double grid of assets
aa = np.tile(np.array([a]), (2,1))
aa = aa.transpose() # Write this better
yy = y*np.ones((I,2)) # 2 dimensional income grid

# INITIAL GUESS
v0 = np.power((yy + r*aa), 1-s)/((1-s)*rho) # value of initial guess from HJB, double check
v = v0

while (delta >= crit and it < maxit):

    # Redefine
    V = v

    # Forward difference

    dVf[:I-1,:] = (V[1:I,:] - V[0:I-1,:])/da
    dVf[I-1,:] = np.tile(np.power((y + r*amax),-s),(1,2)) #will never be used, but impose state constraint a<=amax just in case

    # Backward difference
    dVb[1:,:] = ((V[1:I,:] - V[0:I-1,:]))/da
    dVb[0,:] = np.tile(np.power((y + r*amin),-s),(1,2)) # state constraint boundary condition


    #consumption and savings with forward difference
    cf = np.power(dVf,-1/s) # Take inverse marginal utility of value function (c foc)
    ssf = yy + r*aa - cf # Savings from forward difference

    #consumption and savings with backward difference
    cb = np.power(dVb, -1/s)
    ssb = yy + r*aa - cb


    #consumption and derivative of value function at steady state
    c0 = yy + r*aa

    # Upwind method makes a choice of forward or backward differences based on the sign of the drift    
    If = ssf > 0 #positive drift --> forward difference
    Ib = ssb < 0 #negative drift --> backward difference
    I0 = (1-If-Ib) #at steady state

    c = cf*If + cb*Ib + c0*I0

    # Use util function for this later
    u[:,0] = np.power(c[:,0], (1-s))/(1-s) # Utility without car
    u[:,1] = np.power(c[:,1], (1-s))/(1-s) + kappa # Utility with car

    # Terms to be put into the A matrix
    X = -np.minimum(ssb,0)/da
    Y = np.maximum(ssf,0)/da + np.minimum(ssb,0)/da
    Z = np.maximum(ssf,0)/da
    Z_helper = np.vstack((np.array([0,0]), Z))

    # Construct A matrix
    A1 = sparse.eye(I, format = 'csr')*0 # Intialize sparse zero mat
    A1 += sparse.spdiags(Y[:,0],0,I,I)
    A1 += sparse.spdiags(X[1:,0],-1,I,I)
    A1 += sparse.spdiags(Z_helper[:I-2,0],1,I,I)

    A2 = sparse.eye(I, format = 'csr')*0 # Intialize sparse zero mat
    A2 += sparse.spdiags(Y[:,1],0,I,I)
    A2 += sparse.spdiags(X[1:,1],-1,I,I)
    A2 += sparse.spdiags(Z_helper[:I-2,1],1,I,I)

    # Stack A1 and A2
    block_helper = sparse.eye(I, format = 'csr')*0
    A_upper = sparse.hstack([A1,block_helper], format = 'csr')
    A_lower = sparse.hstack([block_helper,A2], format = 'csr')
    A = sparse.vstack([A_upper,A_lower], format = 'csr') # This is the transition matrix used for the problem

    # B matrix
    B = (rho + 1/Delta)*sparse.eye(2*I) - A

    # Stack utility vector into one long column vector
    u_stacked = np.hstack(([u[:,0]], [u[:,1]])).transpose()
    V_stacked = np.hstack(([V[:,0]], [V[:,1]])).transpose()

    # Outside option
    i_buy = np.ceil(p0/da) # p0 equals i_buy grid points - the points where one can buy the car
    i_buy = int(i_buy)
    # Value of buying car if currently, don't own car d=0
    Vstar[i_buy:I,0] = V[:I - i_buy,1] # sort of 'upper envelope step'

    # Instead of setting Vstar(1:i_buy)=-Inf, do something smoother - to ensure not buying when cannot afford
    slope = (Vstar[i_buy+1,0]-Vstar[i_buy,0])/da
    Vstar[:i_buy,0] = Vstar[i_buy+1,0] + slope*(a[0:i_buy] - a[i_buy+1])

    # Value of selling car if currently own car, d=1
    i_sell = np.ceil(p1/da) #p1 equals i_sell grid points
    i_sell = int(i_sell)
    Vstar[0:I-i_sell,1] = V[i_sell:I,0]
    Vstar[I-i_sell:,1] = V[I-1,0] #assume p = min(p,amax - a), i.e. selling car cannot take you above amax

    # Stack the two optimal value functions

    Vstar_stacked = np.hstack(([Vstar[:,0]], [Vstar[:,1]])).transpose()

    # Parameters for the LPC solver
    vec = u_stacked + V_stacked/Delta
    q = -vec + B*Vstar_stacked

    # using Yuval Tassa's Newton-based LCP solver, download from http://www.mathworks.com/matlabcentral/fileexchange/20952
    z0 = V_stacked - Vstar_stacked
    l = np.zeros(2*I) + 1e-6 # added small term for numerics
    u_ = np.inf*np.ones(2*I)

    # Seems to work now
    z = LCP_solver.LCP_python(B,q,l,u_,z0,0)

    # LCP_error = np.max(abs(z*(B*z + q)))

    # if LCP_error > 1e-5:
    #      print('LCP not solved, Iteration =')
    #      print(n)
    #      break

    V_stacked = z + Vstar_stacked  # calculate value function
    V = np.hstack([V_stacked[:I], V_stacked[I:2*I]])

    Vchange = V - v
    v = V
    # dist = np.amax(np.absolute(Vchange[:,0]))
    # if dist < crit:
    #     print('Value Function Converged, Iteration = ')
    #     print(it)
    #     break

    it += 1

######################
## Measure accuracy ##
######################

# Unpack true policy
c_true = c[5:-5,:].copy()
m_true = a[5:-5].copy()

# Interpolate on 'true' grid
c_interp_1 = tools.interp_linear_1d(sol_m, sol_c[:,0], m_true)
c_interp_2 = tools.interp_linear_1d(sol_m, sol_c[:,1], m_true)

error_1 = 100*1/2*1/6000*np.sum(np.abs(c_interp_1 - c_true[:,0])/c_true[:,0])
error_2 = 100*1/2*1/6000*np.sum(np.abs(c_interp_2 - c_true[:,1])/c_true[:,1])
error = error_1 + error_2
print(error)
