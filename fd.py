import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Local modules
import utility as util
import LCP_solver

##############################
## Finite Difference method ##
##############################

def solve(par,sol):

    # Setup
    dm = (par.m_max - par.m_min)/(par.Nm - 1) # Stepsize for m grid
    y_vals = par.y # Income process
    y_size = (len(y_vals)) # Number of states in income process

    # Dimension of transition matrix
    n = par.Nm * y_size 

    # Initial guess - zero savings - add consumption explicitly
    sol.v = util.u(np.tile(par.grid_a,(y_size,1))*par.r
                    + np.tile(y_vals,(par.Nm,1)).transpose(),par)/par.rho

    # Initialize A matrix
    y_transition = sparse.kron(par.pi, sparse.eye(par.Nm), format = "csr")
    # This yields a sparse matrix containing all the lambdas in the correct position
    # for the A matrix

    # Initialize
    v_old = np.zeros((y_size, par.Nm))
    dv = np.zeros((y_size, par.Nm-1))
    c_foc = np.zeros((y_size, par.Nm-1))
    c0 = np.zeros((y_size,par.Nm))
    ssf = np.zeros((y_size,par.Nm))
    ssb = np.zeros((y_size,par.Nm))
    is_forward = np.zeros((y_size,par.Nm),'bool') # indicators
    is_backward = np.zeros((y_size,par.Nm),'bool') # indicators
    diag_helper = np.zeros((y_size,par.Nm))        
    sol.A = y_transition.copy()
    sol.B = y_transition.copy()
    # AT = y_transition.copy()

    # Intialize loop parameters
    sol.it = 0 # Iteration counter
    sol.delta = 1000.0 # Difference between iterations
    Delta = 1000 # Step size of each iteration - can be large due to implicit updating

    # Iterate the discretized HJB
    while (sol.delta >= par.tol_fd and sol.it < par.max_iter):

        dv = (sol.v[:,1:]-sol.v[:,:-1])/dm # Finite difference approx

        c_foc = np.maximum(util.inv_marg_u(dv,par),1e-4) # Optimal consumption given value function
        c0 = np.tile(par.grid_m,(y_size,1))*par.r \
                        + np.tile(y_vals,(par.Nm,1)).transpose() # Income

        # Savings decomposed into positive and negative components
        ssf[:,:-1] = c0[:,:-1] - c_foc # Backward imputation
        ssb[:,1:] = c0[:,1:] - c_foc # Forward imputation

        # Note that the boundary conditions are handled implicitly as ssf will be zero at m_max and ssb at m_min 
        is_forward = ssf > 0
        is_backward = ssb < 0

        # Compute consumption using upwind scheme
        c0[:,:-1] += (c_foc - c0[:,:-1])*is_forward[:,:-1]
        c0[:,1:] += (c_foc - c0[:,1:])*is_backward[:,1:]

        # UNCOMMENT FOR DEBUGGING
        #plt.plot(self.par.grid_m, self.c0.transpose())#
        #plt.show()

        #c0 = util.u(c0,par) # Check this out
        u_vec = util.u(c0,par)
       
        # Build the A matrix
        sol.A = y_transition.copy() # Transition matrix containing lambdas
        diag_helper = (-ssf*is_forward/dm \
                            + ssb*is_backward/dm).reshape(n) # Stack forwards/backwards indicator n times. 
                            # This is Omega without lambda_j (already there)
        sol.A += sparse.spdiags(diag_helper,0,n,n) # Ad
        diag_helper = (-ssb*is_backward/dm).reshape(n) # Upsilon
        sol.A += sparse.spdiags(diag_helper[1:],-1,n,n) # Add Upsilon ad one column before the diagonal (not in first row due to upwinding)
        diag_helper = (ssf*is_forward/dm).reshape(n) # Psi
        sol.A += sparse.spdiags(np.hstack((0,diag_helper)),1,n,n) # Add Psi to each row of A

        # Solve the system of linear equations corresponding to implicit finite difference scheme
        sol.B = sparse.eye(n)*(1/Delta + par.rho) - sol.A
        b = u_vec.reshape(n,1) + sol.v.reshape(n,1)/Delta
        v_old = sol.v.copy() # Save for comparison
        sol.v = spsolve(sol.B,b).reshape(y_size,par.Nm)

        # Check distance between previous iteration and update iteration counter
        sol.delta = np.amax(np.absolute(v_old-sol.v).reshape(n))
        sol.it += 1

    # Unpack solution - Add what is needed here
    sol.c = c0
    sol.m = par.grid_m
    sol.dv = dv

    return sol

##############
## FD - LCP ##
##############
# Solve the discrete-continuous problem using linear complementarity
# There is still a bit of work left for this

s = 2 # Elasticity of substitution 
r = 0.045 # Rate of return
rho = 0.05 # Discount rate
y = .5 # Income
kappa = 0.2 # Housing utility
p0 = 2.0 #buying price
p1 = 1.6 #selling price

I = 500 # Gridpoints on asset grid
amin = 1e-8 # Minimum assets
amax = 10 # Maximum assets
a = np.linspace(amin,amax,I).transpose() # a grid
da = (amax-amin)/(I-1) # Stepsize for a

maxit = 20 # Max number of iterations
crit = 1e-3 # Stopping criteria
it = 0

Delta = 1000 # Stepsize
delta = 1000 # Distance between iterations - MADS

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


## RUN LOOP FROM HERE ##

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
    l = np.zeros(2*I) #+ 1e-6 # added small term for numerics
    u_ = 100*np.ones(2*I) ## Upper limit on consumption

    # Seems to work now
    z = LCP_solver.LCP_python(M=B,q=q,l=l,x0=z0,display=False)

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