##############################
## Finite Difference method ##
##############################

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import utility as util

def solve(par,sol):

    # Setup
    dm = (par.m_max - par.m_min)/(par.Nm - 1) # Stepsize for m grid
    y_vals = par.y # Income process
    y_size = (len(y_vals)) # Number of states in income process

    # Dimension of transition matrix
    n = par.Nm * y_size 

    # Initial guess on value function = Stay put (UPDATE)
    sol.v = util.u(np.tile(par.grid_a,(y_size,1))*par.r
                    + np.tile(y_vals,(par.Nm,1)).transpose(),par)/par.rho

    # Skill transition matrix
    y_transition = sparse.kron(par.pi, sparse.eye(par.Nm), format = "csr")
    # This yields a sparse matrix containing all the lambdas in the correct possition
    # for 'A' defined in the paper 

    # Initialize
    v_old = np.zeros((y_size, par.Nm))
    dv = np.zeros((y_size, par.Nm-1))
    cf = np.zeros((y_size, par.Nm-1))
    c0 = np.zeros((y_size,par.Nm))
    ssf = np.zeros((y_size,par.Nm))
    ssb = np.zeros((y_size,par.Nm))
    is_forward = np.zeros((y_size,par.Nm),'bool') # indicators
    is_backward = np.zeros((y_size,par.Nm),'bool') # indicators
    diag_helper = np.zeros((y_size,par.Nm))        
    sol.A = y_transition.copy()
    sol.B = y_transition.copy()
    # AT = y_transition.copy()

    sol.it = 0 # Iteration counter
    sol.delta = 1000.0 # Difference between iterations
    Delta = 1000 # Step size of each iteration - can be large due to implicit updating

    # Iterate the discretized HJB
    while (sol.delta >= par.tol_fd and sol.it < par.max_iter):

        dv = (sol.v[:,1:]-sol.v[:,:-1])/dm # Finite difference approx
        cf = util.inv_marg_u(dv,par) # Optimal consumption given value function
        c0 = np.tile(par.grid_m,(y_size,1))*par.r \
                        + np.tile(y_vals,(par.Nm,1)).transpose() # Instantaneous savings given cf

        # Savings decomposed into positive and negative components
        ssf[:,:-1] = c0[:,:-1] - cf # Backward imputation
        ssb[:,1:] = c0[:,1:] - cf # Forward imputation

        # Note that the boundary conditions are handled implicitly as ssf will be zero at m_max and ssb at m_min 
        is_forward = ssf > 0
        is_backward = ssb < 0

        # Update savings based on forward or backward difference based on direction of drift
        c0[:,:-1] += (cf - c0[:,:-1])*is_forward[:,:-1]
        c0[:,1:] += (cf - c0[:,1:])*is_backward[:,1:]
            
        # UNCOMMENT FOR DEBUGGING
        #plt.plot(self.par.grid_m, self.c0.transpose())#
        #plt.show()

        c0 = util.u(c0,par) # Check this out
       
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
        b = c0.reshape(n,1) + sol.v.reshape(n,1)/Delta
        v_old = sol.v.copy() # Save for comparison
        sol.v = spsolve(sol.B,b).reshape(y_size,par.Nm)

        # Check distance between previous iteration and update iteration counter
        sol.delta = np.amax(np.absolute(v_old-sol.v).reshape(n))
        sol.it += 1

    # Unpack solution - Add what is needed here
    sol.c = cf
    sol.a = par.grid_a
    sol.dv = dv

    return sol