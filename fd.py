#################################
## Finite Difference algorithm ##
#################################

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
import utility as util

def solve(par,sol):

    # Setup
    da = (par.a_max - par.a_min)/(par.Na - 1) # Finite approx for derivative
    y_vals = par.y # Income process
    y_size = (len(y_vals)) # Number of states in income process

    # Dimension of transition matrix
    n = par.Na * y_size 

    # Initial guess on value function = Stay put
    sol.v = util.u(np.tile(par.grid_a,(y_size,1))*par.r
                    + np.tile(y_vals,(par.Na,1)).transpose(),par)/par.rho

    # Skill transition matrix
    y_transition = sparse.kron(par.pi, sparse.eye(par.Na), format = "csr")

    # Preallocation of memory
    v_old = np.zeros((y_size, par.Na))
    dv = np.zeros((y_size, par.Na-1))
    cf = np.zeros((y_size, par.Na-1))
    c0 = np.zeros((y_size,par.Na))
    ssf = np.zeros((y_size,par.Na))
    ssb = np.zeros((y_size,par.Na))
    is_forward = np.zeros((y_size,par.Na),'bool') # indicators
    is_backward = np.zeros((y_size,par.Na),'bool') # indicators
    diag_helper = np.zeros((y_size,par.Na))        
    sol.A = y_transition.copy()
    sol.B = y_transition.copy()
    # AT = y_transition.copy()

    sol.it = 0 # Iteration counter
    sol.delta = 1000.0 # Difference between iterations
    Delta = 1000 # Check this out - jump size

    # Iterate the discretized HJB equation
    while (sol.delta >= par.tol_fd and sol.it < par.max_iter):

        # Saving and consumption implied by current guess for value function, using upwind method
        dv = (sol.v[:,1:]-sol.v[:,:-1])/da # Derivative of value function
        cf = util.inv_marg_u(dv*(1+par.r),par) # FOC from HJB
        c0 = np.tile(par.grid_a,(y_size,1))*par.r \
                        + np.tile(y_vals,(par.Na,1)).transpose() # Instantaneous income

        # Savings with forward forward difference and backward difference - FIND OUT WHETHER TO MULTIPLY CF WITH R OR NOT
        ssf[:,:-1] = c0[:,:-1] - cf
        ssb[:,1:] = c0[:,1:] - cf

        # Note that the boundary conditions are handled implicitly as ssf will be zero at a_max and ssb at a_min 
        is_forward = ssf > 0
        is_backward = ssb < 0

        # Update consumption based on forward or backward difference based on direction of drift
        c0[:,:-1] += (cf - c0[:,:-1])*is_forward[:,:-1]
        c0[:,1:] += (cf - c0[:,1:])*is_backward[:,1:]
            
        # UNCOMMENT FOR DEBUGGING
        #plt.plot(self.par.grid_a, self.c0.transpose())#
        #plt.show()

        c0 = util.u(c0,par) # Check this out
        
        # Build the matrix A that summarizes the evolution of the process for (a,z)
        # This is a Poisson transition matrix (aka intensity matrix) with rows adding up to zero
        sol.A = y_transition.copy()
        diag_helper = (-ssf*is_forward/da \
                            + ssb*is_backward/da).reshape(n)
        sol.A += sparse.spdiags(diag_helper,0,n,n)
        diag_helper = (-ssb*is_backward/da).reshape(n)
        sol.A += sparse.spdiags(diag_helper[1:],-1,n,n)
        diag_helper = (ssf*is_forward/da).reshape(n)
        sol.A += sparse.spdiags(np.hstack((0,diag_helper)),1,n,n)

        # Solve the system of linear equations corresponding to implicit finite difference scheme
        sol.B = sparse.eye(n)*(1/Delta + par.rho) - sol.A
        b = c0.reshape(n,1) + sol.v.reshape(n,1)/Delta
        v_old = sol.v.copy()
        sol.v = spsolve(sol.B,b).reshape(y_size,par.Na)

        # Check distance between previous iteration and update iteration counter
        sol.delta = np.amax(np.absolute(v_old-sol.v).reshape(n))
        sol.it += 1

    # Unpack solution - Add what is needed here
    sol.c = cf
    sol.a = par.grid_a

    return sol