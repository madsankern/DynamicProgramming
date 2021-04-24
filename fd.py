# Implement the finite difference algorithm

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

import utility as util # Import our utility package

def solve_fd(par):

    # Initialize solution class
    class sol: pass

    # Setup some remaining parameters (move to model.py later)
    da = (par.a_max - par.a_min)/(par.num_a - 1) # finite approx for derivative
    y_list = [1, 1.5]
    y_vals = np.asarray(y_list) # Revisited income process
    y_size = (len(y_vals)) 
 
    # Rename for ease
    a_size = par.num_a
    a_vals = par.grid_a

    # Iteratrion parameters
    max_iter = 500
    tol_fd = 1e-6

    pi_list = [[-0.5, 0.5], [0.1, -0.1]]
    pi = np.asarray(pi_list) # Poisson jumps

    n = a_size * y_size # Dimension of transition matrix

    # Initial guess on value function
    sol.v = np.log(np.tile(a_vals,(y_size,1))*par.r
                    + np.tile(y_vals,(a_size,1)).transpose())/par.rho

    # Skill transition matrix - check up on this
    y_transition = sparse.kron(pi, sparse.eye(a_size), format = "csr")

    # Preallocation
    v_old = np.zeros((y_size, a_size))
    dv = np.zeros((y_size, a_size-1))
    cf = np.zeros((y_size, a_size-1))
    c0 = np.zeros((y_size,a_size))
    ssf = np.zeros((y_size,a_size))
    ssb = np.zeros((y_size,a_size))
    is_forward = np.zeros((y_size,a_size),'bool') # indicators
    is_backward = np.zeros((y_size,a_size),'bool') # indicators
    diag_helper = np.zeros((y_size,a_size))        
    A = y_transition.copy()
    B = y_transition.copy()
    AT = y_transition.copy()

    sol.it = 0 # Iteration counter
    sol.delta = 1000.0 # Difference between iterations
    Delta = 1000

    # Iterate value function
    while (sol.delta >= tol_fd and sol.it < par.max_iter):

        # compute saving and consumption implied by current guess for value function, using upwind method
        dv = (sol.v[:,1:]-sol.v[:,:-1])/da
        cf = util.inv_marg_u(dv,par) # FOC from HJB

        c0 = np.tile(a_vals,(y_size,1))*par.r \
                        + np.tile(y_vals,(a_size,1)).transpose()

        # computes savings with forward forward difference and backward difference
        ssf[:,:-1] = c0[:,:-1] - cf
        ssb[:,1:] = c0[:,1:] - cf
        # Note that the boundary conditions are handled implicitly as ssf will be zero at a_max and ssb at a_min 
        is_forward = ssf > 0
        is_backward = ssb < 0
        # Update consumption based on forward or backward difference based on direction of drift
        c0[:,:-1] += (cf - c0[:,:-1])*is_forward[:,:-1]
        c0[:,1:] += (cf - c0[:,1:])*is_backward[:,1:]
        
        ######
        # UNCOMMENT FOR DEBUGGING
        #plt.plot(self.a_vals, self.c0.transpose())
        #plt.show()

        c0 = util.u(c0,par)
        
        # Build the matrix A that summarizes the evolution of the process for (a,z)
        # This is a Poisson transition matrix (aka intensity matrix) with rows adding up to zero
        A = y_transition.copy()
        diag_helper = (-ssf*is_forward/da \
                            + ssb*is_backward/da).reshape(n)
        A += sparse.spdiags(diag_helper,0,n,n)
        diag_helper = (-ssb*is_backward/da).reshape(n)
        A += sparse.spdiags(diag_helper[1:],-1,n,n)
        diag_helper = (ssf*is_forward/da).reshape(n)
        A += sparse.spdiags(np.hstack((0,diag_helper)),1,n,n)
        # Solve the system of linear equations corresponding to implicit finite difference scheme
        B = sparse.eye(n)*(1/Delta + par.rho) - A
        b = c0.reshape(n,1) + sol.v.reshape(n,1)/Delta
        v_old = sol.v.copy()
        sol.v = spsolve(B,b).reshape(y_size,a_size)

        # Compute convergence metric and stop if it satisfies the convergence criterion
        sol.delta = np.amax(np.absolute(v_old-sol.v).reshape(n))
        sol.it += 1

    sol.A = A
    sol.c = cf
    sol.a = a_vals

    return sol