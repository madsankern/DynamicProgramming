import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Local modules
import utility as util

# ##############################
# ## Finite Difference method ##
# ##############################

def solve(par,sol):

    ## SETUP ##

    # Asst grid
    N = par.Nm
    m = par.grid_m.transpose() # N x 1 vector
    dm = (par.m_max-par.m_min)/(par.Nm-1) # Stepsize on m grid
    mm = np.tile(np.array([m]), (2,1))
    mm = mm.transpose() # Write this better

    # Income    
    y = par.y
    yy = y*np.ones((N,2)) # 2 dimensional income grid

    Delta_step = 1000 # Stepsize
    delta = 1000 # Distance between iterations - MADS
    it = 0 # Iteration counter

    # Initialize
    v0 = np.zeros((N,2))
    dVf = np.zeros((N,2)) 
    dVb = np.zeros((N,2))
    c = np.zeros((N,2))
    u = np.zeros((N,2))
    V_n = np.zeros((N,2,par.max_iter))
    A = sparse.eye(2*N, format = 'csr') # Transition matrix

    # Transition matrix for income
    y_transition = sparse.kron(par.pi, sparse.eye(par.Nm), format = "csr")
    
    # Initial guess of value function
    v0[:,0] = util.u(yy[:,0] + par.r*mm[:,0], par)/par.rho
    v0[:,1] = util.u(yy[:,1] + par.r*mm[:,1], par)/par.rho
    v = v0

    while (delta >= par.tol_fd and it < par.max_iter):

        # Redefine
        V = v
        V_n[:,:,it] = V

        # Forward difference
        dVf[:N-1,:] = (V[1:N,:] - V[0:N-1,:])/dm
        dVf[N-1,:] = util.marg_u(y + par.r*mm[N-1,:],par) #will never be used, but impose state constraint a<=amax just in case

        # Backward difference
        dVb[1:,:] = ((V[1:N,:] - V[0:N-1,:]))/dm
        dVb[0,:] = util.marg_u(y + par.r*mm[0,:], par) # state constraint boundary condition

        I_concave = dVb > dVf

        #consumption and savings with forward difference
        cf = util.inv_marg_u(dVf,par)# Take inverse marginal utility of value function (c foc)
        ssf = yy + par.r*mm - cf # Savings from forward difference

        #consumption and savings with backward difference
        cb = util.inv_marg_u(dVb, par)
        ssb = yy + par.r*mm - cb

        #consumption and derivative of value function at steady state
        c0 = yy + par.r*mm
        dV0 = util.marg_u(c0,par)

        # Upwind method makes a choice of forward or backward differences based on the sign of the drift    
        If = ssf > 0 #positive drift --> forward difference
        Ib = ssb < 0 #negative drift --> backward difference
        I0 = (1-If-Ib) #at steady state

        dV_upwind = dVf*If + dVb*Ib + dV0*I0
        c = util.inv_marg_u(dV_upwind,par)

        # Utility of choice
        u = util.u(c, par) # Utility without car

        # Terms to be put into the A matrix
        X = -np.minimum(ssb,0)/dm
        Y = - np.maximum(ssf,0)/dm + np.minimum(ssb,0)/dm
        Z = np.maximum(ssf,0)/dm
        Z_helper = np.vstack((np.array([0,0]), Z))

        # Construct A matrix
        A1 = sparse.eye(N, format = 'csr')*0 # Intialize sparse zero mat
        A1 += sparse.spdiags(Y[:,0],0,N,N)
        A1 += sparse.spdiags(X[1:,0],-1,N,N)
        A1 += sparse.spdiags(Z_helper[:N-2,0],1,N,N)

        A2 = sparse.eye(N, format = 'csr')*0 # Intialize sparse zero mat
        A2 += sparse.spdiags(Y[:,1],0,N,N)
        A2 += sparse.spdiags(X[1:,1],-1,N,N)
        A2 += sparse.spdiags(Z_helper[:N-2,1],1,N,N)

        # Stack A1 and A2
        block_helper = sparse.eye(N, format = 'csr')*0
        A_upper = sparse.hstack([A1,block_helper], format = 'csr')
        A_lower = sparse.hstack([block_helper,A2], format = 'csr')
        A = sparse.vstack([A_upper,A_lower], format = 'csr') + y_transition # This is the transition matrix used for the problem

        # Write a check for the row sum of A, should be zero

        # B matrix
        B = (par.rho + 1/Delta_step)*sparse.eye(2*N) - A

        # Stack utility vector into one long column vector
        u_stacked = np.hstack(([u[:,0]], [u[:,1]])).transpose()
        V_stacked = np.hstack(([V[:,0]], [V[:,1]])).transpose()

        b = u_stacked + V_stacked/Delta_step

        # Solve system
        V_stacked = spsolve(B,b)

        V_old = V.copy()
        V[:,0] = V_stacked[:N]
        V[:,1] = V_stacked[N:2*N]
        # np.hstack((V_stacked[:N], V_stacked[N:2*N-1]))

        Vchange = V - V_old
        v = V

        delta = abs(max(max(Vchange[0,:], key=abs), max(Vchange[1,:], key=abs)))
        it += 1

    # Unpack solution
    sol.dV = dV_upwind.transpose()
    sol.v = v.transpose()
    sol.c = c.transpose()
    sol.m = m

    return sol