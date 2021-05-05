import numpy as np
import tools
import scipy.optimize as optimize
import utility as util

# Move utility functions later
def u_c(c,par):

    if par.eta == 1.0:
        u = np.log(c)

    else:
        u = (c**(1-par.eta) - 1.0) / (1.0 - par.eta)

    return u

def u_h(h,par): # Seperate function for housing

    u = np.sqrt(h)

    return u


# Solve the full model of consumption and housing
def solve_VFI(par):

    # Solution method:
    # Solve optimal consumption for constant h
    # Solve optimal h - maybe need to use interpolation?

    # Initialize solution class
    class sol: pass
    sol.c = par.grid_a.copy() # Initial guess is to consume everything
    sol.vc = u(sol.c) # consumption utility function of choice
    sol.a = par.grid_a.copy() # Grid of net worth to loop over
    
    sol.it = 0 # Iteration counter
    sol.delta = 1000.0 # Difference between two iterations
    
    # Iterate value function until convergence or break if no convergence
    while (sol.delta >= par.tol_vfi and sol.it < par.max_iter):
        
        # Use last iteration as the continuation value. See slides if confused
        v_next = sol.vc.copy()
        
        # Loop over asset grid
        for i_a,a in enumerate(par.grid_a):
            
            # Minimize the minus the value function wrt consumption
            obj_fun = lambda x : - value_of_choice(x,a,par.grid_a,v_next,par)
            res = optimize.minimize_scalar(obj_fun, bounds=[0,a+1.0e-4], method='bounded')
            
            # Unpack solution
            sol.v[i_a] = -res.fun
            sol.c[i_a] = res.x
            
        # Update iteration parameters
        sol.it += 1
        sol.delta = max(abs(sol.v - v_next))

    return sol

def value_of_choice(x,a,a_next,v_next,par):
    
    # Unpack consumption (choice variable)
    c = x

    # Intialize expected continuation value
    Ev_next = 0.0
    
    # Loop over each possible state
    for i in [0,1]:
        
        # Next periods state for each income level
        a_plus = par.y[i] + (1+par.r)*(a - c)
        
        #Interpolate continuation given state a_plus
        v_plus = tools.interp_linear_1d_scalar(a_next,v_next,a_plus)
    
        # Append continuation value to calculate expected value
        Ev_next += par.Pi[i] * v_plus
        
    # Value of choice
    v_guess = util.u(c,par) + par.beta * Ev_next

    return v_guess