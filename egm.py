# Solve problem using endogneous grid method

import numpy as np
import tools
import scipy.optimize as optimize
import utility as util

def solve_EGM(par):
    
    # Initialize solution class
    class sol: pass    
    
    # Initial guess is like a 'last period' choice - consume everything
    sol.a = np.linspace(par.a_min,par.a_max,par.num_a+1) # a is pre descision, so for any state consume everything
    sol.c = sol.a.copy() # Consume everyting - this could be improved
    
    sol.it = 0 # Iteration counter
    sol.delta = 1000.0 # Difference between iterations

    # Iterate value function until convergence or break if no convergence
    while (sol.delta >= par.tol_egm and sol.it < par.max_iter):

        # Use last iteration to compute the continuation value
        # therefore, copy c and a grid from last iteration.
        c_next = sol.c.copy()
        a_next = sol.a.copy()

        # Loop over exogneous states (post decision states)
        for i_s,s in enumerate(par.grid_s):
            
            #Next periods assets and consumption
            a_plus = (1+par.r)*s + par.y # post decision state. Note vector
            c_plus = tools.interp_linear_1d(a_next, c_next, a_plus)
            
            # Marginal utility of next periods consumption
            marg_u_plus = util.marg_u(c_plus, par)
            av_marg_u_plus = np.sum(par.Pi*marg_u_plus) # Compute expected utility in next period

            # Optimal c in current period from inverted euler
            # +1 in indexation as we add zero consumption afterwards
            sol.c[i_s+1] = util.inv_marg_u((1+par.r)*par.beta*av_marg_u_plus,par)
            sol.a[i_s+1] = s + sol.c[i_s+1] # Endogenous state
            
        # add zero consumption
        sol.a[0] = 0
        sol.c[0] = 0
        
        # Update iteration parameters
        sol.it += 1
        sol.delta = max(abs(sol.c - c_next))
        
        # Uncomment for debugging
        # sol.test = abs(sol.c - c_next)
         
    return sol