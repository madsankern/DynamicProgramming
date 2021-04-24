# Solve problem using endogneous grid method
# General kommentar: Først og fremmest, insane arbejde! Jeg synes dog vi bør overveje, at splitte funktionerne op, så
# EGM-funktionerne kun omhandler for-loopet, der itererer igennem end-of-period assets, og while-loopet i stedet bliver kaldt
# i en solver-funktion i filen "model.py", da vi på den måde adskiller hvad der er EGM og hvad der er contraction mapping.  

import numpy as np
import tools
import scipy.optimize as optimize
import utility as util

def solve_EGM(par):
    
    # Initialize solution class
    class sol: pass    
    
    # Initial guess is like a 'last period' choice - consume everything
    # **** Jeg tænker ikke, at vi behøver at kalde linspace-funktionen nedenunder igen, da vi jo allerede har oprettet griddet i
    # model.py klassen. Måske kan man bare slette linjen nedenunder, og erstatte
    # "sol.c = sol.a.copy()" med "sol.c = par.grid_a.copy()"
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
            a_plus = (1+par.r)*s + par.y[0] # post decision state. Note vector
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


# Opdated EGM algorithm with multiple state - Generalised Markov transiton probabilities
def solve_EGM_2d(par):
    
    # Initialize solution class
    class sol: pass    
    
    # Initial guess is like a 'last period' choice - consume everything
    sol.a = np.tile(np.linspace(par.a_min,par.a_max,par.num_a+1), np.shape(par.y.transpose())) # a is pre descision, so for any state consume everything.
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
            a_plus = (1+par.r)*s + np.transpose(par.y[0]) # Transpose for dimension to fit

            # Interpolate next periods consumption - can this be combined?
            c_plus_1 = tools.interp_linear_1d(a_next[0,:], c_next[0,:], a_plus) # State 1
            c_plus_2 = tools.interp_linear_1d(a_next[1,:], c_next[1,:], a_plus) # State 2

            #Combine into a vector. Rows indicate income state, columns indicate asset state
            c_plus = np.vstack((c_plus_1, c_plus_2))

            # Marginal utility
            marg_u_plus = util.marg_u(c_plus,par)
            av_marg_u_plus = np.sum(par.P*marg_u_plus, axis = 1) # Dot product by row (axis = 1)

            # Add optimal consumption and endogenous state
            sol.c[:,i_s+1] = util.inv_marg_u((1+par.r)*par.beta*av_marg_u_plus,par)
            sol.a[:,i_s+1] = s + sol.c[:,i_s+1]
       
        # Update iteration parameters
        sol.it += 1
        sol.delta = max( max(abs(sol.c[0] - c_next[0])), max(abs(sol.c[1] - c_next[1]))) # check this, is this optimal
    
    # add zero consumption
    sol.a[:,0] = 0
    sol.c[:,0] = 0

    return sol



## BACKUP OF MARKOV EGM BEFORE UPDATE ##
# def solve_EGM_2d(par):
    
#     # Initialize solution class
#     class sol: pass    
    
#     # Initial guess is like a 'last period' choice - consume everything
#     sol.a = np.tile(np.linspace(par.a_min,par.a_max,par.num_a+1), (2,1)) # a is pre descision, so for any state consume everything
#     # print((sol.a[0] == sol.a[1]).all()) - debugging
#     sol.c = sol.a.copy() # Consume everyting - this could be improved

#     sol.it = 0 # Iteration counter
#     sol.delta = 1000.0 # Difference between iterations

#     # Iterate value function until convergence or break if no convergence
#     while (sol.delta >= par.tol_egm and sol.it < par.max_iter):

#         # Use last iteration to compute the continuation value
#         # therefore, copy c and a grid from last iteration.
#         c_next = sol.c.copy()
#         a_next = sol.a.copy()

#         # Loop over exogneous states (post decision states)
#         for i_s,s in enumerate(par.grid_s):
            
#             #Next periods assets and consumption
#             # **** Måske skulle vi overveje bare at transponere par.y allerede i model.py filen. Det er f.eks. lidt forvirrende
#             # at vi transponerer i denne EGM-funktion, men ikke i ovenstående (selvom det self ikke er nødvendigt i ovenstående)
#             a_plus = (1+par.r)*s + np.transpose(par.y) # Transpose for dimension to fit

#             c_plus_1 = tools.interp_linear_1d(a_next[0,:], c_next[0,:], a_plus) # State 1
#             c_plus_2 = tools.interp_linear_1d(a_next[1,:], c_next[1,:], a_plus) # State 2
#             # print((c_plus_1 == c_plus_2).all())

#             marg_u_plus_1 = util.marg_u(c_plus_1,par)
#             marg_u_plus_2 = util.marg_u(c_plus_2,par)

#             # Expected utility in next period - computed by matrix multiplication
#             av_marg_u_plus_1 = np.sum(par.P[0]*marg_u_plus_1)
#             av_marg_u_plus_2 = np.sum(par.P[1]*marg_u_plus_2)

#             sol.c[0, i_s+1] = util.inv_marg_u((1+par.r)*par.beta*av_marg_u_plus_1,par)
#             sol.c[1, i_s+1] = util.inv_marg_u((1+par.r)*par.beta*av_marg_u_plus_2,par)

#             sol.a[0, i_s+1] = s + sol.c[0, i_s+1] # Endogenous state
#             sol.a[1, i_s+1] = s + sol.c[1, i_s+1] # Endogenous state

#         # Update iteration parameters
#         sol.it += 1
#         sol.delta = max( max(abs(sol.c[0] - c_next[0])),   max(abs(sol.c[1] - c_next[1]))) # check this
#         # print(sol.delta)
#         # Uncomment for debugging
#         # sol.test = abs(sol.c - c_next)
    
#     # add zero consumption
#     sol.a[0,0] = 0
#     sol.c[0,0] = 0

#     sol.a[1,0] = 0
#     sol.c[1,0] = 0
         
#     return sol
