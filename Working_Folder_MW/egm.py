import numpy as np
import tools
import utility as util

######################
## Solver using EGM ##
######################

def solve(sol, par, c_next, m_next):

    # Copy last iteration of the value function
    v_old = sol.v.copy()

    # Expand exogenous asset grid
    a = np.tile(par.grid_a, np.size(par.y)) # does this work?

    m_plus = (1+par.r)

    # Loop over exogneous states (post decision states)
    for a_i,a in enumerate(par.grid_a):
        
        #Next periods assets and consumption
        m_plus = (1+par.r)*a + np.transpose(par.y) # Transpose for dimension to fit

        # Interpolate next periods consumption - can this be combined?
        c_plus_1 = tools.interp_linear_1d(m_next[0,:], c_next[0,:], m_plus) # State 1
        c_plus_2 = tools.interp_linear_1d(m_next[1,:], c_next[1,:], m_plus) # State 2

        #Combine into a vector. Rows indicate income state, columns indicate asset state
        c_plus = np.vstack((c_plus_1, c_plus_2))

        # Marginal utility
        marg_u_plus = util.marg_u(c_plus,par)
        av_marg_u_plus = np.sum(par.P*marg_u_plus, axis = 1) # Dot product by row (axis = 1)

        # Add optimal consumption and endogenous state
        sol.c[:,a_i+1] = util.inv_marg_u((1+par.r)*par.beta*av_marg_u_plus,par)
        sol.m[:,a_i+1] = a + sol.c[:,a_i+1]
        sol.v = util.u(sol.c,par)

    #Compute valu function and update iteration parameters
    sol.delta = max( max(abs(sol.v[0] - v_old[0])), max(abs(sol.v[1] - v_old[1])))
    sol.it += 1

    # sol.delta = max( max(abs(sol.c[0] - c_next[0])), max(abs(sol.c[1] - c_next[1])))

    return sol


## Attempt at vectorizing the code
# def solve_vec(sol, par, c_next, m_next):

#     # Copy last iteration of the value function
#     v_old = sol.v.copy()

#     # Expand exogenous asset grid
#     shape = (np.size(par.y),1)


#     a = np.tile(par.grid_a, shape) # does this work?

#     y_help = np.array([par.y])
#     y_help = np.transpose(y_help)
#     y = np.tile(y_help, (1,par.Na))

#     m_plus = (1+par.r)*a + y
#     # m_plus = (1+par.r)*a + np.transpose(par.y)

#     # Interpolate next periods consumption
#     c_plus_1 = tools.interp_linear_1d(m_next[0,:], c_next[0,:], m_plus[0,:]) # State 1
#     c_plus_2 = tools.interp_linear_1d(m_next[1,:], c_next[1,:], m_plus[1,:]) # State 2

#     #Combine into a vector. Rows indicate income state, columns indicate asset state
#     c_plus = np.vstack((c_plus_1, c_plus_2))

#     # Marginal utility
#     marg_u_plus = util.marg_u(c_plus,par)
#     av_marg_u_plus = np.sum(par.P*marg_u_plus)

#     av_marg_u_plus = np.sum(par.P*marg_u_plus, axis = 1) # Dot product by row (axis = 1)

#     # Add optimal consumption and endogenous state
#     sol.c = util.inv_marg_u((1+par.r)*par.beta*av_marg_u_plus,par)
#     sol.m = a + sol.c[:,a_i+1]
#     sol.v = util.u(sol.c,par)



    # Loop over exogneous states (post decision states)
    for a_i,a in enumerate(par.grid_a):
        
        #Next periods assets and consumption
        m_plus = (1+par.r)*a + np.transpose(par.y) # Transpose for dimension to fit

        # Interpolate next periods consumption - can this be combined?
        c_plus_1 = tools.interp_linear_1d(m_next[0,:], c_next[0,:], m_plus) # State 1
        c_plus_2 = tools.interp_linear_1d(m_next[1,:], c_next[1,:], m_plus) # State 2

        #Combine into a vector. Rows indicate income state, columns indicate asset state
        c_plus = np.vstack((c_plus_1, c_plus_2))

        # Marginal utility
        marg_u_plus = util.marg_u(c_plus,par)
        av_marg_u_plus = np.sum(par.P*marg_u_plus, axis = 1) # Dot product by row (axis = 1)

        # Add optimal consumption and endogenous state
        sol.c[:,a_i+1] = util.inv_marg_u((1+par.r)*par.beta*av_marg_u_plus,par)
        sol.m[:,a_i+1] = a + sol.c[:,a_i+1]
        sol.v = util.u(sol.c,par)

    #Compute valu function and update iteration parameters
    sol.delta = max( max(abs(sol.v[0] - v_old[0])), max(abs(sol.v[1] - v_old[1])))
    sol.it += 1

    # sol.delta = max( max(abs(sol.c[0] - c_next[0])), max(abs(sol.c[1] - c_next[1])))

    return sol












### Unused code below ###

def solve_EGM(par):
    
    # Initialize solution class
    class sol: pass    
    
    # Initial guess is like a 'last period' choice - consume everything
    # **** Jeg tænker ikke, at vi behøver at kalde linspace-funktionen nedenunder igen, da vi jo allerede har oprettet griddet i
    # model.py klassen. Måske kan man bare slette linjen nedenunder, og erstatte
    # "sol.c = sol.a.copy()" med "sol.c = par.grid_a.copy()"
    sol.a = np.linspace(par.a_min,par.a_max,par.num_a) # a is pre descision, so for any state consume everything
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
        for a_i,s in enumerate(par.grid_s):
            
            #Next periods assets and consumption
            m_plus = (1+par.r)*s + par.y # post decision state. Note vector
            c_plus = tools.interp_linear_1d(a_next, c_next, m_plus)
            
            # Marginal utility of next periods consumption
            marg_u_plus = util.marg_u(c_plus, par)
            av_marg_u_plus = np.sum(par.Pi*marg_u_plus) # Compute expected utility in next period

            # Optimal c in current period from inverted euler
            # +1 in indexation as we add zero consumption afterwards
            sol.c[a_i+1] = util.inv_marg_u((1+par.r)*par.beta*av_marg_u_plus,par)
            sol.a[a_i+1] = s + sol.c[a_i+1] # Endogenous state
            
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

    # Shape parameter for the solution vector
    shape = (np.size(par.y),1)
    
    # Initial guess is like a 'last period' choice - consume everything
    sol.a = np.tile(np.linspace(par.a_min,par.a_max,par.num_a+1), shape) # a is pre descision, so for any state consume everything.
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
        for a_i,s in enumerate(par.grid_s):
            
            #Next periods assets and consumption
            m_plus = (1+par.r)*s + np.transpose(par.y) # Transpose for dimension to fit

            # Interpolate next periods consumption - can this be combined?
            c_plus_1 = tools.interp_linear_1d(a_next[0,:], c_next[0,:], m_plus) # State 1
            c_plus_2 = tools.interp_linear_1d(a_next[1,:], c_next[1,:], m_plus) # State 2

            #Combine into a vector. Rows indicate income state, columns indicate asset state
            c_plus = np.vstack((c_plus_1, c_plus_2))

            # Marginal utility
            marg_u_plus = util.marg_u(c_plus,par)
            av_marg_u_plus = np.sum(par.P*marg_u_plus, axis = 1) # Dot product by row (axis = 1)

            # Add optimal consumption and endogenous state
            sol.c[:,a_i+1] = util.inv_marg_u((1+par.r)*par.beta*av_marg_u_plus,par)
            sol.a[:,a_i+1] = s + sol.c[:,a_i+1]
       
        # Update iteration parameters
        sol.it += 1
        sol.delta = max( max(abs(sol.c[0] - c_next[0])), max(abs(sol.c[1] - c_next[1]))) # check this, is this optimal
    
    # add zero consumption
    sol.a[:,0] = 0
    sol.c[:,0] = 0

    return sol