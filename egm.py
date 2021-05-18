import numpy as np
import tools
import utility as util

#####################################
## Nested EGM ##
#####################################


# Objective function for the keeper
def obj_keep(arg, n, m, v_next, par, m_next): # I have added m_next to the interpolation since it changes throughout iterations

    # Unpack
    c = arg

    # End of period assets
    m_plus = (1+par.r)*(m - c) + par.y1

    # Continuation value
    v_plus = tools.interp_linear_1d_scalar(m_next, v_next, m_plus)

    # Value of choice
    value = util.u_h(c,n,par) + par.beta*v_plus

    return value

# Solution algorithm
def solve_dc(sol, par, v_next, c_next, h_next, m_next):

    # a. Solve the keeper problem

    shape = (2,np.size(par.grid_a)) # Row for each state of housing and colums for exogenous end-of-period assets grid

    # Intialize
    v_keep = np.zeros(shape) + np.nan
    c_keep = np.zeros(shape) + np.nan
    h_keep = np.zeros(shape) + np.nan

    # Loop over housing states
    for n in range(2):
        
        # Loop over exogenous states (post decision states)
        for m_i,a in enumerate(par.grid_a):

            #Next periods assets and consumption
            m_plus = (1+par.r)*a + par.y1

            # Interpolate next periods consumption 
            c_plus = tools.interp_linear_1d_scalar(m_next[n,:], c_next[n,:], m_plus) 
            
            # Marginal utility
            marg_u_plus = util.marg_u(c_plus,par)
            #av_marg_u_plus = np.sum(par.P*marg_u_plus, axis = 1) # Dot product by row (axis = 1) #### no average

            # Add optimal consumption and endogenous state using Euler equation
            
            #sol.c[:,a_i+1] = util.inv_marg_u((1+par.r)*par.beta*av_marg_u_plus,par)
            #sol.m[:,a_i+1] = a + sol.c[:,a_i+1]
            #sol.v = util.u(sol.c,par)
            
            c_keep[n,m_i] = util.inv_marg_u((1+par.r)*par.beta*marg_u_plus,par) #### no average
            v_keep[n,m_i] = util.u_h(c_keep[n,m_i],n,par) 
            h_keep[n,m_i] = n

            
    ## b. Upper envelope ## ... do we need to include 'h' in the upper envelope algorithm? 
    
    # raw c, m and v for each housing state (can probably be written into a loop or vectorized)
    c_raw_0 = c_keep[0]
    c_raw_1 = c_keep[1]
    m_raw   = c_keep + np.transpose(par.grid_a) 
    m_raw_0 = m_raw[0]
    m_raw_1 = m_raw[1]
    v_raw_0 = v_keep[0]
    v_raw_1 = v_keep[1]
    

    # This is all choices of c and associated value where the necessary condition of the euler is true.
    # In the upper envelope algorithm below, all non optimal choices are removed.

    ### first for housing state == 0 ###

    # Reorderining making G_m strictly increasing 
    m_0 = sorted(m_raw_0)  # alternatively, choose a common grid exogeneously. This, however, creates many points around the kink
    I_0 = m_raw_0
    c_0 = [x for _,x in sorted(zip(I_0,c_raw_0))]  # Here Thomas basically merges/zips the raw grids together, so that the c's and v's are associated with the correct m's
    v_0 = [x for _,x in sorted(zip(I_0,v_raw_0))]

    # Loop through the endogenous grid
    for i in range(np.size(m_raw_0)-2): # Why minus 2? 
        m_low_0 = m_raw_0[i]
        m_high_0 = m_raw_0[i+1]
        c_slope_0 = (c_raw_0[i+1]-c_raw_0[i])/(m_high_0-m_low_0)

        # Loop through the common grid
        for j in range(len(m_0)):

            if  m_0[j]>=m_low_0 and m_0[j]<=m_high_0:

                c_guess_0 = c_raw_0[i] + c_slope_0*(m_0[j]-m_low_0)
                # v_guess_0 = value_of_choice(m[j],c_guess,z_plus,t,sol,par) # value_of_choice should be changed to object_keep
                v_guess_0 = obj_keep(c_guess_0, 0, m_0[j], v_next[0], par, m_next[0]) # check v_next

                # Update
                if v_guess_0 >v_0[j]:
                    v_0[j]=v_guess_0
                    c_0[j]=c_guess_0

    ### then for housing state == 1 ###

    # Reorderining making G_m strictly increasing 
    m_1 = sorted(m_raw_1)  # alternatively, choose a common grid exogeneously. This, however, creates many points around the kink
    I_1 = m_raw_1
    c_1 = [x for _,x in sorted(zip(I_1,c_raw_1))]  # Here Thomas basically merges/zips the raw grids together, so that the c's and v's are associated with the correct m's
    v_1 = [x for _,x in sorted(zip(I_1,v_raw_1))]

    # Loop through the endogenous grid
    for i in range(np.size(m_raw_1)-2): # Why minus 2? 
        m_low_1 = m_raw_1[i]
        m_high_1 = m_raw_1[i+1]
        c_slope_1 = (c_raw_1[i+1]-c_raw_1[i])/(m_high_1-m_low_1)

        # Loop through the common grid
        for j in range(len(m_1)):

            if  m_1[j]>=m_low_1 and m_1[j]<=m_high_1:

                c_guess_1 = c_raw_1[i] + c_slope_1*(m_1[j]-m_low_1)
                # v_guess_0 = value_of_choice(m[j],c_guess,z_plus,t,sol,par) # value_of_choice should be changed to object_keep
                v_guess_1 = obj_keep(c_guess_1, 1, m_1[j], v_next[1], par, m_next[1]) # check v_next

                # Update
                if v_guess_1 >v_1[j]:
                    v_1[j]=v_guess_1
                    c_1[j]=c_guess_1                    

#    return m,c,v

    #c = np.zeros(shape) + np.nan (old)
    c_keep[0] = c_0
    c_keep[1] = c_1
    #v = np.zeros(shape) + np.nan (old)
    v_keep[0] = v_0
    v_keep[1] = v_1
    m_grid = np.zeros(shape) + np.nan
    m_grid[0] = m_0
    m_grid[1] = m_1
    
    # c. Solve the adjuster problem

    # Initialize
    v_adj = np.zeros(shape) + np.nan
    c_adj = np.zeros(shape) + np.nan
    h_adj = np.zeros(shape) + np.nan

    # Loop over housing state
    for n in range(2):

        # Housing choice is reverse of state n if adjusting
        h = 1 - n

        # Loop over asset grid
        for m_i,m in enumerate(m_grid[n]): # endogenous grid or par.grid_m?

            # If adjustment is not possible
            if n == 0 and m < par.ph :
                v_adj[n,m_i] = -np.inf
                c_adj[n,m_i] = 0
                h_adj[n,m_i] = np.nan

            else:

                # Assets available after adjusting
                x = m - par.ph*(h - n)

                # Value of choice
                v_adj[n,m_i] = tools.interp_linear_1d_scalar(m_grid[n], v_keep[h,:], x) # endogenous grid or par.grid_m?
                c_adj[n,m_i] = tools.interp_linear_1d_scalar(m_grid[n], c_keep[h,:], x) # endogenous grid or par.grid_m?
                h_adj[n,m_i] = h

    # d. Combine solutions

    # Loop over asset grid again
    for n in range(2):
        for m_i,m in enumerate(m_grid[n]): # endogenous grid or par.grid_m?

            # If keeping is optimal
            if v_keep[n,m_i] > v_adj[n,m_i]:
                sol.v[n,m_i] = v_keep[n,m_i]
                sol.c[n,m_i] = c_keep[n,m_i]
                sol.h[n,m_i] = n
                sol.m[n,m_i] = m_grid[n,m_i] # added

            # If adjusting is optimal
            else:
                sol.v[n,m_i] = v_adj[n,m_i]
                sol.c[n,m_i] = c_adj[n,m_i]
                sol.h[n,m_i] = 1 - n
                sol.m[n,m_i] = m_grid[n,m_i] # added
                


    return sol
    
    


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