import numpy as np
import tools
import utility as util

#####################################
##            Nested EGM           ##
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
        for a_i,a in enumerate(par.grid_a):

            #Next periods assets and consumption
            m_plus = (1+par.r)*a + par.y1

            # Interpolate next periods consumption 
            c_plus = tools.interp_linear_1d_scalar(m_next[n,:], c_next[n,:], m_plus) 
            
            # Marginal utility
            marg_u_plus = util.marg_u(c_plus,par)
            #av_marg_u_plus = np.sum(par.P*marg_u_plus, axis = 1) # Dot product by row (axis = 1) #### no average

            # Add optimal consumption and endogenous state using Euler equation
            c_keep[n,a_i] = util.inv_marg_u((1+par.r)*par.beta*marg_u_plus,par) #### no average
            # v_keep[n,a_i] = obj_keep(c_keep[n,a_i], n, c_keep[n,a_i] + a, v_next[n,:], par, m_next[n, :])
            # The line below is faster and more precise as it avoids numerical errors
            v_keep[n,a_i] = util.u_h(c_keep[n,a_i],n,par) + par.beta*tools.interp_linear_1d_scalar(m_next[n,:], v_next[n,:], m_plus)
            h_keep[n,a_i] = n

    ### UPPER ENVELOPE ###

    c_keep, v_keep, m_grid = upper_envelope(c_keep, v_keep, v_next, m_next, shape, par)
            

    ### Add points at the constraints ###
    
    m_con = np.array([np.linspace(0+1e-8,m_grid[0,0]-1e-4,par.N_bottom), np.linspace(0+1e-8,m_grid[1,0]-1e-4,par.N_bottom)])
    c_con = m_con.copy()
    v_con_0 = [obj_keep(c_con[0,i],0,m_con[0,i],v_next[0, :], par, m_next[0, :]) for i in range(par.N_bottom)] # From N_bottom or whole
    v_con_1 = [obj_keep(c_con[1,i],1,m_con[1,i],v_next[1, :], par, m_next[1, :]) for i in range(par.N_bottom)] # From N_bottom or whole
    v_con = np.array([v_con_0, v_con_1])

    # initialize new larger keeper containers

    new_shape = (2,np.size(par.grid_a) + par.N_bottom)
    c_keep_append = np.zeros(new_shape) + np.nan
    v_keep_append = np.zeros(new_shape) + np.nan
    m_grid_append = np.zeros(new_shape) + np.nan

    # append

    for i in range(2):
        c_keep_append[i, :] = np.append(c_con[i,:], c_keep[i, :])
        v_keep_append[i, :] = np.append(v_con[i,:], v_keep[i, :])
        m_grid_append[i, :] = np.append(m_con[i,:], m_grid[i, :])

    # b. Solve the adjuster problem

    # Initialize
    v_adj = np.zeros(new_shape) + np.nan
    c_adj = np.zeros(new_shape) + np.nan
    h_adj = np.zeros(new_shape) + np.nan

    # Loop over housing state
    for n in range(2):

        # Housing choice is reverse of state n if adjusting
        h = 1 - n

        # Loop over asset grid
        for a_i,m in enumerate(m_grid_append[n]): # endogenous grid

            # If adjustment is not possible
            if n == 0 and m < par.ph :
                v_adj[n,a_i] = -np.inf
                c_adj[n,a_i] = 0
                h_adj[n,a_i] = np.nan

            else:

                # Assets available after adjusting
                if n==1:
                    p = par.p1
                else:
                    p = par.ph

                x = m - p*(h - n)

                # Value of choice
                v_adj[n,a_i] = tools.interp_linear_1d_scalar(m_grid_append[h], v_keep_append[h,:], x) 
                c_adj[n,a_i] = tools.interp_linear_1d_scalar(m_grid_append[h], c_keep_append[h,:], x) 
                h_adj[n,a_i] = h

    # c. Combine solutions

    # Loop over asset grid again
    for n in range(2):
        for a_i,m in enumerate(m_grid_append[n]): # endogenous grid
            
            # If keeping is optimal
            if v_keep_append[n,a_i] > v_adj[n,a_i]:
                sol.v[n,a_i] = v_keep_append[n,a_i]
                sol.c[n,a_i] = c_keep_append[n,a_i]
                sol.h[n,a_i] = n
                sol.m[n,a_i] = m_grid_append[n,a_i] # added

            # If adjusting is optimal
            else:
                sol.v[n,a_i] = v_adj[n,a_i]
                sol.c[n,a_i] = c_adj[n,a_i]
                sol.h[n,a_i] = 1 - n
                sol.m[n,a_i] = m_grid_append[n,a_i] # added
                
    for i in range(2):
        sol.delta_save[i, sol.it] = max(abs(sol.v[i] - v_next[i]))

    return sol
    
#############################
## Upper envelope for NEGM ##
#############################

def upper_envelope(c_keep, v_keep, v_next, m_next, shape, par):

    m_grid = np.zeros(shape) + np.nan

    for i in range(2):

        c_raw = c_keep[i,:]
        m_raw = c_raw + par.grid_a 
        v_raw = v_keep[i,:]

        # This is all choices of c and associated value where the necessary condition of the euler holds.
        # In the upper envelope algorithm below, all suboptimal choices are removed.

        # Reorderining making G_m strictly increasing 
        m = sorted(m_raw)  # Sorted grid in ascending order
        I = m_raw
        c = [x for _,x in sorted(zip(I,c_raw))]  # Merges/zips the raw grids together, so that the c's and v's are associated with the correct m's
        v = [x for _,x in sorted(zip(I,v_raw))]

        # Loop through the endogenous grid
        for q in range(np.size(m_raw)-2): # Why minus 2? 
            m_low = m_raw[q]
            m_high = m_raw[q+1]
            c_slope = (c_raw[q+1]-c_raw[q])/(m_high-m_low)

            # Loop through the common grid
            for j in range(len(m)):

                if  m[j]>=m_low and m[j]<=m_high:

                    c_guess = c_raw[q] + c_slope*(m[j]-m_low)
                    # v_guess_0 = value_of_choice(m[j],c_guess,z_plus,t,sol,par) # value_of_choice should be changed to object_keep
                    v_guess = obj_keep(c_guess, 0, m[j], v_next[0,par.N_bottom:], par, m_next[0,par.N_bottom:]) # check v_next

                    # Update
                    if v_guess >v[j]:
                        v[j]=v_guess
                        c[j]=c_guess
        v_keep[i,:] = v
        c_keep[i,:] = c
        m_grid[i,:] = m
    return c_keep, v_keep, m_grid    



######################
## Solver using EGM ##
######################

def solve(sol, par, c_next, m_next):

    # Copy last iteration of the value function
    v_old = sol.v.copy()

    # Expand exogenous asset grid
    a = np.tile(par.grid_a, np.size(par.y)) # 2d end-of-period asset grid

    # m_plus = (1+par.r)

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
        # Compute expectation below
        av_marg_u_plus = np.array([par.P[0,0]*marg_u_plus[0,0] + par.P[0,1]*marg_u_plus[1,1], par.P[1,1]*marg_u_plus[1,1] + par.P[1,0]*marg_u_plus[0,0]])
        # Add optimal consumption and endogenous state
        sol.c[:,a_i+1] = util.inv_marg_u((1+par.r)*par.beta*av_marg_u_plus,par)
        sol.m[:,a_i+1] = a + sol.c[:,a_i+1]
        sol.v = util.u(sol.c,par)

    #Compute value function and update iteration parameters
    sol.delta = max( max(abs(sol.v[0] - v_old[0])), max(abs(sol.v[1] - v_old[1])))
    sol.it += 1

    # sol.delta = max( max(abs(sol.c[0] - c_next[0])), max(abs(sol.c[1] - c_next[1])))

    return sol

