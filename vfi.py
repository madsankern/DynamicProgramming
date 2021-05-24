import numpy as np
import tools
import utility as util
import scipy.optimize as optimize
import matplotlib.pyplot as plt

#####################################
## Nested value function iteration ##
#####################################
# This is the nested VFI algorithm for solving the discrete-continuous model.
# The solution method is to solve the keeper problem where consumption is the only choice,
# then solve the adjuster problem and find the implied value by interpolating
# the solution of the keeper problem

# Objective function for the keeper
def obj_keep(arg, n, m, v_next, par):

    # Unpack
    c = arg

    # End of period assets
    m_plus = (1+par.r)*(m - c) + par.y1

    # Continuation value
    v_plus = tools.interp_linear_1d_scalar(par.grid_m, v_next, m_plus)

    # Value of choice
    value = util.u_h(c,n,par) + par.beta*v_plus

    return value

# Solution algorithm
def solve_dc(sol, par, v_next):

    # a. Solve the keeper problem

    shape = (2,np.size(par.grid_m)) # Row for each state of housing - move to model.py file

    # Intialize
    v_keep = np.zeros(shape) + np.nan
    c_keep = np.zeros(shape) + np.nan
    h_keep = np.zeros(shape) + np.nan

    # Loop over housing states
    for n in range(2):

        # Loop over asset grid
        for m_i,m in enumerate(par.grid_m):

            # High and low bounds
            c_low =  1e-4 #np.fmin(m/2,1e-6)
            c_high = m

            # Call optimizer
            obj_fun = lambda arg : - obj_keep(arg, n, m, v_next[n,:], par)
            res = optimize.minimize_scalar(obj_fun, bounds = [c_low,c_high], method = 'bounded')
            
            # Unpack solution
            v_keep[n,m_i] = -res.fun
            c_keep[n,m_i] = res.x
            h_keep[n,m_i] = n

    ## For debugging ##
    # plt.plot(par.grid_m,c_keep[1])
    # plt.plot(par.grid_m,c_keep[0])
    # plt.show()

    # b. Solve the adjuster problem

    # Intialize
    v_adj = np.zeros(shape) + np.nan
    c_adj = np.zeros(shape) + np.nan
    h_adj = np.zeros(shape) + np.nan

    # Loop over housing state
    for n in range(2):

        # Housing choice is reverse of state n if adjusting
        h = 1 - n

        # Loop over asset grid
        for m_i,m in enumerate(par.grid_m):

            # If adjustment is not possible
            if n == 0 and m < par.ph :
                v_adj[n,m_i] = -np.inf
                c_adj[n,m_i] = 0
                h_adj[n,m_i] = np.nan

            else:

                # Assets available after adjusting
                if n==1:
                    p = par.p1
                else:
                    p = par.ph

                x = m - p*(h - n)

                # Value of choice
                v_adj[n,m_i] = tools.interp_linear_1d_scalar(par.grid_m, v_keep[h,:], x)
                c_adj[n,m_i] = tools.interp_linear_1d_scalar(par.grid_m, c_keep[h,:], x)
                h_adj[n,m_i] = h

    # c. Combine solutions

    # Loop over asset grid again
    for n in range(2):
        for m_i,m in enumerate(par.grid_m):

            # If keeping is optimal
            if v_keep[n,m_i] > v_adj[n,m_i]:
                sol.v[n,m_i] = v_keep[n,m_i]
                sol.c[n,m_i] = c_keep[n,m_i]
                sol.h[n,m_i] = n

            # If ajusting is optimal
            else:
                sol.v[n,m_i] = v_adj[n,m_i]
                sol.c[n,m_i] = c_adj[n,m_i]
                sol.h[n,m_i] = 1 - n
    
    return sol



##############################
## Value function iteration ##
##############################
# This is the VFI algortihm for solving the simple consumption saving model.

def solve(sol, par, v_next):

    # Loop over asset grid
    for m_i,m in enumerate(par.grid_m):

        # Loop over income states
        for y in range(2):

            # Minimize the minus the value function wrt consumption conditional on income state
            obj_fun = lambda x : - value_of_choice(x,m,par.grid_m,v_next[y,:],par,y)
            res = optimize.minimize_scalar(obj_fun, bounds=[0+1.0e-4, m+1.0e-4], method='bounded')
        
            # Unpack solutions
            sol.v[y,m_i] = -res.fun
            sol.c[y,m_i] = res.x

    return sol

# Function that returns value of consumption choice conditional on the income state
def value_of_choice(x,m,m_next,v_next,par,state):
    
    # Unpack consumption (choice variable)
    c = x

    m_plus = par.y + (1 + par.r)*(m - c)

    v_plus = tools.interp_linear_1d(m_next, v_next, m_plus) # Returns one point for each income state

    Ev_next = np.sum(par.P[state]*v_plus)

    # Value of choice given choice c = x
    value = util.u(c,par) + par.beta * Ev_next

    return value