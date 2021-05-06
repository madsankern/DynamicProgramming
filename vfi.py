import numpy as np
import tools
import utility as util
import scipy.optimize as optimize

# Collection of functions used for value function iteration

def solve(sol, par, v_next, state1, state2):
    # Solve function used for value function iteration

    # Loop over asset grid
    for m_i,m in enumerate(par.grid_m):

        # FUNCTIONS BELOW CAN BE WRITTEN AS LOOP - for i=0,1 - AND BE STORED IN AN ARRAY/LIST WITH TWO ENTRIES - a la res[i]=optimize.minimize....

        # Minimize the minus the value function wrt consumption conditional on unemployment state
        obj_fun = lambda x : - value_of_choice(x,m,par.grid_m,v_next[0,:],par,state1)
        res_1 = optimize.minimize_scalar(obj_fun, bounds=[0+1.0e-4, m+1.0e-4], method='bounded')

        # Minimize the minus the value function wrt consumption conditional on employment state
        obj_fun = lambda x : - value_of_choice(x,m,par.grid_m,v_next[1,:],par,state2)
        res_2 = optimize.minimize_scalar(obj_fun, bounds=[0+1.0e-4, m+1.0e-4], method='bounded')
        
        # Unpack solutions
        # State 1
        sol.v[0,m_i] = -res_1.fun
        sol.c[0,m_i] = res_1.x

        # State 2
        sol.v[1,m_i] = -res_2.fun
        sol.c[1,m_i] = res_2.x

    return sol

# Function that returns value of consumption choice conditional on the state
def value_of_choice(x,m,m_next,v_next,par,state):
    
    # Unpack consumption (choice variable)
    c = x

    m_plus = par.y + (1 + par.r)*(m - c)

    v_plus = tools.interp_linear_1d(m_next, v_next, m_plus) # Returns one point for each state

    Ev_next = np.sum(par.P[state]*v_plus)

    # Value of choice given choice c = x
    value = util.u(c,par) + par.beta * Ev_next

    return value

# Function that returns value of consumption choice conditional on the state
def value_of_choice_2(x,m,m_next,v_next,par,state):
    
    # Unpack consumption (choice variable)
    c = x

    # Intialize expected continuation value
    Ev_next = 0.0
    
    # Compute value of choice conditional on being in state 1 (unemployment state)
    ###### VECTORIZE THIS
    if state==1:
        # Loop over each possible state
        for i in [0,1]:
        
            # Next periods state for each income level
            m_plus = par.y[i] + (1+par.r)*(m - c)
        
            #Interpolate continuation given state a_plus
            v_plus = tools.interp_linear_1d_scalar(m_next,v_next,m_plus)
    
            # Append continuation value to calculate expected value
            Ev_next += par.P[0,i] * v_plus
   
    # Compute value of choice conditional on being in state 2 (employment state)
    else:
         # Loop over each possible state
         ###### VECTORIZE THIS
        for i in [0,1]:
        
            # Next periods state for each income level
            m_plus = par.y[i] + (1+par.r)*(m - c)
        
            #Interpolate continuation given state a_plus
            v_plus = tools.interp_linear_1d_scalar(m_next,v_next,m_plus)
    
            # Append continuation value to calculate expected value
            Ev_next += par.P[1,i] * v_plus  
    
    # Value of choice
    v_guess = util.u(c,par) + par.beta * Ev_next

    return v_guess