import numpy as np
import tools
import utility as util

# Function that returns value of consumption choice conditional on the state
def value_of_choice(x,m,m_next,v_next,par,state):
    
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