# Import packages
import numpy as np
import tools


# Solve for optimal choice using upper envelope
def EGM (sol,z_plus, t,par):

    # Draw raw dendent on state of housing
    if z_plus == 1:     # House 
        w_raw, avg_marg_u_plus = house(sol,z_plus,t,par)
    else:               # No house
        w_raw, avg_marg_u_plus = no_house(sol,z_plus,t,par)

    # raw c, m and v
    c_raw = inv_marg_util(par.beta*par.R*avg_marg_u_plus,par)
    m_raw = c_raw + par.grid_a[t,:]
    v_raw = util(c_raw,z_plus, par) + par.beta * w_raw
    
    # This is all choices of c and associated value where the necessary condition of the euler is true.
    # In the upper envelope algorithm below, all non optimal choices are removed.

    ## UPPER ENVELOPE ##

    # Reorderining making G_m strictly increasing 
    m = sorted(m_raw)  # alternatively, choose a common grid exogeneously. This, however, creates many points around the kink
    I = m_raw
    c = [x for _,x in sorted(zip(I,c_raw))]  #Check these
    v = [x for _,x in sorted(zip(I,v_raw))]

    # Loop through the endogenous grid
    for i in range(np.size(m_raw)-2): # Why minus 2?
        m_low = m_raw[i]
        m_high = m_raw[i+1]
        c_slope = (c_raw[i+1]-c_raw[i])/(m_high-m_low)

        # Loop through the common grid
        for j in range(len(m)):

            if  m[j]>=m_low and m[j]<=m_high:

                c_guess = c_raw[i] + c_slope*(m[j]-m_low)
                v_guess = value_of_choice(m[j],c_guess,z_plus,t,sol,par)
                    
                # Update
                if v_guess >v[j]:
                    v[j]=v_guess
                    c[j]=c_guess

    return m,c,v

# Having a house at the beginning of the period
def house(sol,z_plus,t,par):

    # Prepare
    a = np.repeat(par.grid_a[t],par.Nxi) # Compute a grid for each node for integration - remove

    # Next period states
    m_plus = par.R*a + par.W # Here, not considering housing
    shape = (2,m_plus.size) # One row for each choice of housing

    # Intialize
    v_plus = np.nan+np.zeros(shape)
    c_plus = np.nan+np.zeros(shape)
    marg_u_plus = np.nan+np.zeros(shape)

    for i in range(2): # Range keeping house (i = 1) and selling house (i = 0)
                
        # Choice specific house gain
        gain = (1-i)*par.ph

        # Choice specific value
        v_plus[i,:]=tools.interp_linear_1d(sol.m[t+1,i,par.N_bottom:],sol.v[t+1,i,par.N_bottom:], m_plus + gain)
            
        #Choice specific consumption
        c_plus[i,:] = tools.interp_linear_1d(sol.m[t+1,i,par.N_bottom:],sol.c[t+1,i,par.N_bottom:], m_plus + gain)
            
        # Choice specific Marginal utility
        marg_u_plus[i,:] = marg_util(c_plus[i,:],par)

    # Expected value
    V_plus, prob = logsum(v_plus[0],v_plus[1],par.sigma_eta) # logsum computes the optimal of housing in the next period
    w_raw = V_plus
    avg_marg_u_plus = prob[0,:]*marg_u_plus[0] + prob[1,:]*marg_u_plus[1] #Expected margnal utility dependend on choice probabilities

    return w_raw, avg_marg_u_plus

# Not having a house at the beginning of the period
def no_house(sol,z_plus,t,par):
    
    # Prepare
    a = np.repeat(par.grid_a[t],par.Nxi) 

    # Next period states
    m_plus = par.R*a+par.W

    shape = (2,m_plus.size)
    v_plus = np.nan+np.zeros(shape)
    c_plus = np.nan+np.zeros(shape)
    marg_u_plus = np.nan+np.zeros(shape)

    for i in range(2):  # Range over buying house (i = 1) and no house (i = 0)

        # Choice specific cost of the house
        cost = i*par.ph

        # Choice specific value
        v_plus[i,:]=tools.interp_linear_1d(sol.m[t+1,i,par.N_bottom:],sol.v[t+1,i,par.N_bottom:], m_plus - cost)

        #Choice specific consumption
        c_plus[i,:] = tools.interp_linear_1d(sol.m[t+1,i,par.N_bottom:],sol.c[t+1,i,par.N_bottom:], m_plus - cost)

        # Choice specific Marginal utility
        marg_u_plus[i,:] = marg_util(c_plus[i,:], par)

    # Expected value
    V_plus, prob = logsum(v_plus[0],v_plus[1],par.sigma_eta)
    w_raw = V_plus
    avg_marg_u_plus = prob[0,:]*marg_u_plus[0] + prob[1,:]*marg_u_plus[1] #Expected margnal utility dependend on choice probabilities

    return w_raw, avg_marg_u_plus

# Value of choice - used for computing value function for given choice of c
def value_of_choice(m,c,h,t,sol,par):

    # Next period ressources
    a = np.repeat(m-c,(par.xi.size))
    m_plus = par.R * a + par.W

 
    # Note: Siden vi laver upper-envelope på både house og no-house, så skal vi indføre et if-statement.
    # Next-period value if you choose no-house today
    if h==0:
        v_plus0 = tools.interp_linear_1d(sol.m[t+1,0,par.N_bottom:],sol.v[t+1,0,par.N_bottom:], m_plus) # No house
        v_plus1 = tools.interp_linear_1d(sol.m[t+1,1,par.N_bottom:],sol.v[t+1,1,par.N_bottom:], m_plus-par.ph) # House
    # Next-period value if you choose house today.    
    else:
        v_plus0 = tools.interp_linear_1d(sol.m[t+1,0,par.N_bottom:],sol.v[t+1,0,par.N_bottom:], m_plus+par.ph) # No house
        v_plus1 = tools.interp_linear_1d(sol.m[t+1,1,par.N_bottom:],sol.v[t+1,1,par.N_bottom:], m_plus) # House    
    
    V_plus, _ = logsum(v_plus0,v_plus1,par.sigma_eta) # Find the maximum of v0 and v1

    # This period value
    v = util(c,h,par) + par.beta*V_plus
    return v

###############
## FUNCTIONS ##
###############

def util(c,h,par):
    return ((c**(1.0-par.rho))/(1.0-par.rho) + par.kappa*h)

def marg_util(c,par):
    return c**(-par.rho)


def inv_marg_util(u,par):
    return u**(-1/par.rho)


def logsum(v1,v2,sigma):

    # setup
    V = np.array([v1, v2])

    # Maximum over the discrete choices
    mxm = V.max(0)

    # check the value of sigma
    if abs(sigma) > 1e-10:
    
        # numerically robust log-sum
        log_sum = mxm + sigma*(np.log(np.sum(np.exp((V - mxm) / sigma),axis=0)))
        
        # d. numerically robust probability
        prob = np.exp((V- log_sum) / sigma)    

    else: # No smmothing --> max-operator
        id = V.argmax(0)    #Index of maximum
        log_sum = mxm
        prob = np.zeros((v1.size*2))
        I = np.cumsum(np.ones((v1.size,1)))+id*(v1.size)-1
        I = I.astype(int)  # change type to integer
        prob[I] = 1
    
        prob = np.reshape(prob,(2,v1.size),'A')

    return log_sum,prob