# import packages 
import numpy as np
import tools

def setup():
    class par: pass

    par.T = 10

    # Model parameters
    par.rho = 2
    par.beta = 0.96
    par.alpha = 0.75
    par.kappa = 0.5
    par. R = 1.04
    par.W = 1
    par.sigma_xi = 0.1
    par.sigma_eta = 0.1

    # Grids and numerical integration
    par.m_max = 10
    par.m_phi = 1.1 # Curvature parameters
    par.a_max = 10
    par.a_phi = 1.1  # Curvature parameters
    par.p_max = 2.0
    par.p_phi = 1.0 # Curvature parameters

    par.Nxi = 8
    par.Nm = 150
    par.Na = 150
    par.Np = 100

    par.Nm_b = 50
    
    return par

def create_grids(par):

    # Check parameters
    assert (par.rho >= 0), 'not rho > 0'

    # Shocks
    par.xi,par.xi_w = tools.GaussHermite_lognorm(par.sigma_xi,par.Nxi)
    
    # End of period assets
    par.grid_a = np.nan + np.zeros([par.T,par.Na])
    for t in range(par.T):
        par.grid_a[t,:] = tools.nonlinspace(0+1e-6,par.a_max,par.Na,par.a_phi)

    # Cash-on-hand
    par.grid_m =  np.concatenate([np.linspace(0+1e-6,1-1e-6,par.Nm_b), tools.nonlinspace(1+1e-6,par.m_max,par.Nm-par.Nm_b,par.m_phi)])    # Permanent income

    # Permanent income
    par.grid_p = tools.nonlinspace(0+1e-4,par.p_max,par.Np,par.p_phi)

    # Set seed
    np.random.seed(2020)

    return par

def solve(par):
    
    # Initialize
    class sol: pass

    shape=(par.T,2,par.Nm,par.Np)
    sol.m = np.nan+np.zeros(shape)
    sol.c = np.nan+np.zeros(shape)
    sol.v = np.nan+np.zeros(shape)
    
    # Last period, (= consume all) 
    for i_p in range(par.Np):
        for z_plus in range(2):
            sol.m[par.T-1,z_plus,:,i_p] = par.grid_m
            sol.c[par.T-1,z_plus,:,i_p] = par.grid_m
            sol.v[par.T-1,z_plus,:,i_p] = util(sol.c[par.T-1,z_plus,:,i_p],z_plus,par)

    # Before last period
    for t in range(par.T-2,-1,-1):

        #Choice specific fundtion
        for i_p, p in enumerate(par.grid_p):
           
            for z_plus in range(2):

                # Solve model with EGM
                c,v = EGM(sol,z_plus,p,t,par)
                sol.c[t,z_plus,:,i_p] = c
                sol.v[t,z_plus,:,i_p] = v
               
    return sol

def EGM (sol,z_plus,p, t,par): 

    if z_plus == 1:     #Retired =  Not working
        w_raw, avg_marg_u_plus = retired(sol,z_plus,p,t,par)
    else:               # Working
        w_raw, avg_marg_u_plus = working(sol,z_plus,p,t,par)

    # raw c, m and v
    c_raw = inv_marg_util(par.beta*par.R*avg_marg_u_plus,par)
    m_raw = c_raw + par.grid_a[t,:]
   
    # Upper Envelope
    c,v = upper_envelope(t,z_plus,c_raw,m_raw,w_raw,par)
    
    return c,v


def retired(sol,z_plus,p, t,par):
    # Prepare
    w = np.ones((par.Na))
    a = par.grid_a[t,:]
    p = np.tile(p,par.Na)

    # Next period states
    p_plus = p
    m_plus = par.R*a+par.kappa*p_plus

    # value
    w_raw = tools.interp_2d_vec(par.grid_m,par.grid_p,sol.v[t+1,z_plus], m_plus, p_plus)
    
    # Consumption
    c_plus = tools.interp_2d_vec(par.grid_m,par.grid_p,sol.c[t+1,z_plus], m_plus, p_plus)
       
    #Marginal utility
    marg_u_plus = marg_util(c_plus,par)

    #Expected average marginal utility
    avg_marg_u_plus = marg_u_plus*w 

    return w_raw, avg_marg_u_plus

def working(sol,z_plus,p, t,par):
    # Prepare
    xi = np.tile(par.xi,par.Na)
    p = np.tile(p, par.Na*par.Nxi)
    a = np.repeat(par.grid_a[t],par.Nxi) 
    w = np.tile(par.xi_w,(par.Na,1))


    # Next period states
    p_plus = xi*p
    m_plus = par.R*a+par.W*p_plus

    # Value, consumption, marg_util
    shape = (2,m_plus.size)
    v_plus = np.nan+np.zeros(shape)
    c_plus = np.nan+np.zeros(shape)
    marg_u_plus = np.nan+np.zeros(shape)

    for i in range(2): #Range over working and not working next period
        # Choice specific value
        v_plus[i,:] = tools.interp_2d_vec(par.grid_m,par.grid_p,sol.v[t+1,i], m_plus, p_plus)
    
        # Choice specific consumption    
        c_plus[i,:] = tools.interp_2d_vec(par.grid_m,par.grid_p,sol.c[t+1,i], m_plus, p_plus)
       
        # Choice specific Marginal utility
        marg_u_plus[i,:] = marg_util(c_plus[i,:], par) 
       
    # Expected value
    V_plus, prob = logsum(v_plus[0],v_plus[1],par.sigma_eta) 
    w_raw = w*np.reshape(V_plus,(par.Na,par.Nxi))
    w_raw = np.sum(w_raw,1)
    marg_u_plus = prob[0,:]*marg_u_plus[0] + prob[1,:]*marg_u_plus[1]  

    #Expected  average marg. utility
    avg_marg_u_plus = w*np.reshape(marg_u_plus,(par.Na,par.Nxi))
    avg_marg_u_plus = np.sum(avg_marg_u_plus ,1)

    return w_raw, avg_marg_u_plus

def upper_envelope(t,z_plus,c_raw,m_raw,w_raw,par):
    
    # Add a point at the bottom
    c_raw = np.append(1e-6,c_raw)  
    m_raw = np.append(1e-6,m_raw) 
    a_raw = np.append(0,par.grid_a[t,:]) 
    w_raw = np.append(w_raw[0],w_raw)

    # Initialize c and v   
    c = np.nan + np.zeros((par.Nm))
    v = -np.inf + np.zeros((par.Nm))
    
    # Loop through the endogenous grid
    size_m_raw = m_raw.size
    for i in range(size_m_raw-1):    

        c_now = c_raw[i]        
        m_low = m_raw[i]
        m_high = m_raw[i+1]
        c_slope = (c_raw[i+1]-c_now)/(m_high-m_low)
        
        w_now = w_raw[i]
        a_low = a_raw[i]
        a_high = a_raw[i+1]
        w_slope = (w_raw[i+1]-w_now)/(a_high-a_low)


        # Loop through the common grid
        for j, m_now in enumerate(par.grid_m):

            interp = (m_now >= m_low) and (m_now <= m_high) 
            extrap_above = (i == size_m_raw-1) and (m_now > m_high)

            if interp or extrap_above:
                # Consumption
                c_guess = c_now+c_slope*(m_now-m_low)
                
                # post-decision values
                a_guess = m_now - c_guess
                w = w_now+w_slope*(a_guess-a_low)
                
                # Value of choice
                v_guess = util(c_guess,z_plus,par)+par.beta*w
                
                # Update
                if v_guess >v[j]:
                    v[j]=v_guess
                    c[j]=c_guess

    return c,v


# FUNCTIONS
def util(c,L,par):
    return ((c**(1.0-par.rho))/(1.0-par.rho)-par.alpha*(1-L))


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
