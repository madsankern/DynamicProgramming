# import packages 
import numpy as np
import tools

def setup():
    class par: pass

    par.T = 20

    # Model parameters
    par.beta = 0.96
    par.rho = 2
    par.alpha = 0.75
    par.R = 1.04
    par.W = 1
    par.sigma_xi = 0.0
    par.sigma_eta = 0.2

    
    # Grids and numerical integration
    par.a_max = 10
    par.a_phi = 1.1 # Curvature parameters
    par.Nxi = 1     
    par.Na = 150
    par.N_bottom = 10 # Number of points at the bottom in the G2-EGM algorithm
    
    return par

def create_grids(par):

    # Check parameters
    assert (par.rho >= 0), 'not rho > 0'

    # Shocks
    par.xi,par.xi_w = tools.GaussHermite_lognorm(par.sigma_xi,par.Nxi)
    
    # End of period assets
    par.grid_a = np.nan + np.zeros([par.T,par.Na])
    for t in range(par.T):
        par.grid_a[t,:] = tools.nonlinspace(0+1e-8,par.a_max,par.Na,par.a_phi)

    # Set seed
    np.random.seed(2020)

    return par

def solve(par):
    
    # Initialize
    class sol: pass

    shape=(par.T,2,par.Na+par.N_bottom)
    sol.m = np.nan+np.zeros(shape)
    sol.c = np.nan+np.zeros(shape)
    sol.v = np.nan+np.zeros(shape)
    
    # Last period, (= consume all) 
    for z_plus in range(2):
        sol.m[par.T-1,z_plus,:] = np.linspace(0+1e-8,par.a_max,par.Na+par.N_bottom)
        sol.c[par.T-1,z_plus,:] = np.linspace(0+1e-8,par.a_max,par.Na+par.N_bottom)
        sol.v[par.T-1,z_plus,:] = util(sol.c[par.T-1,z_plus,:],z_plus,par)

    # Before last period
    for t in range(par.T-2,-1,-1):

        #Choice specific fundtion
        for z_plus in range(2):

            # Solve model with EGM
            m,c,v = EGM(sol,z_plus,t,par)   

            # Add points at the constraints
            m_con = np.linspace(0+1e-8,m[0]-1e-8,par.N_bottom)
            c_con = m_con.copy()
            v_con = value_of_choice(m_con,c_con,z_plus,t,sol,par)

            sol.m[t,z_plus] = np.append(m_con, m)
            sol.c[t,z_plus] = np.append(c_con, c)
            sol.v[t,z_plus] = np.append(v_con, v)
    return sol


def EGM (sol,z_plus, t,par):
    # Prepare
    if z_plus == 1:     #Retired
        w_raw, avg_marg_u_plus = retired(sol,z_plus,t,par)
    else:               #Working
        w_raw, avg_marg_u_plus = working(sol,z_plus,t,par)


    # raw c, m and v
    c_raw = inv_marg_util(par.beta*par.R*avg_marg_u_plus,par)
    m_raw = c_raw + par.grid_a[t,:]
    v_raw = util(c_raw,z_plus,par) + par.beta * w_raw

    # UPPER ENVELOPE 

    # Reorderining making G_m strictly increasing 
    m = sorted(m_raw)  # alternatively, choose a common grid exogeneously. This, however, creates many points around the kink
    I = m_raw
    c = [x for _,x in sorted(zip(I,c_raw))] 
    v = [x for _,x in sorted(zip(I,v_raw))]

    #If retired: No Upper-envelope
    if z_plus == 1:
        return m,c,v

    # Loop through the endogenous grid
    for i in range(np.size(m_raw)-2):
        m_low = m_raw[i]
        m_high = m_raw[i+1]
        c_slope = (c_raw[i+1]-c_raw[i])/(m_high-m_low)

        # Loop through the common grid
        for j in range(len(m)):

            if  m[j]>=m_low and m[j]<=m_high:

                c_guess = c_raw[i]+c_slope*(m[j]-m_low)
                v_guess = value_of_choice(m[j],c_guess,z_plus,t,sol,par)
                
                # Update
                if v_guess >v[j]:
                    v[j]=v_guess
                    c[j]=c_guess



    return m,c,v

def retired(sol,z_plus,t,par):
    # Prepare 
    w = np.ones((par.Na))
    a = par.grid_a[t,:]
    xi = np.zeros(par.Na)

    # Next period states
    m_plus = par.R*a+par.W*xi

    #value
    w_raw=tools.interp_linear_1d(sol.m[t+1,z_plus,par.N_bottom:],sol.v[t+1,z_plus,par.N_bottom:], m_plus)
        
    # Consumption
    c_plus = tools.interp_linear_1d(sol.m[t+1,z_plus,par.N_bottom:],sol.c[t+1,z_plus,par.N_bottom:], m_plus)
        
    #Marginal utility
    marg_u_plus = marg_util(c_plus,par)

    #Expected average marginal utility
    avg_marg_u_plus = marg_u_plus*w

    return w_raw, avg_marg_u_plus


def working(sol,z_plus,t,par):
    # Prepare
    xi = np.tile(par.xi,par.Na)
    a = np.repeat(par.grid_a[t],par.Nxi) 
    w = np.tile(par.xi_w,(par.Na,1))

    # Next period states
    m_plus = par.R*a+par.W*xi

    shape = (2,m_plus.size)
    v_plus = np.nan+np.zeros(shape)
    c_plus = np.nan+np.zeros(shape)
    marg_u_plus = np.nan+np.zeros(shape)

    for i in range(2): #Range over working and not working next period
        # Choice specific value
        v_plus[i,:]=tools.interp_linear_1d(sol.m[t+1,i,par.N_bottom:],sol.v[t+1,i,par.N_bottom:], m_plus)
        
        #Choice specific consumption
        c_plus[i,:] = tools.interp_linear_1d(sol.m[t+1,i,par.N_bottom:],sol.c[t+1,i,par.N_bottom:], m_plus)
        
        # Choice specific Marginal utility
        marg_u_plus[i,:] = marg_util(c_plus[i,:] ,par) 

    # Expected value
    V_plus, prob = logsum(v_plus[0],v_plus[1],par.sigma_eta)
    w_raw = w*np.reshape(V_plus,(par.Na,par.Nxi))
    w_raw = np.sum(w_raw,1)
    marg_u_plus = prob[0,:]*marg_u_plus[0] + prob[1,:]*marg_u_plus[1]  


    #Expected  average marg. utility
    avg_marg_u_plus = w*np.reshape(marg_u_plus,(par.Na,par.Nxi))
    avg_marg_u_plus = np.sum(avg_marg_u_plus ,1)

    return w_raw, avg_marg_u_plus


def value_of_choice(m,c,L,t,sol,par):
    
    xi_w_mat = np.tile(par.xi_w,(c.size,1))
    xi_mat = np.tile(par.xi,(c.size))
    
    
    # Next period ressources
    a = np.repeat(m-c,(par.xi.size))
    m_plus = par.R * a + par.W*xi_mat

    # Next-period value
    v_plus0 = tools.interp_linear_1d(sol.m[t+1,0,par.N_bottom:],sol.v[t+1,0,par.N_bottom:], m_plus)
    v_plus1 = tools.interp_linear_1d(sol.m[t+1,1,par.N_bottom:],sol.v[t+1,1,par.N_bottom:], m_plus)
    V_plus, _ = logsum(v_plus0,v_plus1,par.sigma_eta)
    V_plus = np.reshape(V_plus,(c.size,par.xi_w.size))
    V_plus = np.sum(xi_w_mat*V_plus,1)

    # This period value
    v = util(c,L,par)+par.beta*V_plus
    return v


#FUNCTIONS

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
