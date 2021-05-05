# Define utility function of the households
import numpy as np
import model_2
import model_1d

# Utility
def u(c,par):

    if par.eta == 1.0:
        u = np.log(c)

    else:
        u = (c**(1-par.eta) - 1.0) / (1.0 - par.eta)

    return u

# Marginal utility
def marg_u(c,par):
    return c**(-par.eta)

# Inverse marginal utility
def inv_marg_u(u,par):
    return u**(-1/par.eta)



    # Utility with housing
def u_with_housing(c,h,par):

    if par.eta == 1.0:
        u1 = np.log(c)

    else:
        u1 = (c**(1-par.eta) - 1.0) / (1.0 - par.eta)
    
    #u2 = par.b*h**par.alpha
    u2 = h

    u = u1 + u2
    return u
