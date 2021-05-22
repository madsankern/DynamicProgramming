import numpy as np

# Without housing
def u(c,par):

    if par.eta == 1.0:
        u = np.log(c)

    else:
        u = (c**(1-par.eta) - 1.0) / (1.0 - par.eta)

    return u

# With housing
def u_h(c,h,par):

    if par.eta == 1.0:
        u = np.log(c) + par.kappa*h

    else:
        u = (c**(1-par.eta) - 1.0) / (1.0 - par.eta) + par.kappa*h

    return u

# Marginal utility
def marg_u(c,par):
    return c**(-par.eta)

# Inverse marginal utility
def inv_marg_u(u,par):
    return u**(-1.0/par.eta)

# Inverse utility
def inv_u(u,par):

    if par.eta == 1.0:
        val = 1.0/u
    
    else:
        val = ((1.0-par.eta)*u + 1.0)**(1.0/(1.0-par.eta))

    return val