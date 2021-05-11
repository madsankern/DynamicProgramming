# Define utility function of the households
import numpy as np

# Utility
def u(c,par):

    if par.eta == 1.0:
        u = np.log(c)

    else:
        u = (c**(1-par.eta) - 1.0) / (1.0 - par.eta)

    return u

# Utility
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