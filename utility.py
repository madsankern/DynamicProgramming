# Define utility function of the households
import numpy as np
import model

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