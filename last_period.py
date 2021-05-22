import utility as util
import numpy as np

# Solves the last period of the consumer problem

def solve(sol, par):

    grid_last = np.linspace(par.m_min, par.m_max, par.Nm + par.N_bottom)

    if np.size(sol.c[0,:]) >= 101:
        grid = grid_last
    else:
        grid = par.grid_m

    for n in range(2): # Loop over housing state
        for m_i,m in enumerate(grid): # Loop over exogeneous asset grid
                 
                if n == 0:

                    # Cannot buy a house
                    if m < par.ph: 
                        sol.c[n,m_i] = m
                        sol.h[n,m_i] = 0

                    # Can buy a house
                    else:
                        u_dif = util.u_h(m, 0, par) - util.u_h(m - par.ph, 1, par) # dif in utility from choice

                        if u_dif >= 0:  # If not buying a house is optimal
                            sol.c[n,m_i] = m
                            sol.h[n,m_i] = 0
                        
                        else: # If buying a house is optimal
                            sol.c[n,m_i] = m - par.ph
                            sol.h[n,m_i] = 1

                if n == 1:

                    u_dif = util.u_h(m, 1, par) - util.u_h(m + par.ph, 0, par) # dif in utility from choice

                    if u_dif >= 0:
                        sol.c[n,m_i] = m
                        sol.h[n,m_i] = 1
                    
                    else:
                        sol.c[n,m_i] = m + par.ph
                        sol.h[n,m_i] = 0

                # Compute value of choice
                sol.v[n,m_i] = util.u_h(sol.c[n,m_i], sol.h[n,m_i], par)