import utility as util
import numpy as np

# Solves the last period of the consumer problem

def solve(sol, par):

    for n in range(2): # Loop over housing state
        for m_i,m in enumerate(par.grid_m): # Loop over exogeneous asset grid
                 
                if n == 0:

                    # Cannot buy a house
                    if m < par.ph: 
                        sol.c[n,m_i+par.N_bottom] = m
                        sol.h[n,m_i+par.N_bottom] = 0

                    # Can buy a house
                    else:
                        u_dif = util.u_h(m, 0, par) - util.u_h(m - par.ph, 1, par) # dif in utility from choice

                        if u_dif >= 0:  # If not buying a house is optimal
                            sol.c[n,m_i+par.N_bottom] = m
                            sol.h[n,m_i+par.N_bottom] = 0
                        
                        else: # If buying a house is optimal
                            sol.c[n,m_i+par.N_bottom] = m - par.ph
                            sol.h[n,m_i+par.N_bottom] = 1

                if n == 1:

                    u_dif = util.u_h(m, 1, par) - util.u_h(m + par.ph, 0, par) # dif in utility from choice

                    if u_dif >= 0:
                        sol.c[n,m_i+par.N_bottom] = m
                        sol.h[n,m_i+par.N_bottom] = 1
                    
                    else:
                        sol.c[n,m_i+par.N_bottom] = m + par.ph
                        sol.h[n,m_i+par.N_bottom] = 0

                # Compute value of choice
                sol.v[n,m_i+par.N_bottom] = util.u_h(sol.c[n,m_i+par.N_bottom], sol.h[n,m_i+par.N_bottom], par)