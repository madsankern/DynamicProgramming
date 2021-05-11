import utility as util
import numpy as np

# Solve the last period of the consumer problem

def solve(sol, par):

    for n in range(2): # Loop over housing state
        for m_i,m in enumerate(par.grid_m): # Loop over exogeneous asset grid
                 
                if n == 0 :

                    # Cannot buy a house
                    if m < par.ph: 
                        sol.c[n,m_i] = m
                        sol.h[n,m_i] = 0

                    # Can buy a house
                    else:
                        u_gain = util.u(m, 0, par) - util.u(m - par.ph, 1, par) # Gain from not buying a house

                        if u_gain >= 0:  # If not buying a house is optimal
                            sol.c[n,m_i] = m
                            sol.h[n,m_i] = 0
                        
                        else: # If buying a house is optimal
                            sol.c[n,m_i] = m - par.ph
                            sol.h[n,m_i] = 1

                if n == 1 :

                    u_gain = util.u(m, 1, par) - util.u(m + par.ph, 0, par) # Gain from not selling the house

                    if u_gain >= 0:
                        sol.c[n,m_i] = m
                        sol.h[n,m_i] = 1
                    
                    else:
                        sol.c[n,m_i] = m + par.ph
                        sol.h[n,m_i] = 0

                # Compute value of choice
                sol.v[n,m_i] = util.u(sol.c[n,m_i], sol.h[n,m_i], par)