# Defines the simpel 1d model to be solved

# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import vfi
import egm
import fd
import utility as util
# import scipy.optimize as optimize
import last_period


class model_1d():

    def __init__(self,name=None):
        """ defines default attributes """

        # Names
        self.par = SimpleNamespace()
        self.sol_vfi = SimpleNamespace()
        self.sol_nvfi = SimpleNamespace()
        self.sol_egm = SimpleNamespace()
        self.sol_fd = SimpleNamespace()
        self.sol_negm = SimpleNamespace()

    ###########
    ## Setup ##
    ###########
    # Setup parameters used for all solvers
    # for the 1d model of consumption

    def setup(self):

        # Initialize
        par = self.par

        # Model
        par.beta =  0.98
        par.rho = 1.0 - par.beta
        par.eta = 0.5
        
        par.r = 0.01
        par.y1 = 1.0
        par.y2 = 2.0
        par.y = np.array([par.y1, par.y2])
        
        par.P_11 = 0.6
        par.P_22 = 0.9
        par.P = np.array([[par.P_11, 1 - par.P_11], [1 - par.P_22, par.P_22]]) # Transition matrix

        # Poisson jumps - rewrite to correspond to P above
        par.pi_list = [[-0.1, 0.1], [0.1, -0.1]]
        par.pi = np.asarray(par.pi_list)

        # Extra parameters for housing
        par.kappa = 2.25
        par.ph = 3.0

        # Grid settings
        par.Nm = 110
        par.m_max = 10.0
        par.m_min = 1e-4

        par.Na = 100
        par.a_min = 1e-4
        par.a_max = par.m_max + 1.0 # Check this out later

        par.Nx = 100
        par.x_max = par.m_max + par.ph # add price of selling house to the top of the x grid (grid when selling/buying house)
        par.x_min = 1e-4
        
        par.max_iter = 5
        par.tol_vfi = 10e-4
        par.tol_egm = 10e-4
        par.tol_fd = 10e-4

        par.N_bottom = 10 

    # Asset grids
    def create_grids(self):

        par = self.par
        
        # Pre desicion
        par.grid_m = np.linspace(par.m_min, par.m_max, par.Nm)
        
        # Post desicion
        par.grid_a = np.linspace(par.a_min, par.a_max, par.Na)
        
        # x grid
        par.grid_x = np.linspace(par.x_min, par.x_max, par.Nx)

        # housing grid - discrete choice
        par.grid_n = np.array([0,1])


        # Convert these to nonlinspace later.

###########################################
##### SIMPLE CONUSMPTION SAVING MODEL #####
###########################################

    ##############################
    ## Value function iteration ##
    ##############################

    def solve_vfi(self):

        # Initialize
        par = self.par
        sol = self.sol_vfi
        
        # Initialize
        shape = (np.size(par.y),1) # Shape to fit nr of income states
        sol.c = np.tile(par.grid_a.copy(), shape) # Initial guess - consume all
        sol.v = util.u(sol.c,par) # Utility of consumption
        sol.m = par.grid_m.copy() # Copy the exogenous asset grid for consistency (with EGM algortihm)

        state1 = 0 # UNEMPLOYMENT STATE
        state2 = 1 # EMPLOYMENT STATE

        sol.it = 0 # Iteration counters
        sol.delta = 1000.0 # Distance between iterations

        # Iterate untill convergence
        while (sol.delta >= par.tol_vfi and sol.it < par.max_iter):
            
            # Use last iteration as the continuation value. See slides if confused
            v_next = sol.v.copy()

            # Find optimal c given v_next
            sol = vfi.solve(sol, par, v_next, state1, state2)
                    
            # Update iteration parameters
            sol.it += 1
            sol.delta = max( max(abs(sol.v[0] - v_next[0])), max(abs(sol.v[1] - v_next[1]))) # Update this maybe

    #############################
    ## Endogeneous grid method ##
    #############################

    def solve_egm(self):

        # Initialize
        par = self.par
        sol = self.sol_egm

        # Shape parameter for the solution vector
        shape = (np.size(par.y),1)
        
        # Initial guess is like a 'last period' choice - consume everything
        sol.m = np.tile(np.linspace(par.a_min,par.a_max,par.Na+1), shape) # a is pre descision, so for any state consume everything.
        sol.c = sol.m.copy() # Consume everyting - this could be improved
        sol.v = util.u(sol.c,par) # Utility of consumption

        sol.it = 0 # Iteration counter
        sol.delta = 1000.0 # Difference between iterations

        # Iterate value function until convergence or break if no convergence
        while (sol.delta >= par.tol_egm and sol.it < par.max_iter):

            # Use last iteration to compute the continuation value
            # therefore, copy c and a grid from last iteration.
            c_next = sol.c.copy()
            m_next = sol.m.copy()

            # Call EGM function here
            sol = egm.solve(sol, par, c_next, m_next)

        # add zero consumption
        sol.m[:,0] = 1.0e-4
        sol.c[:,0] = 1.0e-4

    ##############################
    ## Finite difference method ##
    ##############################
    
    def solve_fd(self):

        # Initialize
        par = self.par
        sol = self.sol_fd

        sol = fd.solve(par,sol)


##########################################
#### DISCRETE-CONTINUOUS CHOICE MODEL ####
##########################################

################
## Nested VFI ##
################
    
    def solve_vfi_dc(self):

        # Solve my backwards induction. In the last period, solve by last_period.py. 
        # Then solve backwards until covergence

        # Initialize
        par = self.par
        sol = self.sol_nvfi

        # Shape parameter
        shape = (2,np.size(par.grid_m)) #  Row for each state of housing

        # Initialize
        sol.m = np.tile(np.linspace(par.a_min,par.a_max,par.Na+1), shape)
        sol.c = np.zeros(shape) + np.nan
        sol.h = np.zeros(shape) + np.nan
        sol.v = np.zeros(shape) + np.nan

        # Solve last period
        last_period.solve(sol,par)

        sol.it = 0 # Iteration counter
        sol.delta = 1000.0 # Difference between iterations

        # Iterate value function until convergence or break if no convergence
        while (sol.delta >= par.tol_egm and sol.it < par.max_iter):

            # Continuation value
            c_next = sol.c.copy()
            h_next = sol.h.copy()            
            v_next = sol.v.copy()

            # Solve the keeper problem
            sol = vfi.solve_dc(sol, par, v_next, c_next, h_next)

            sol.it += 1
            
################
## Nested EGM ##
################
    
    def solve_negm_dc(self):

        # Solve my backwards induction. In the last period, solve by last_period.py. 
        # Then solve backwards until covergence

        # Initialize
        par = self.par
        sol = self.sol_negm

        # Shape parameter
        shape = (2,np.size(par.grid_a) + par.N_bottom) #  Row for each state of housing and columns for exogenous end-of-period asset grid 

        # Initialize
        sol.m = np.tile(np.linspace(par.a_min,par.a_max,par.Na + par.N_bottom), (2,1))
        sol.c = np.zeros(shape) + np.nan
        sol.h = np.zeros(shape) + np.nan
        sol.v = np.zeros(shape) + np.nan

        # Solve last period
        last_period.solve(sol,par)

        sol.it = 0 # Iteration counter
        sol.delta = 1000.0 # Difference between iterations

        # Iterate value function until convergence or break if no convergence
        while (sol.delta >= par.tol_egm and sol.it < par.max_iter):
            
            # Continuation value
            m_next = sol.m.copy()
            c_next = sol.c.copy()
            h_next = sol.h.copy()            
            v_next = sol.v.copy()

            # Solve the keeper problem
            sol = egm.solve_dc(sol, par, v_next, c_next, h_next, m_next)

            # # Add points at the constraints (alternatively, look at exercise 1) --> DONE INSIDE EGM.PY
            # m_con = np.linspace(0+1e-8,m[0]-1e-8,par.N_bottom)
            # c_con = m_con.copy()
            # v_con = value_of_choice(m_con,c_con,z_plus,t,sol,par)

            for n in range(2):
                sol.m[n] = np.append(m_con_'n', m)
                sol.c[n] = np.append(c_con_'n', c)
                sol.h[n] = np.append(v_con_'n', v)
                sol.v[n] = np.append(v_con_'n', v)
                
            sol.it += 1
            
            
