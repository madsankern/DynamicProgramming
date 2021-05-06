# Defines the simpel 1d model to be solved

# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import vfi
import egm
import utility as util
import scipy.optimize as optimize


class model_1d():

    def __init__(self,name=None):
        """ defines default attributes """

        # Names
        self.par = SimpleNamespace()
        self.sol_vfi = SimpleNamespace()
        self.sol_egm = SimpleNamespace()
        # self.sol_fd = SimpleNamespace()

    ###########
    ## Setup ##
    ###########
    # Setup parameters used for all solvers
    # for the 1d model of consumption

    def setup(self):

        # Initialize
        par = self.par

        # Model
        par.beta =  0.96
        par.rho = 1.0 - par.beta
        par.eta = 1.0
        
        par.r = 0.01
        par.y1 = 1.0
        par.y2 = 1.5
        par.y = np.array([par.y1, par.y2])
        
        par.P_11 = 0.6
        par.P_22 = 0.9
        par.P = np.array([[par.P_11, 1 - par.P_11], [1 - par.P_22, par.P_22]]) # Transition matrix

        # Settings - note the naming in the grid
        par.Na = 100 # Remember this
        par.a_min = 1e-8 # Slightly above 0 for numerical reasons
        par.a_max = 20 # Largest point in a grid
        par.max_iter = 500 # Maximum nr of iterations
        par.tol_vfi = 10e-6
        par.tol_egm = 10e-6
        par.tol_fd = 10e-6
        
    # Grids of assets. Either pre or post decision
    # dependent on the solver used
    def create_grids(self):

        par = self.par
        
        # Pre desicion
        par.grid_m = np.linspace(par.a_min, par.a_max, par.Na)
        
        # Post desicion
        par.grid_a = np.linspace(par.a_min, par.a_max, par.Na)
        
        # Convert these to nonlinspace later.
        # Easier just to use two different grids
        # for VFI and EGM

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

        state1 = 1 # UNEMPLOYMENT STATE
        state2 = 0 # EMPLOYMENT STATE

        sol.it = 0 # Iteration counter
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
        sol.c = sol.a.copy() # Consume everyting - this could be improved

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
        sol.a[:,0] = 0
        sol.c[:,0] = 0