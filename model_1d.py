# Defines the simpel 1d model to be solved

# import packages 
import numpy as np
import tools
from types import SimpleNamespace
import vfi
import utility as util
import scipy.optimize as optimize


class model_1d():

    def __init__(self,name=None):
        """ defines default attributes """

        # Names
        self.par = SimpleNamespace()
        self.sol_vfi = SimpleNamespace()
        # self.sol_egm = SimpleNamespace()
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

        return par

    def solve_vfi(self):

        # Initialize
        par = self.par
        sol = self.sol_vfi
        
        # Initialize
        shape = (np.size(par.y),1) # Shape to fit nr of income states
        sol.c = np.tile(par.grid_a.copy(), shape) # Initial guess - consume all
        sol.v = util.u(sol.c,par) # Utility of consumption
        sol.a = par.grid_m.copy() # Copy the exogenous asset grid for consistency (with EGM algortihm)

        state1 = 1 # UNEMPLOYMENT STATE
        state2 = 0 # EMPLOYMENT STATE

        sol.it = 0 # Iteration counter
        sol.delta = 10000.0 # Distance between iterations

        # Iterate untill convergence
        while (sol.delta >= par.tol_vfi and sol.it < par.max_iter):
            
            # Use last iteration as the continuation value. See slides if confused
            v_next = sol.v.copy()

            # Loop over asset grid
            for m_i,m in enumerate(par.grid_m):

                # FUNCTIONS BELOW CAN BE WRITTEN AS LOOP - for i=0,1 - AND BE STORED IN AN ARRAY/LIST WITH TWO ENTRIES - a la res[i]=optimize.minimize....

                # Minimize the minus the value function wrt consumption conditional on unemployment state
                obj_fun = lambda x : - vfi.value_of_choice(x,m,par.grid_m,v_next[0,:],par,state1)
                res_1 = optimize.minimize_scalar(obj_fun, bounds=[0,m+1.0e-4], method='bounded')

                # Minimize the minus the value function wrt consumption conditional on employment state
                obj_fun = lambda x : - vfi.value_of_choice(x,m,par.grid_m,v_next[1,:],par,state2)
                res_2 = optimize.minimize_scalar(obj_fun, bounds=[0,m+1.0e-4], method='bounded')
                
                # Unpack solutions
                # State 1
                sol.v[0,m_i] = -res_1.fun
                sol.c[0,m_i] = res_1.x

                # State 2
                sol.v[1,m_i] = -res_2.fun
                sol.c[1,m_i] = res_2.x
                    
            # Update iteration parameters
            sol.it += 1
            sol.delta = max( max(abs(sol.v[0] - v_next[0])), max(abs(sol.v[1] - v_next[1]))) # check this, is this optimal  
