# Script to set up the parameters of the model

import numpy as np
import tools
import scipy.optimize as optimize

# To do:
# Use nonlinspace for grids

# Setup household
def setup():
    class par: pass # Define class for parameters
    
    # Deep parameters
    #par.beta = 0.98 # discount rate (ORIGINAL)
    par.beta = 0.90 # discount rate - higher discount rate for faster convergence of VFI
    par.eta = 1.0 # Elasticity parameter
    par.alpha = 0.8 # alpha>1. Parameter for housing utility which takes the form, b*h^a
    par.b = 6 # Parameter for housing utility which takes the form, b*h^a


    # Institutional parameters
    #par.r = 0.01 # net rate of return (must be lower than 1/beta i think) (ORIGINAL)
    par.r = 0.035 # net rate of return (must be lower than 1/beta i think)
    par.hp = 1 # Housing price
    par.h_min = 2.3 # Minimum house size
    # par.y1 = 1.0 # Low income
    # par.y2 = 1.5 # High income
    # par.y = np.array([[par.y1, par.y2]]) # Collect income as an array

    # Income process
    par.y1 = 1.0
    par.y2 = 1.5
    par.y = np.array([par.y1, par.y2])
    
    par.pi = 0.5 # Probability parameter
    par.Pi = np.array([par.pi, 1 - par.pi]) # Probability weights

    par.P_11 = 0.6 # Prob of staying in state 1 (unemployment state)
    par.P_22 = 0.9 # Prob of staying in state 2 (employment state)
    par.P = np.array([[par.P_11, 1 - par.P_11], [1 - par.P_22, par.P_22]]) # Transition matrix
    
    # Settings parameters
    par.num_a = 100 # Point in the a grid
    par.a_min = 1e-8 # Minimum assets
    par.a_max = 20 # Largest point in a grid
    par.max_iter = 500 # Maximum nr of iterations
    par.tol_vfi = 10e-4 # Tolerance for convergence (VFI) (ORIGINAL)
    par.tol_egm = 10e-4 # Tolerance for convergence (EGM)
    par.tol_fd = 10e-6

    # We need to figure out the correct tolerance. EGM is much more sensitive to the tolerance of choice

    # Extra parameters for FD/continuous time model
    par.rho = 0.02 # Discount factor    

    #Setup grids
    setup_grids(par)

    return par

# Grids of assets
def setup_grids(par):
    
    # Grid of assets
    par.grid_a = np.linspace(par.a_min, par.a_max, par.num_a) # Maybe use non-linear grid at some point? *** God idé!
    
    # Exogenous grid of end-of-period assets := s_t. Used for EGM
    par.grid_s = np.linspace(par.a_min, par.a_max, par.num_a)
    
    return par
