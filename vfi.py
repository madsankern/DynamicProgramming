# Solve the model using value function iteration

# To do:
# Rewrite value_of_choice to vectorize
# Vectorize inner loop in solve_VFI

import numpy as np
import tools
import scipy.optimize as optimize
import utility as util

def solve_VFI(par):
    
    # Initialize solution class
    class sol: pass
    sol.c = par.grid_a.copy() # Initial guess is to consume everything
    sol.v = util.u(sol.c,par) # Utility of consumption
    sol.a = par.grid_a.copy() # Copy the exogenous asset grid for consistency (with EGM algortihm) -- Jeg kan se, at denne initialisering er nødvendig for at kunne
    # plotte på et a-grid, men vi initialisere jo allerede griddet i setup(), så måske kan man kalde par.grid_a direkte, når der plottes? Har prøvet, men den klager.
    
    sol.it = 0 # Iteration counter
    sol.delta = 1000.0 # Difference between two iterations
    
    # Iterate value function until convergence or break if no convergence
    while (sol.delta >= par.tol_vfi and sol.it < par.max_iter):
        
        # Use last iteration as the continuation value. See slides if confused
        v_next = sol.v.copy()
        
        # Loop over asset grid
        for i_a,a in enumerate(par.grid_a):
            
            # Minimize the minus the value function wrt consumption
            obj_fun = lambda x : - value_of_choice(x,a,par.grid_a,v_next,par)
            res = optimize.minimize_scalar(obj_fun, bounds=[0,a+1.0e-4], method='bounded')
            
            # Unpack solution
            sol.v[i_a] = -res.fun
            sol.c[i_a] = res.x
            
        # Update iteration parameters
        sol.it += 1
        sol.delta = max(abs(sol.v - v_next))

    return sol

        
# Function that returns value of consumption choice
def value_of_choice(x,a,a_next,v_next,par):
    
    # Unpack consumption (choice variable)
    c = x

    # Intialize expected continuation value
    Ev_next = 0.0
    
    # Loop over each possible state
    for i in [0,1]:
        
        # Next periods state for each income level
        a_plus = par.y[i] + (1+par.r)*(a - c)
        
        #Interpolate continuation given state a_plus
        v_plus = tools.interp_linear_1d_scalar(a_next,v_next,a_plus)
    
        # Append continuation value to calculate expected value
        Ev_next += par.Pi[i] * v_plus
        
    # Value of choice
    v_guess = util.u(c,par) + par.beta * Ev_next

    return v_guess

def solve_VFI_2d(par):

    # Initialize solution class
    class sol: pass
    shape = (np.size(par.y),1)
    sol.c = np.tile(par.grid_a.copy(), shape) # Initial guess is to consume everything for each state
    sol.v = util.u(sol.c,par) # Utility of consumption
    sol.a = par.grid_a.copy() # Copy the exogenous asset grid for consistency (with EGM algortihm)

    sol.it = 0 # Iteration counter
    sol.delta = 1000.0 # Difference between two iterations

    # Iterate value function until convergence or break if no convergence
    while (sol.delta >= par.tol_vfi and sol.it < par.max_iter):
        
        # Use last iteration as the continuation value. See slides if confused
        v_next = sol.v.copy()

         # Loop over asset grid
        for i_a,a in enumerate(par.grid_a):
            
            # Minimize the minus the value function wrt consumption conditional on unemployment state
            obj_fun = lambda x : - value_of_choice_2d(x,a,par.grid_a,v_next[0,:],par,1)
            res_1 = optimize.minimize_scalar(obj_fun, bounds=[0,a+1.0e-4], method='bounded')

            # Minimize the minus the value function wrt consumption conditional on employment state
            obj_fun = lambda x : - value_of_choice_2d(x,a,par.grid_a,v_next[1,:],par,0)
            res_2 = optimize.minimize_scalar(obj_fun, bounds=[0,a+1.0e-4], method='bounded')
            
            # Unpack solutions
            # State 1
            sol.v[0,i_a] = -res_1.fun
            sol.c[0,i_a] = res_1.x

            # State 2
            sol.v[1,i_a] = -res_2.fun
            sol.c[1,i_a] = res_2.x

            # Update iteration parameters
        sol.it += 1
        sol.delta = max( max(abs(sol.v[0] - v_next[0])), max(abs(sol.v[1] - v_next[1]))) # check this, is this optimal  
    
    return sol
            
            
# Function that returns value of consumption choice conditional on the state
def value_of_choice_2d(x,a,a_next,v_next,par,state):
    
    # Unpack consumption (choice variable)
    c = x

    # Intialize expected continuation value
    Ev_next = 0.0
    
    # Compute value of choice conditional on being in state 1 (unemployment state)
    if state==1:
        # Loop over each possible state
        for i in [0,1]:
        
            # Next periods state for each income level
            a_plus = par.y[i] + (1+par.r)*(a - c)
        
            #Interpolate continuation given state a_plus
            v_plus = tools.interp_linear_1d_scalar(a_next,v_next,a_plus)
    
            # Append continuation value to calculate expected value
            Ev_next += par.P[0,i] * v_plus
    # Compute value of choice conditional on being in state 2 (employment state)
    else:
         # Loop over each possible state
        for i in [0,1]:
        
            # Next periods state for each income level
            a_plus = par.y[i] + (1+par.r)*(a - c)
        
            #Interpolate continuation given state a_plus
            v_plus = tools.interp_linear_1d_scalar(a_next,v_next,a_plus)
    
            # Append continuation value to calculate expected value
            Ev_next += par.P[1,i] * v_plus  
    # Value of choice
    v_guess = util.u(c,par) + par.beta * Ev_next

    return v_guess



    
# Copy below into solve.model() file to plot policy functions:
    
# Plot some stuff
#fig = plt.figure(figsize=(14,5))
#ax = fig.add_subplot(1,2,1)
#ax.plot(sol_vfi_2d.a, sol_vfi_2d.c[0,:], linestyle = ':', color = 'red', label = '$y_1$')
#ax.plot(sol_vfi_2d.a, sol_vfi_2d.c[1,:], linestyle = ':', color = 'blue', label = '$y_2$')
#ax.plot(sol_vfi_2d.a[:10], sol_vfi_2d.a[:10], linestyle = '--', color = '0.6') # Check with 45 degree line. Seems correct
#ax.set_xlabel(f"Assets, $a_t$")
#ax.set_ylabel(f"Consumption, $c^\star_t$")
#ax.set_title(f'Policy function')
#ax.set_xlim([-1,20])
#ax.legend(frameon=True)
#plt.show()