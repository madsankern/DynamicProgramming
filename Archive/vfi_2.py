# Solve the model using value function iteration

# To do:
# Rewrite value_of_choice to vectorize
# Vectorize inner loop in solve_VFI

import numpy as np
import tools
import scipy.optimize as optimize
import utility as util
import quantecon as qe # Package for Nelder-Mead algorithm
from numba import njit # Package for Nelder-Mead algorithm

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

    state1 = 1 # UNEMPLOYMENT STATE Used as boolean in "value_of_choice" - Defined here for readability
    state2 = 0 # EMPLOYMENT STATE Used as boolean in "value_of_choice" - Defined here for readability

    sol.it = 0 # Iteration counter
    sol.delta = 1000.0 # Difference between two iterations

    # Iterate value function until convergence or break if no convergence
    while (sol.delta >= par.tol_vfi and sol.it < par.max_iter):
        
        # Use last iteration as the continuation value. See slides if confused
        v_next = sol.v.copy()

         # Loop over asset grid
        for i_a,a in enumerate(par.grid_a):
            # FUNCTIONS BELOW CAN BE WRITTEN AS LOOP - for i=0,1 - AND BE STORED IN AN ARRAY/LIST WITH TWO ENTRIES - a la res[i]=optimize.minimize....
            # Minimize the minus the value function wrt consumption conditional on unemployment state
            obj_fun = lambda x : - value_of_choice_2d(x,a,par.grid_a,v_next[0,:],par,state1)
            res_1 = optimize.minimize_scalar(obj_fun, bounds=[0,a+1.0e-4], method='bounded')

            # Minimize the minus the value function wrt consumption conditional on employment state
            obj_fun = lambda x : - value_of_choice_2d(x,a,par.grid_a,v_next[1,:],par,state2)
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
    ###### VECTORIZE THIS
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
         ###### VECTORIZE THIS
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

# Attempt with 2 choice variables
#### Notes: Fordi vi skal løse problemet i to dele, så er vi nødt til at lave et if-statement, så det ene problem kun kører, når h_min>a. Hvis 
#### h_min>a og vi OGSÅ forsøger at løse problemet med h bounded mellem h_min og a, så kan vi jo ikke få nogen løsninger der overholder budget constraint, da valg 
#### af housing ikke er feasible. f.eks. hvis vi kun har a=2 og h_min=4
#### Idéen har derfor været kun at køre problemet, hvor der vælges consumption så længe h_min > a, og else køre begge problemer, og sammenligne i sidstnævnte scenarie
def solve_VFI_2dfull(par):

    # Initialize solution class
    class sol: pass
    shape = (np.size(par.y),1)
    sol.c = np.tile(par.grid_a.copy(), shape) # Initial guess is to consume everything for each state
    sol.h = np.zeros(np.shape(sol.c)) # Initial guess for housing is therefore zero for each state
    sol.v = util.u_with_housing(sol.c,sol.h,par) # Utility of consumption
    sol.a = par.grid_a.copy() # Copy the exogenous asset grid for consistency (with EGM algortihm)

    state1 = 1 # UNEMPLOYMENT STATE Used as boolean in "value_of_choice" - Defined here for readability
    state2 = 0 # EMPLOYMENT STATE Used as boolean in "value_of_choice" - Defined here for readability

    sol.it = 0 # Iteration counter
    sol.delta = 1000.0 # Difference between two iterations

    # Iterate value function until convergence or break if no convergence
    while (sol.delta >= par.tol_vfi and sol.it < par.max_iter):
        
        # Use last iteration as the continuation value. See slides if confused
        v_next = sol.v.copy()

         # Loop over asset grid
        for i_a,a in enumerate(par.grid_a):
                        
            if par.h_min > a:
                ### THIS IS ESSENTIALLY THE SAME PROBLEM AS BEFORE IN SCALAR-CASE. (Maybe we should use minimize_scalar, if it is faster)

               # Minimize the minus the value function wrt consumption conditional on unemployment state
                obj_fun = lambda x : - value_of_choice_2d(x,a,par.grid_a,v_next[0,:],par,state1)
                res_1 = optimize.minimize_scalar(obj_fun, bounds=[0,a+1.0e-4], method='bounded')

                # Minimize the minus the value function wrt consumption conditional on employment state
                obj_fun = lambda x : - value_of_choice_2d(x,a,par.grid_a,v_next[1,:],par,state2)
                res_2 = optimize.minimize_scalar(obj_fun, bounds=[0,a+1.0e-4], method='bounded')
            
                # Unpack solutions
                # State 1
                sol.v[0,i_a] = -res_1.fun
                sol.c[0,i_a] = res_1.x
                sol.h[0,i_a] = 0

                # State 2
                sol.v[1,i_a] = -res_2.fun
                sol.c[1,i_a] = res_2.x
                sol.h[1,i_a] = 0

            else:
                ### COMPUTE SOLUTION WITH h=0, AND h element of h_min and a, AND COMPARE SOLUTIONS
                
                # Minimize the minus the value function wrt consumption conditional on unemployment state
                obj_fun = lambda x : - value_of_choice_2d(x,a,par.grid_a,v_next[0,:],par,state1)
                res_1 = optimize.minimize_scalar(obj_fun, bounds=[0,a+1.0e-4], method='bounded')

                # Minimize the minus the value function wrt consumption conditional on employment state
                obj_fun = lambda x : - value_of_choice_2d(x,a,par.grid_a,v_next[1,:],par,state2)
                res_2 = optimize.minimize_scalar(obj_fun, bounds=[0,a+1.0e-4], method='bounded')
                
                # IMPLEMENT OPTIMIZER BELOW.
                ###
                ### DEBUG SOMEWHERE AROUND HERE ###
                ###
                arguments = [a,par] #### Se link for idé: https://stackoverflow.com/questions/54611746/scipy-minimize-how-to-pass-args-to-both-the-objective-and-the-constraint
                constraint = {'type': 'ineq', 'fun': feasibility_constraint, 'args': arguments}
                # Initial guess, x0
                #### MULTISTART GUESSES
                #####
                ## BEMÆRK: Den klager nogle over, at "Values in x were outside bounds during a" (lader til at den klager i første iteration) - kan ikke se hvorfor.
                #x0 = np.array([np.zeros(2)])+np.array([1.0e-3, par.h_min+1.0e-4])
                #x0 = np.array([np.zeros(2)])+np.array([1.0e-6, a-1.0e-4]) # It usually converges for Quasi utility with these initial values.
                x0 = np.array([np.zeros(2)])+np.array([1.0e-3, a-1.0e-4]) # It usually converges for Quasi utility with these initial values.
                #x0 = np.array([np.zeros(2)])+np.array([1.0e-3, (a+par.h_min)/2])
                #x0 = np.array([np.zeros(2)])+np.array([(a-par.h_min)/2, (a+par.h_min)/2])
                #x0 = np.array([np.zeros(2)])+np.array([a-par.h_min-1.0e-6, (a+par.h_min)/2])
                #x0 = np.array([np.zeros(2)])+np.array([a-par.h_min-1.0e-2, par.h_min+1.0e-4])
                
                # Minimize the minus the value function wrt consumption conditional on unemployment state
                #res_3 = optimize.minimize(objective_fun, x0, method='SLSQP', args = (a,par.grid_a,v_next[0,:],par,1), bounds = np.array([[1e-8,a+1.0e-4],[par.h_min,a+1.0e-4]]), constraints = constraint)
                res_3 = optimize.minimize(objective_fun, x0, method='SLSQP', args = (a,par.grid_a,v_next[0,:],par,state1), bounds = np.array([[1.0e-4,a],[par.h_min,a+1.0e-4]]), constraints = constraint)
                # Minimize the minus the value function wrt consumption conditional on employment state
                #res_4 = optimize.minimize(objective_fun, x0, method='SLSQP', args = (a,par.grid_a,v_next[0,:],par,0), bounds = np.array([[1e-8,a+1.0e-4],[par.h_min,a+1.0e-4]]), constraints = constraint)
                res_4 = optimize.minimize(objective_fun, x0, method='SLSQP', args = (a,par.grid_a,v_next[1,:],par,state2), bounds = np.array([[1.0e-4,a],[par.h_min,a+1.0e-4]]), constraints = constraint)
                # Unpack solutions
                # State 1
                # NOTEE: Can maybe store res_1, res_2, res_3, and res_4 in an array/list and run code below in a loop.
                sol.v[0,i_a] = max(-res_1.fun,-res_3.fun)
                if -res_1.fun>=-res_3.fun:
                    sol.c[0,i_a] = res_1.x
                    sol.h[0,i_a] = 0
                else:
                    sol.c[0,i_a] = res_3.x[0]
                    sol.h[0,i_a] = res_3.x[1]

                # State 2
                sol.v[1,i_a] = max(-res_2.fun,-res_4.fun)
                if -res_2.fun>=-res_4.fun:
                    sol.c[1,i_a] = res_2.x
                    sol.h[1,i_a] = 0
                else:
                    sol.c[1,i_a] = res_4.x[0]
                    sol.h[1,i_a] = res_4.x[1]
            # Update iteration parameters
        sol.it += 1
        sol.delta = max( max(abs(sol.v[0] - v_next[0])), max(abs(sol.v[1] - v_next[1]))) # check this, is this optimal  
        print(sol.delta)
    return sol
            
            
# Function that returns value of consumption choice conditional on the state
def value_of_choice_2dfull(c,h,a,a_next,v_next,par,state):
    
    # Unpack consumption (choice variable)
    #c = x[0]
    #h = x[1]

    # Intialize expected continuation value
    Ev_next = 0.0
    
    # Compute value of choice conditional on being in state 1 (unemployment state)
    if state==1:
        # Loop over each possible state
        for i in [0,1]:
        
            # Next periods state for each income level
            a_plus = par.y[i] + (1+par.r)*(a - par.hp*h - c)
        
            #Interpolate continuation given state a_plus
            v_plus = tools.interp_linear_1d_scalar(a_next,v_next,a_plus)
    
            # Append continuation value to calculate expected value
            Ev_next += par.P[0,i] * v_plus
    # Compute value of choice conditional on being in state 2 (employment state)
    else:
         # Loop over each possible state
        for i in [0,1]:
        
            # Next periods state for each income level
            a_plus = par.y[i] + (1+par.r)*(a - par.hp*h - c)
        
            #Interpolate continuation given state a_plus
            v_plus = tools.interp_linear_1d_scalar(a_next,v_next,a_plus)
    
            # Append continuation value to calculate expected value
            Ev_next += par.P[1,i] * v_plus  
    # Value of choice
    v_guess = util.u_with_housing(c,h,par) + par.beta * Ev_next

    return v_guess

def objective_fun(x,a,a_next,v_next,par,state):
    # Unpack consumption (choice variable)
    c = x[0]
    h = x[1]
    return -value_of_choice_2dfull(c,h,a,a_next,v_next,par,state)

def feasibility_constraint(x,a,par):
    # Ensure that consumption and housing jointly cannot exceed cash on hands
    c = x[0]
    h = x[1]
    return a-c-par.hp*h


###########################
####### NELDER-MEAD ####### - Virker ikke endnu!
###########################
@njit
def objective_fun_NELDER(x,a,a_next,v_next,par,state):
    # Unpack consumption (choice variable)
    c = x[0]
    h = x[1]

    penalty = 0
    if c+h > a:

        penalty = 10_000*(c+h-a)
        #c /= (c+h)/a
        #d /= (c+h)/a
    return value_of_choice_2dfull(c,h,a,a_next,v_next,par,state) - penalty # maximization

@njit(parallel=True)
def solve_VFI_2dfull_NELDER(par):
    #### BEMÆRK: FUNKTIONEN SKAL INITIALISERES ANDERLEDES, NÅR MAN BRUGER NJIT - MAN KAN IKKE BRUGE CLASS.
    # Initialize solution class
    class sol: pass
    shape = (np.size(par.y),1)
    sol.c = np.tile(par.grid_a.copy(), shape) # Initial guess is to consume everything for each state
    sol.h = np.zeros(np.shape(sol.c)) # Initial guess for housing is therefore zero for each state
    sol.v = util.u_with_housing(sol.c,sol.h,par) # Utility of consumption
    sol.a = par.grid_a.copy() # Copy the exogenous asset grid for consistency (with EGM algortihm)

    state1 = 1 # UNEMPLOYMENT STATE Used as boolean in "value_of_choice" - Defined here for readability
    state2 = 0 # EMPLOYMENT STATE Used as boolean in "value_of_choice" - Defined here for readability

    sol.it = 0 # Iteration counter
    sol.delta = 1000.0 # Difference between two iterations

    # Iterate value function until convergence or break if no convergence
        
    # Use last iteration as the continuation value. See slides if confused
    v_next = sol.v.copy()

        # Loop over asset grid
    for i_a,a in enumerate(par.grid_a):
                    
        if par.h_min > a:
            ### THIS IS ESSENTIALLY THE SAME PROBLEM AS BEFORE IN SCALAR-CASE. (Maybe we should use minimize_scalar, if it is faster)

            # Minimize the minus the value function wrt consumption conditional on unemployment state
            obj_fun = lambda x : - value_of_choice_2d(x,a,par.grid_a,v_next[0,:],par,state1)
            res_1 = optimize.minimize_scalar(obj_fun, bounds=[0,a+1.0e-4], method='bounded')

            # Minimize the minus the value function wrt consumption conditional on employment state
            obj_fun = lambda x : - value_of_choice_2d(x,a,par.grid_a,v_next[1,:],par,state2)
            res_2 = optimize.minimize_scalar(obj_fun, bounds=[0,a+1.0e-4], method='bounded')
        
            # Unpack solutions
            # State 1
            sol.v[0,i_a] = -res_1.fun
            sol.c[0,i_a] = res_1.x
            sol.h[0,i_a] = 0

            # State 2
            sol.v[1,i_a] = -res_2.fun
            sol.c[1,i_a] = res_2.x
            sol.h[1,i_a] = 0

        else:
            ### COMPUTE SOLUTION WITH h=0, AND h element of h_min and a, AND COMPARE SOLUTIONS
            
            # Minimize the minus the value function wrt consumption conditional on unemployment state
            obj_fun = lambda x : - value_of_choice_2d(x,a,par.grid_a,v_next[0,:],par,state1)
            res_1 = optimize.minimize_scalar(obj_fun, bounds=[0,a+1.0e-4], method='bounded')

            # Minimize the minus the value function wrt consumption conditional on employment state
            obj_fun = lambda x : - value_of_choice_2d(x,a,par.grid_a,v_next[1,:],par,state2)
            res_2 = optimize.minimize_scalar(obj_fun, bounds=[0,a+1.0e-4], method='bounded')
            
            # IMPLEMENT OPTIMIZER BELOW.
            ###
            ### DEBUG SOMEWHERE AROUND HERE ###
            ###
            arguments = [a,par] #### Se link for idé: https://stackoverflow.com/questions/54611746/scipy-minimize-how-to-pass-args-to-both-the-objective-and-the-constraint
            constraint = {'type': 'ineq', 'fun': feasibility_constraint, 'args': arguments}
            # Initial guess, x0
            #### MULTISTART GUESSES
            #x0 = np.array([np.zeros(2)])+np.array([1.0e-3, par.h_min+1.0e-4])
            x0 = np.array([np.zeros(2)])+np.array([1.0e-6, a-1.0e-4]) # It usually converges for Quasi utility with these initial values.
            #x0 = np.array([np.zeros(2)])+np.array([1.0e-3, (a+par.h_min)/2])
            #x0 = np.array([np.zeros(2)])+np.array([(a-par.h_min)/2, (a+par.h_min)/2])
            #x0 = np.array([np.zeros(2)])+np.array([a-par.h_min-1.0e-6, (a+par.h_min)/2])
            #x0 = np.array([np.zeros(2)])+np.array([a-par.h_min-1.0e-2, par.h_min+1.0e-4])
            
            #res_3 = optimize.minimize(objective_fun, x0, method='SLSQP', args = (a,par.grid_a,v_next[0,:],par,1), bounds = np.array([[1.0e-4,a],[par.h_min,a+1.0e-4]]), constraints = constraint)
            res_3 = qe.optimize.nelder_mead(objective_fun_NELDER,x0, 
                    bounds = np.array([[1.0e-4,a],[par.h_min,a+1.0e-4]]), 
                    args = (a,par.grid_a,v_next[0,:],par,1),
                    tol_x=par.tol_vfi, 
                    max_iter=1000)
            #res_4 = optimize.minimize(objective_fun, x0, method='SLSQP', args = (a,par.grid_a,v_next[0,:],par,0), bounds = np.array([[1.0e-4,a],[par.h_min,a+1.0e-4]]), constraints = constraint)
            res_4 = qe.optimize.nelder_mead(objective_fun_NELDER,x0, 
                    bounds = np.array([[1.0e-4,a],[par.h_min,a+1.0e-4]]), 
                    args = (a,par.grid_a,v_next[1,:],par,0),
                    tol_x=par.tol_vfi, 
                    max_iter=1000)
            sol.v[0,i_a] = max(-res_1.fun,-res_3.fun)
            if -res_1.fun>=-res_3.fun:
                sol.c[0,i_a] = res_1.x
                sol.h[0,i_a] = 0
            else:
                sol.c[0,i_a] = res_3.x[0]
                sol.h[0,i_a] = res_3.x[1]

            # State 2
            sol.v[1,i_a] = max(-res_2.fun,-res_4.fun)
            if -res_2.fun>=-res_4.fun:
                sol.c[1,i_a] = res_2.x
                sol.h[1,i_a] = 0
            else:
                sol.c[1,i_a] = res_4.x[0]
                sol.h[1,i_a] = res_4.x[1]
    return sol
