import numpy as np
from numba import njit, int64, double
import math

# interpolation functions:
@njit(int64(int64,int64,double[:],double))
def binary_search(imin,Nx,x,xi):
        
    # a. checks
    if xi <= x[0]:
        return 0
    elif xi >= x[Nx-2]:
        return Nx-2
    
    # b. binary search
    half = Nx//2
    while half:
        imid = imin + half
        if x[imid] <= xi:
            imin = imid
        Nx -= half
        half = Nx//2
        
    return imin

@njit(double(double[:],double[:],double))
def interp_linear_1d_scalar(grid,value,xi):
    """ raw 1D interpolation """

    # a. search
    ix = binary_search(0,grid.size,grid,xi)
    
    # b. relative positive
    rel_x = (xi - grid[ix])/(grid[ix+1]-grid[ix])
    
    # c. interpolate
    return value[ix] + rel_x * (value[ix+1]-value[ix])

@njit
def interp_linear_1d(grid,value,xi):

    yi = np.empty(xi.size)

    for ixi in range(xi.size):

        # c. interpolate
        yi[ixi] = interp_linear_1d_scalar(grid,value,xi[ixi])
    
    return yi

def gauss_hermite(n):

    # a. calculations
    i = np.arange(1,n)
    a = np.sqrt(i/2)
    CM = np.diag(a,1) + np.diag(a,-1)
    L,V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:,I].T

    # b. nodes and weights
    x = L[I]
    w = np.sqrt(math.pi)*V[:,0]**2

    return x,w