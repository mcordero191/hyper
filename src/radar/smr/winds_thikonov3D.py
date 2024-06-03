'''
Created on 4 Sep 2022

@author: mcordero
'''
# A module getting windfield cuts using Brian's wind field  inversion on MMARIA systems

import os
import datetime
import pytz
import h5py

import bisect
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scipy import optimize
from scipy import stats

def printerase(s):
    print(s)

def reshape_solution_3D(m, Nx, Ny, Nz, est_w=False, domain_idx=None):
    '''
    Reshape flattened/raveled solution vector into a 2D u,v,w
    
    m = solution vector
    Ny,Nx = shape of reconstruction grid
    est_w = bool, whether vertical wind is part of the solution
    domain_idx = 1D bool array of length 2*Nx*Ny or 3*Nx*Ny depending
                 on est_w. It indicates which of the pixels are actually
                 estimated during the inversion. If None, this indicates 
                 all pixels were estimated
    '''
    N = Nx*Ny*Nz
    shape = (Nx,Ny,Nz)
    
    if domain_idx is not None:
        if est_w:
            N3 = 3*N
        else:
            N3 = 2*N
        mfull = np.nan * np.zeros(N3)
        mfull[domain_idx] = m
    else:
        mfull = m
    
    U = np.reshape(mfull[:N],shape)
    V = np.reshape(mfull[N:2*N], shape)
    W = np.zeros(shape)
    if est_w:
        W = np.reshape(mfull[2*N:3*N], shape)
        
    return U,V,W

def gcv(A, Dc, d, Nx, Ny, bounds=[1e2, 1e4], tol=1e-3, verbose=False):
    '''
    Run Generalized Cross Validation to find the best regularization parameter. This is prohibitively expensive
    to run for large grids (i.e., Nx,Ny > 40).
    
    A - observation matrix
    Dc - regularization penalty matrix
    d - data
    bounds - good guess for initial bracket for optimize.minimize_scalar
    tol - tolerance for optimize.minimize_scalar
    verbose - If True, print full output from optimize.minimize_scalar
    mscale_thresh - [km, in reduced coords] threshold: how many km between meteors.
    
    Returns:
    lam0_opt: optimal regularization parameter for problem:
    
    min ||A*m - d||^2 + lam0 * ||Dc*m||^2
    
    NOTE: Be sure the problem formulation here matches the formulation in the "invert" function below.
    
    '''
    # Note scaling so that lam won't change if M changes or if Nx, Ny changes.
    M,N = np.shape(A)
    Bleft = A.T*A
    Bright = M/(N/3) * Dc.T*Dc

    def gcv_func(lam0):
        B = Bleft + lam0*Bright
        Binv = np.linalg.inv(B.toarray())
        Ahash = Binv @ A.T.toarray()
        m = Ahash @ d
        res = A*m - d
        den = (1-np.diag(A @ Ahash)).sum()**2
        g = res@res/den
        return g

    res = optimize.minimize_scalar(gcv_func, bounds, tol=tol)
    lam0_opt = res.x
    
    if verbose:
        print(res)
    
    return lam0_opt

def get_mdens(X, Y, Z, xm, ym, zm, xsc=1, ysc=1, zsc=1):
    '''
    Kernel density estimator to get meteor density on the grid, in reduced coordinates.
    
    X, Y, Z - (Nx,Nx,Nz) 2D reconstruction grid 
    xm, ym - (1D) coordinates of meteors
    xsc - float, how much to "stretch" x direction (i.e., how much less to penalize x curvatures)
    ysc - float, how much to "stretch" y direction (i.e., how much less to penalize y curvatures)
    
    Returns: 
    
    D - density of meteors in reduced coordinates
    
    '''
    Nx,Ny,Nz = X.shape
    M = len(xm)
    
    values = np.vstack((xsc*xm, ysc*ym, zsc*zm))
    kernel = stats.gaussian_kde(values)
    D = M*xsc*ysc*zsc*kernel(np.vstack([xsc*X.ravel(), ysc*Y.ravel(), zsc*Z.ravel()])).reshape( (Nx,Ny,Nz) )
    return D

def sort_into_grid(x, xd):
    '''
    Sort meteors into pre-defined 1D grid. Meteors outside the grid will be nan. A regular grid is assumed. 
    This function can be called more than once to do multiple dimensions.
    
    x = grid (len M)
    xd = meteor detections (len N)
    
    Returns
    i = indices (len N) of the grid, one for each meteor
    '''
    dx = x[1]-x[0]
    xe = np.linspace(x[0]-dx/2, x[-1]+dx/2, len(x)+1)
    i = []
    for xdi in xd:
        if (xdi < xe[0]) or (xdi > xe[-1]): # outside grid
            i.append(np.nan)
        else:
            ind = bisect.bisect(xe, xdi)-1
            i.append(ind)
        
        if i==len(x):
            print('xe',xe)
            print('xdi',xdi)
            raise Exception()
            
    return np.array(i)

def observation_matrix(x, y, z,
                       xm, ym, zm,
                       coeffs,
                       est_w = False):
    '''
    Construct observation matrix d = A*m, where d is the measured meteor doppler shifts and m is the 3*Nx*Ny
    vector of unknown u, v, w on the 2D grid. Note that x and y are given below but they are general and can
    represent x, y, altitude, whatever.
    
    Note that matrix indexing is done like A[y,x].
    
    x - (len Nx) Coordinates of grid in the "x" dimension
    y - (len Ny) Coordinates of grid in the "y" dimension
    xm - (len M) Coordinates of meteors in the "x" dimension
    ym - (len M) Coordinates of meteors in the "y" dimension
    coeffs - (shape Mx3) coefficients of forward model: d_i = coeffs[0]*u + coeffs[1]*v + coeffs[2]*w
             where u,v,w represent wind at the meteor detection position
             If saved in the standard format, this is df[['coeff_u','coeff_v','coeff_w']].values
    est_w - (bool) If False, the right third of the matrix will be trimmed, commensurate with the 
            assumption that the vertical wind is 0.
             
             
    Returns:
    A    - sparse forward matrix of shape (M, 3*Nx*Ny) or 2*Nx*Ny if est_w = False
    good - boolean index array, of which meteors were used (i.e., which were not outside the specified grid)
    '''
    Ny = len(y)
    Nx = len(x)
    Nz = len(z)
    
    # Sort meteors into grid
    i = sort_into_grid(x, xm)
    j = sort_into_grid(y, ym)
    k = sort_into_grid(z, zm)
    
    good = np.isfinite(i) & np.isfinite(j) & np.isfinite(k)

    # Grid indices of meteors
    mi = i[good].astype(int)
    mj = j[good].astype(int)
    mk = k[good].astype(int)
    
    midx = np.ravel_multi_index((mi,mj,mk), (Nx,Ny,Nz))
    M = len(midx)

    # Sparse construction. Easiest to construct each as Mx3 matrix then unravel.
    Ai = np.tile(np.arange(M), (3,1)).T # 3 matrix entries for every meteor
    Aj = np.tile(midx,         (3,1)).T
    Aj[:,1] += Nx*Ny*Nz
    Aj[:,2] += 2*Nx*Ny*Nz
    Av = coeffs[good,:]

    A = sp.coo_matrix((Av.ravel(),[Ai.ravel(),Aj.ravel()]), shape=(M, 3*Nx*Ny*Nz))
    A = A.tocsc()
    
    if not est_w:
        A = A[:,:2*Nx*Ny*Nz]
    
    return A, good

def curvature_matrix(dx, dy, dz, Nx, Ny, Nz, est_w = False):
    '''
    Generate curvature matrix, which is used in penalty term. It is assumed that 3 variables [u,v,w] are
    to be estimated on the 2D grid.
    
    Nx,Ny = shape of grid. This doesn't have to relate to actual x and y horizontal coordinates (i.e., you 
            can use altitude for x or y). X,Y are assumed to be indexed like meshgrid: X,Y = np.meshgrid(x,y).
    dx,dy = grid size in x and y, used to calculate derivatives to penalize. Note that if you want to penalize
            x derivatives more than y derivatives you can do so by making dx smaller.
    est_w - (bool) If False, the right third of the matrix will be trimmed, commensurate with the 
            assumption that the vertical wind is 0.
            
    returns Dc, a sparse matrix, Dc = sp.vstack((Dxx,Dyy,Dxy,Dyx))
    '''
    
    shape = (Nx,Ny,Nz)
    N = Nx*Ny*Nz
    
    # Construct Dxx
    Di = []
    Dj = []
    Dv = []
    row = 0 # counts which penalty term we're on
    for idx in range(N): # loop over all pixels and only keep those we can calculate derivative on
        i,j,k = np.unravel_index(idx, shape)
        ileft = i-1
        iright = i+1
        
        if (ileft  >= 0 and ileft  < Nx and \
            iright >= 0 and iright < Nx and \
            j >= 0 and j < Ny and \
            k >= 0 and k < Nz ):
            
            idxleft = np.ravel_multi_index((ileft,j,k), shape)
            idxright = np.ravel_multi_index((iright,j,k), shape)
            
            Di.extend([row,row,row])
            Dj.extend([idxleft,idx,idxright])
            Dv.extend( np.array([-1,2,-1])/dx**2 )
            
            Di.extend([1 + row, 1 + row, 1 + row])
            Dj.extend([N + idxleft, N + idx, N + idxright])
            Dv.extend( np.array([-1,2,-1])/dx**2 )
            
            Di.extend([2 + row, 2 + row, 2 + row])
            Dj.extend([2*N + idxleft, 2*N + idx, 2*N + idxright])
            Dv.extend( np.array([-1,2,-1])/dx**2 )
            row += 3
            
    Dxx = sp.coo_matrix((Dv,[Di,Dj]), shape=(max(Di)+1, 3*N))

    # Construct Dyy
    Di = []
    Dj = []
    Dv = []
    row = 0 # counts which penalty term we're on
    for idx in range(N): # loop over all pixels and only keep those we can calculate derivative on
        i,j,k = np.unravel_index(idx, shape)
        
        jleft = i-1
        jright = i+1
        
        if (jleft  >= 0 and jright  < Ny and \
            i >= 0 and i < Nx and \
            k >= 0 and k < Nz ):
            
            idxleft = np.ravel_multi_index((i,jleft,k), shape)
            idxright = np.ravel_multi_index((i,jright,k), shape)
            
            Di.extend([row,row,row])
            Dj.extend([idxleft,idx,idxright])
            Dv.extend((np.array([-1,2,-1])/dy**2))
            Di.extend([1 + row, 1 + row, 1 + row])
            Dj.extend([N + idxleft, N + idx, N + idxright])
            Dv.extend((np.array([-1,2,-1])/dy**2))
            Di.extend([2 + row, 2 + row, 2 + row])
            Dj.extend([2*N + idxleft, 2*N + idx, 2*N + idxright])
            Dv.extend((np.array([-1,2,-1])/dy**2))
            row += 3
            
    Dyy = sp.coo_matrix((Dv,[Di,Dj]), shape=(max(Di)+1, 3*N))

    # Construct Dzz
    Di = []
    Dj = []
    Dv = []
    row = 0 # counts which penalty term we're on
    for idx in range(N): # loop over all pixels and only keep those we can calculate derivative on
        i,j,k = np.unravel_index(idx, shape)
        
        kleft = k-1
        kright = k+1
        
        if (kleft  >= 0 and kright  < Nz and \
            i >= 0 and i < Nx and \
            j >= 0 and j < Ny ):
            
            idxleft = np.ravel_multi_index((i,j,kleft), shape)
            idxright = np.ravel_multi_index((i,j,kright), shape)
            
            Di.extend([row,row,row])
            Dj.extend([idxleft,idx,idxright])
            Dv.extend((np.array([-1,2,-1])/dz**2))
            Di.extend([1 + row, 1 + row, 1 + row])
            Dj.extend([N + idxleft, N + idx, N + idxright])
            Dv.extend((np.array([-1,2,-1])/dz**2))
            Di.extend([2 + row, 2 + row, 2 + row])
            Dj.extend([2*N + idxleft, 2*N + idx, 2*N + idxright])
            Dv.extend((np.array([-1,2,-1])/dz**2))
            row += 3
            
    Dzz = sp.coo_matrix((Dv,[Di,Dj]), shape=(max(Di)+1, 3*N))
    
    
    # Construct Dxy
    Di = []
    Dj = []
    Dv = []
    row = 0 # counts which penalty term we're on
    for idx in range(N): # loop over all pixels and only keep those we can calculate derivative on
        i, j, k = np.unravel_index(idx, shape)
        
        ileft = i-1
        iright = i+1
        jleft = j-1
        jright = j+1
        
        if (ileft >= 0 and iright < Nx and \
            jleft >= 0 and jright < Ny and \
            k >= 0 and k < Nz ):
            
            idxleft = np.ravel_multi_index((ileft,jleft,k), shape)
            idxright = np.ravel_multi_index((iright,jright,k), shape)
            
            Di.extend([row,row,row])
            Dj.extend([idxleft,idx,idxright])
            Dv.extend((np.array([-1,2,-1])/(dy*dx)))
            Di.extend([1 + row, 1 + row, 1 + row])
            Dj.extend([N + idxleft, N + idx, N + idxright])
            Dv.extend((np.array([-1,2,-1])/(dy*dx)))
            Di.extend([2 + row, 2 + row, 2 + row])
            Dj.extend([2*N + idxleft, 2*N + idx, 2*N + idxright])
            Dv.extend((np.array([-1,2,-1])/(dy*dx)))
            row += 3
            
    Dxy = sp.coo_matrix((Dv,[Di,Dj]), shape=(max(Di)+1, 3*N))

    # Construct Dxz
    Di = []
    Dj = []
    Dv = []
    row = 0 # counts which penalty term we're on
    for idx in range(N): # loop over all pixels and only keep those we can calculate derivative on
        i, j, k = np.unravel_index(idx, shape)
        
        ileft = i-1
        iright = i+1
        kleft = k-1
        kright = k+1
        
        if (ileft >= 0 and iright < Nx and \
            kleft >= 0 and kright < Nz and \
            j >= 0 and j < Ny ):
            
            idxleft = np.ravel_multi_index((ileft,j,kleft), shape)
            idxright = np.ravel_multi_index((iright,j,kright), shape)
            
            Di.extend([row,row,row])
            Dj.extend([idxleft,idx,idxright])
            Dv.extend((np.array([-1,2,-1])/(dz*dx)))
            Di.extend([1 + row, 1 + row, 1 + row])
            Dj.extend([N + idxleft, N + idx, N + idxright])
            Dv.extend((np.array([-1,2,-1])/(dz*dx)))
            Di.extend([2 + row, 2 + row, 2 + row])
            Dj.extend([2*N + idxleft, 2*N + idx, 2*N + idxright])
            Dv.extend((np.array([-1,2,-1])/(dz*dx)))
            row += 3
            
    Dxz = sp.coo_matrix((Dv,[Di,Dj]), shape=(max(Di)+1, 3*N))
    
    # Construct Dyz
    Di = []
    Dj = []
    Dv = []
    row = 0 # counts which penalty term we're on
    for idx in range(N): # loop over all pixels and only keep those we can calculate derivative on
        i, j, k = np.unravel_index(idx, shape)
        
        jleft = j-1
        jright = j+1
        kleft = k-1
        kright = k+1
        
        if (jleft >= 0 and jright < Ny and \
            kleft >= 0 and kright < Nz and \
            i >= 0 and i < Nx ):
            
            idxleft = np.ravel_multi_index((i,jleft,kleft), shape)
            idxright = np.ravel_multi_index((i,jright,kright), shape)
            
            Di.extend([row,row,row])
            Dj.extend([idxleft,idx,idxright])
            Dv.extend((np.array([-1,2,-1])/(dy*dz)))
            Di.extend([1 + row, 1 + row, 1 + row])
            Dj.extend([N + idxleft, N + idx, N + idxright])
            Dv.extend((np.array([-1,2,-1])/(dy*dz)))
            Di.extend([2 + row, 2 + row, 2 + row])
            Dj.extend([2*N + idxleft, 2*N + idx, 2*N + idxright])
            Dv.extend((np.array([-1,2,-1])/(dy*dz)))
            row += 3
            
    Dyz = sp.coo_matrix((Dv,[Di,Dj]), shape=(max(Di)+1, 3*N))
    
    # Construct Dyx
    Di = []
    Dj = []
    Dv = []
    row = 0 # counts which penalty term we're on
    for idx in range(N): # loop over all pixels and only keep those we can calculate derivative on
        i,j,k = np.unravel_index(idx, shape)
        
        ileft = i-1
        iright = i+1
        jleft = j+1
        jright = j-1
        
        if (ileft >= 0 and iright < Nx and \
            jleft < Ny and jright >=0 and \
            k >= 0 and k < Nz ):
            
            idxleft = np.ravel_multi_index((ileft,jleft,k),shape)
            idxright = np.ravel_multi_index((iright,jright,k),shape)
            
            Di.extend([row,row,row])
            Dj.extend([idxleft,idx,idxright])
            Dv.extend((np.array([-1,2,-1])/(dy*dx)))
            Di.extend([1 + row, 1 + row, 1 + row])
            Dj.extend([N + idxleft, N + idx, N + idxright])
            Dv.extend((np.array([-1,2,-1])/(dy*dx)))
            Di.extend([2 + row, 2 + row, 2 + row])
            Dj.extend([2*N + idxleft, 2*N + idx, 2*N + idxright])
            Dv.extend((np.array([-1,2,-1])/(dy*dx)))
            row += 3
    Dyx = sp.coo_matrix((Dv,[Di,Dj]), shape=(max(Di)+1, 3*N))
    
    # Construct Dzx
    Di = []
    Dj = []
    Dv = []
    row = 0 # counts which penalty term we're on
    for idx in range(N): # loop over all pixels and only keep those we can calculate derivative on
        i,j,k = np.unravel_index(idx, shape)
        
        ileft = i-1
        iright = i+1
        kleft = k+1
        kright = k-1
        
        if (ileft >= 0 and iright < Nx and \
            kleft < Nz and kright >=0 and \
            j >= 0 and j < Ny ):
            
            idxleft = np.ravel_multi_index((ileft,j,kleft),shape)
            idxright = np.ravel_multi_index((iright,j,kright),shape)
            
            Di.extend([row,row,row])
            Dj.extend([idxleft,idx,idxright])
            Dv.extend((np.array([-1,2,-1])/(dz*dx)))
            Di.extend([1 + row, 1 + row, 1 + row])
            Dj.extend([N + idxleft, N + idx, N + idxright])
            Dv.extend((np.array([-1,2,-1])/(dz*dx)))
            Di.extend([2 + row, 2 + row, 2 + row])
            Dj.extend([2*N + idxleft, 2*N + idx, 2*N + idxright])
            Dv.extend((np.array([-1,2,-1])/(dz*dx)))
            row += 3
    Dzx = sp.coo_matrix((Dv,[Di,Dj]), shape=(max(Di)+1, 3*N))
    
    # Construct Dzy
    Di = []
    Dj = []
    Dv = []
    row = 0 # counts which penalty term we're on
    for idx in range(N): # loop over all pixels and only keep those we can calculate derivative on
        i,j,k = np.unravel_index(idx, shape)
        
        jleft = j-1
        jright = j+1
        kleft = k+1
        kright = k-1
        
        if (jleft >= 0 and jright < Ny and \
            kleft < Nz and kright >=0 and \
            i >= 0 and i < Nx ):
            
            idxleft = np.ravel_multi_index((i,jleft,kleft),shape)
            idxright = np.ravel_multi_index((i,jright,kright),shape)
            
            Di.extend([row,row,row])
            Dj.extend([idxleft,idx,idxright])
            Dv.extend((np.array([-1,2,-1])/(dy*dz)))
            Di.extend([1 + row, 1 + row, 1 + row])
            Dj.extend([N + idxleft, N + idx, N + idxright])
            Dv.extend((np.array([-1,2,-1])/(dy*dz)))
            Di.extend([2 + row, 2 + row, 2 + row])
            Dj.extend([2*N + idxleft, 2*N + idx, 2*N + idxright])
            Dv.extend((np.array([-1,2,-1])/(dy*dz)))
            row += 3
    Dzy = sp.coo_matrix((Dv,[Di,Dj]), shape=(max(Di)+1, 3*N))
    
    Dc = sp.vstack((Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, Dyx, Dzx, Dzy))
    
    Dc = Dc.tocsc()
    if not est_w:
        Dc = Dc[:,:2*N]
    
    return Dc

def create_inversion_matrices(df,
                              xcoord, ycoord, zcoord,
                              x, y, z,
                              t, dt,
                              xsc=1, ysc=1, zsc=17.3,
                              time_taper=False, z_taper=False,
                              est_w=False, use_sigma=True, sig_w=0.,
                              mask=None):
    '''    
    Construct observation matrix, data vector, and penalty matrix. Note that weights are incorporated
    in A and d.
    
    df - DataFrame containing meteors to invert. 
    xcoord - string in ['x', 'y', 'z']. Which coordinate to consider "x" for the inversion (resolved)
    ycoord - string in ['x', 'y', 'z']. Which coordinate to consider "y" for the inversion (resolved)
    zcoord - string in ['x', 'y', 'z']. Which coordinate to consider "z" for the inversion (unresolved)
    x - 1D array of x reconstruction grid
    y - 1D array of y reconstruction grid
    z - 1D array of z reconstruction grid
    
    t - [pandas datetime] center time of reconstruction
    dt - [hr] interval for reconstruction (t-dt/2, t+dt/2)
    xsc - float, how much to "stretch" x direction (i.e., how much less to penalize x curvatures)
    ysc - float, how much to "stretch" y direction (i.e., how much less to penalize x curvatures)
    time_taper - [bool] whether to weight meteors by their proximity to center time
    z_taper    - [bool] whether to weight meteors by their proximity to center z
    est_w - [bool] If True, try to estimate the vertical wind. If False, assume it is zero.
    use_sigma - [bool] If True, use the Doppler uncertainties to set data weights.
    sig_w   - [m/s] If est_w=False, and sig_w > 0, then treat the vertical wind as a random variable with
                    the specified sigma. This solves the problem of fitting to mostly-vertical Bragg vectors
                    with small uncertainties when it is assumed w=0.
    mask    - [Nx,Ny bool array]. If None, do nothing. If specified, then use the given mask to effectively
              ignore grid points *before* the inversion. (True = include the point)
    
    Returns:
    A - observation matrix (d = A*m)
    d - data
    Dc - curvature matrix for penalty term
    idx - boolean index array, of which meteors were used (i.e., which were not outside the specified grid)
    domain_idx - 1D bool array of length 2*Nx*Ny or 3*Nx*Ny depending on est_w. It indicates which of the 
                pixels are actually estimated during the inversion. If None, this indicates all pixels were estimated    
    '''
    Nx = len(x)
    Ny = len(y)
    Nz = len(z)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    
    # Construct observation matrix
    A, good = observation_matrix(x, y, z,
                                 df[xcoord], df[ycoord], df[zcoord],
                                 df[['braggs_x','braggs_y','braggs_z']].values,
                                 est_w=est_w)
    # Construct data vector
    d = -2*np.pi*df['dops'][good].values 
    
    # Construct penalty matrix
    Dc = curvature_matrix(xsc*dx, ysc*dy, zsc*dz, Nx, Ny, Nz, est_w=est_w) # w estimation not implemented yet
    
    if mask is not None:
        
        # A 1D boolean array specifying which elements of the model vector m will actually be estimated.
        if est_w:
            domain_idx = np.concatenate([mask.flatten()]*3)
        else:
            domain_idx = np.concatenate([mask.flatten()]*2)
            
        # Sub-sample A and d
        Asub = A[:,domain_idx] # Exclude poorly observed grid points from forward model
        data_idx = np.array(Asub.sum(axis=1)).squeeze() != 0 # which meteors are actually in the new grid
        dsub = d[data_idx]      # Exclude data from outside grid
        Asub = Asub[data_idx,:] # Exclude the corresponding forward model equations
        # Sub-sample Dc, curvature constraint
        reach_outside = np.array(abs(Dc[:,~domain_idx]).sum(axis=1) > 0).squeeze() # which constraint eqs (rows) involve grid points outside domain
        Dcsub = Dc[~reach_outside, :]
        Dcsub = Dcsub[:,domain_idx] # not sure why this needs to be two steps
        
        d = dsub
        A = Asub
        Dc = Dcsub
        i = np.where(good)[0][~data_idx] # meteors indices that were used before but now aren't
        good[i] = False
        
    else:
        _,M = np.shape(A)
        domain_idx = np.ones(M, dtype=bool)
    
    
    M = len(d) # number of measurements
    # Weight matrix
    wt = np.ones(M)
    
    # Use uncertainties given in file to set weights
    if use_sigma:
        sig = df['dop_errs'][good].values
        if est_w is False:
            cw = df['braggs_z'][good].values
            sig = np.sqrt(sig**2 + cw**2*sig_w**2)
        wt = wt * 1./sig # weight matrix is Sig^(-1/2)
        
    # Add weight based on proximity to center time
    if time_taper:
        dtm = ((df['t'][good] - t).values/3600).astype(float)/1e9 # ugh. time offset in hrs
        wt = wt * 2*abs(dtm)/(dt/2) # factor of 2 to keep mean value the same
    
    # Add weight based on proximity to center zcoord.
    if z_taper:
        dtm = np.exp(-(((df[zcoord][good] - z).values)**2/(2.*(dz)**2)).astype(float))
        wt = wt * dtm/(np.sqrt(2*np.pi)*(dz)) # factor to keep area equal to 1.
        
    WT = sp.diags(wt)
    A = WT*A
    d = WT*d
    
    return A, d, Dc, good, domain_idx

def invert(dfraw,
           thr=6.0, dt=0.5,
           xcoord='x', ycoord='y', zcoord='z',
           xmin=-200, xmax=200,
           ymin=-200, ymax=200,
           zmin=80, zmax=105,
           Nx=20, Ny=20, Nz=12,
           xsc=1.0, ysc=1, zsc=17.2,
           run_gcv=False, log10_regparam=3.4,
           time_taper=False, z_taper=False, est_w=False, use_sigma=False,
           data='all', mscale_thresh=np.inf, mask_before=True, Mthresh=50,
           sig_w=0.,
           ):
    '''
    High-level function to run the 2D wind field inversion
    
    Coordinates:
    - xcoord, ycoord will be resolved by inversion. Choose 'x' longitude , 'y' latitude, or 'z' altitude. 
      A slice is taken along zcoord (whatever is left), according to z0 and dz parameters.
    - xmin, xmax, ymin, ymax - define region being reconstructed [km]
    - z0 - where to take slice along zcoord [km]
    - dz - how wide the slice is [km]
    Reconstruction parameters:
    - Nx, Ny: size of reconstruction grid [pixels]
    - xsc, ysc: How much to "stretch" each coordinate before performing inversion. Recommend 1 for horizontal and 17.2 for vertical.
    - run_gcv: If true, use Generalized Cross Validation (on a reduced resolution version) to determine the regularization
               parameter, instead of specifying it a priori (see next input below).
    - log10_regparam: log10 of the regularization parameter to use for Tikhonov regularization. Note this will be
                      be ignored if run_gcv==True
    - time_taper: [bool]. Whether to weight meteors by their proximity to the current time
    - z_taper: [bool]. Whether to weight meteors by their proximity to the z0 (Gaussian tapering)
    - est_w: [bool]. Whether to estimate w (False = assume 0)
    - use_sigma - [bool] If True, use the Doppler uncertainties to set data weights.
    - data: ['even','odd','all'] - a testing/debugging input. Use only even meteors, odd meteors, or all meteors. 
    - mscale_thresh - [km, in reduced coords] threshold: how many km between meteors. At lower density,
                    the result will be masked. If np.inf (default), no masking will be done.
    - mask_before - bool. This parameter matters only if mscale_thresh is not np.inf. 
                          If True,  mask the domain before doing the inversion.
                          If False, mask the domain after doing the inversion.
    - Mthresh - [# meteors] if the number of meteors to be analyzed is smaller than this, don't even try to invert.
    - sig_w   - [m/s] If est_w=False, and sig_w > 0, then treat the vertical wind as a random variable with
                      the specified sigma. This solves the problem of fitting to mostly-vertical Bragg vectors
                      with small uncertainties when it is assumed w=0.
    
    Returns:
    U,V,W - wind on grid [m/s]
    D - density of meteors on grid, in reduced coordinates
    x,y - 1D coordinates of grid
    df - DataFrame containing input meteor data, with residuals. Only contains meteors which were actually used.
    '''
    
    assert data in ['even','odd','all'], "Input 'data' must be in ['even', 'odd', 'all']. Currently it is '%s'"% data
        
    # Slice to reconstruct
    #tbase = pd.to_datetime(dfraw['t'][0].date())
    tbase = pd.to_datetime(dfraw['t'][dfraw['t'].keys()[0]].date())
    t = tbase + pd.to_timedelta(thr, unit='h') # Center time for reconstruction
    tmin = pd.to_datetime(t) - pd.to_timedelta(dt/2, unit='h')
    tmax = pd.to_datetime(t) + pd.to_timedelta(dt/2, unit='h')
    
    df = dfraw[(dfraw['t'] >= tmin) & (dfraw['t'] < tmax) ]

    x = np.linspace(xmin,xmax,Nx)
    y = np.linspace(ymin,ymax,Ny)
    z = np.linspace(zmin,zmax,Nz)
    X,Y,Z = np.meshgrid(x,y,z)
    
    # try:

    if df.shape[0] < Mthresh:
        # raise Exception('Too few meteors (%i). Need at least Mthresh=%i' % (df.shape[0], Mthresh))
        return(None, None, None, None, None, None, None, None, None)
    if data == 'even':
        df = df[df.index % 2 == 0]
    elif data == 'odd':
        df = df[df.index % 2 == 1]
    
    # Density
    D = get_mdens(X, Y, Z, df[xcoord], df[ycoord], df[zcoord], xsc, ysc, zsc)
    S = 1./np.sqrt(D) # distance between meteors on average [km, reduced coordinates]
    mask = np.ones((Nx,Ny,Nz), dtype=bool) # Use all the pixels by default
    if mask_before: # Trim domain before the inversion
        mask = S < mscale_thresh
    
    ############## INVERSION ###############
    # Optional low-res version to run GCV to get reg param:
    # Set to simple case so as not to affect GCV weighting:
    # (I'm not 100% sure why this is necessary)
    # - time_taper = False
    # - z_taper = Falsex

    # Real inversion
    A, d, Dc, idx, domain_idx = create_inversion_matrices(df,
                                                          xcoord, ycoord, zcoord,
                                                          x, y, z,
                                                          t, dt,
                                                          xsc, ysc, zsc,
                                                          time_taper=time_taper,
                                                          z_taper=z_taper,
                                                          est_w=est_w,
                                                          use_sigma=use_sigma,
                                                          sig_w=sig_w,
                                                          mask=mask)
    dfm = df.loc[idx]
    
    # Set up problem
    M,N = np.shape(A)
    Bleft = A.T @ A
    Bright = M/(N/3) * Dc.T @ Dc # Note scaling so that lam won't change if M changes, if Nx, Ny changes, or if masking changes
    rhs = A.T @ d
    
    # Solve
    lam0 = 10**log10_regparam
    B = Bleft + lam0*Bright
    m = sp.linalg.spsolve(B,rhs)
    U,V,W = reshape_solution_3D(m, Nx, Ny, Nz, est_w=est_w, domain_idx=domain_idx)
    resid = d - A*m
    dfm = dfm.assign(resid=resid)

    if not mask_before:
        mask = S < mscale_thresh
        U[~mask] = np.nan
        V[~mask] = np.nan
        W[~mask] = np.nan
    
    # except Exception as e:
    #     print(e)
    #
    #     U = np.zeros((Nx,Ny,Nz))+np.nan
    #     V = np.zeros((Nx,Ny,Nz))+np.nan
    #     W = np.zeros((Nx,Ny,Nz))+np.nan
    #     D = np.zeros((Nx,Ny,Nz))
    #     log10_regparam= 0.0
    #     dfm = None
    
    return U, V, W, D, x, y, z, dfm, log10_regparam


def run_xy(dfraw,
           fn,
           outlier_thresh=1.0,
           run_gcv=False,
           time_taper=False,
           z_taper = False,
           use_sigma=False,
           xsc=1.0, ysc=1.0, zsc=17.3,
           log10_regparam=3.2,
           est_w=False,
           mscale_thresh=30.,
           mask_before = False,
           data='all',
           tstep=1/12.,
           dt=1,
           Nx=15,
           Ny=15,
           Nz=20,
           xmin=-150, xmax=150,
           ymin=-150, ymax=150,
           zmin=70,zmax=110,
           tmin=0, tmax=3,
           verbose=False,
           sig_w=0.0,
           Mthresh=4.
           ):
    '''
        Unpolished function to run an x-y domain inversion, and loop over z and time to create 4D wind field. Also save an h5 file.
        
        Advanced options:
        
        outlier_thresh: if < 1.0, then run the inversion twice. After the first, remove all meteors that have chi^2
        contributions above the p^th percentile, where p = 100*outlier_thresh. (default 1.0 = run only once, use all meteors)
        run_gcv: if True, ignore the "log10_regparam" input and instead use Generalized Cross Validation to find the "optimal"
        parameter, on a lower resolution version for speed (20x20).
        
        '''
    
    # Optional pre-run to find outliers.
    # This pre-run has the following simplifications:
    #  - time_taper = False  (so as not to skew the residuals by the weighting)
    #  - z_taper = False  (so as not to skew the residuals by Gaussian weighting)
    #  - run_gcv = False   (for speed. reg param will be set to whatever is inputted).
    #  - outlier_thresh = 1.0 (obviously)
    if outlier_thresh < 1.0:
        fn_pre = fn[:-3] + '_preclean.h5'
        run_xy(dfraw, fn_pre, outlier_thresh=1.0,
               time_taper = False, z_taper = False,
               use_sigma=use_sigma, run_gcv=False,
               log10_regparam=log10_regparam,
               xsc=xsc, ysc=ysc, est_w=est_w, 
               mscale_thresh=mscale_thresh,
               mask_before=mask_before,
               tstep=tstep,
               dt=dt,
               Nx=Nx, Ny=Ny,Nz=Nz,
               xmin=xmin, xmax=xmax,
               ymin=ymin, ymax=ymax,
               zmin=zmin, zmax=zmax,
               tmin=tmin, tmax=tmax,
               verbose=False,
               sig_w=sig_w,
               Mthresh=Mthresh)
            
        with h5py.File(fn_pre, 'r') as f:
            r = f['meteor_resid'][...]
            i = f['meteor_index'][...]
            rall = np.concatenate(r.flatten())
            iall = np.concatenate(i.flatten())
            df = pd.DataFrame(index=iall, data={'chi2':rall**2}) # Contains duplicate meteors
            dfc = df.groupby(df.index).mean() # Combine (average) duplicates

            p = dfc['chi2'].quantile(q=outlier_thresh)
            bad_meteors = dfc[dfc['chi2'] > p].index
            dfraw = dfraw.drop(bad_meteors)
            if verbose:
                print('Removed %i meteors' % (bad_meteors.size))

    ############ RUN ##############
    xcoord = 'x'
    ycoord = 'y'
    # The following notation is a bit confusing since it has to translate between inversion x,y,z and real x,y,z (i.e., x,z,y)

    thr = np.arange(tmin,tmax,tstep)
    Nt = len(thr)

    U = np.nan * np.zeros((Nt, Nx, Ny, Nz))
    V = np.nan * np.zeros((Nt, Nx, Ny, Nz))
    W = np.nan * np.zeros((Nt, Nx, Ny, Nz))
    D = np.nan * np.zeros((Nt, Nx, Ny, Nz))

    met_resid = []
    met_idx = []
    loglams = []

    for i in range(Nt):
        
        print('#%d/%d' %(i+1, Nt) )
        
        Ui,Vi,Wi,Di,x,y,z,dfm,loglam = invert(dfraw,
                                            run_gcv=run_gcv,
                                            thr=thr[i],
                                            dt=dt,
                                            xcoord=xcoord, ycoord=ycoord,
                                            Nx=Nx, Ny=Ny, Nz=Nz,
                                            xsc=xsc, ysc=ysc, zsc=zsc,
                                            log10_regparam=log10_regparam,
                                            time_taper=time_taper,
                                            z_taper = z_taper,
                                            use_sigma=use_sigma,
                                            data=data,
                                            est_w=est_w,
                                            sig_w=sig_w,
                                            mscale_thresh=mscale_thresh,
                                            mask_before=mask_before,
                                            xmin=xmin, xmax=xmax,
                                            ymin=ymin, ymax=ymax,
                                            zmin=zmin, zmax=zmax,
                                            Mthresh=Mthresh)
            
        if dfm is None:
            met_resid.append([np.nan])
            met_idx.append([-1])
            loglams.append(loglam)
            print('x', end='')
            continue
        
        print('.', end='')
        
        U[i,:,:,:] = Ui
        V[i,:,:,:] = Vi
        W[i,:,:,:] = Wi
        D[i,:,:,:] = Di
        met_resid.append(dfm['resid'].values)
        met_idx.append(dfm.index.values)
        loglams.append(loglam)


    if (Nt>1):
        met_idx = np.reshape(met_idx, (Nt))
        met_resid = np.reshape(met_resid, (Nt))
    else:
        met_idx = np.reshape(met_idx, (1,1,len(met_idx[0])))
        met_resid = np.reshape(met_resid, (1,1,len(met_resid[0])))

    ############# SAVE ###############
    # http://stackoverflow.com/questions/20928136/input-and-output-numpy-arrays-to-h5py
    with h5py.File(fn, 'w') as hf:
        
        # Wind
        hf.create_dataset('wind_u', data=U).attrs['units'] = 'm/s'
        hf.create_dataset('wind_v', data=V).attrs['units'] = 'm/s'
        hf.create_dataset('wind_w', data=W).attrs['units'] = 'm/s'
        
        # Meteor and inversion info
        # hf.create_dataset('log10_regparam', data=np.array(loglams).reshape((Nt)))
        hf.create_dataset('meteor_density', data=D).attrs['units'] = 'km^-2'
        # Variable data length: http://docs.h5py.org/en/stable/special.html#arbitrary-vlen-data
        # hf.create_dataset('meteor_resid', data=met_resid).attrs['units'] = 'Hz'

        dset = hf.create_dataset('meteor_index', shape=(Nt), dtype=h5py.special_dtype(vlen=np.dtype('int32')))
        for i in range(Nt):
            dset[i] = met_idx[i]
        dset = hf.create_dataset('meteor_resid', shape=(Nt), dtype=h5py.special_dtype(vlen=np.dtype('f')))
        for i in range(Nt):
            dset[i] = met_resid[i]

        # Coordinates
        #tbase = pd.to_datetime(dfraw['t'][0].date())
        tbase = pd.to_datetime(dfraw['t'][dfraw['t'].keys()[0]].date())
        t = tbase + pd.to_timedelta(thr, unit='h')
        ts = np.array([ti.timestamp() for ti in t])
        hf.create_dataset('t', data=ts).attrs['units'] = 's'
        hf.create_dataset('x', data=x).attrs['units'] = 'km'
        hf.create_dataset('y', data=y).attrs['units'] = 'km'
        hf.create_dataset('z', data=z).attrs['units'] = 'km'
        # hf.create_dataset('lon0', data=input_misc['lon0']).attrs['units'] = 'deg'
        # hf.create_dataset('lat0', data=input_misc['lat0']).attrs['units'] = 'deg'

        # Inputs
        hf.create_dataset('in_dt', data=dt).attrs['units'] = 'hr'
        hf.create_dataset('in_estw', data=est_w)
        hf.create_dataset('in_timetaper', data=time_taper)
        hf.create_dataset('in_ztaper', data=z_taper)
        hf.create_dataset('in_usesigma', data=use_sigma)
        hf.create_dataset('in_sigw', data=sig_w)
        hf.create_dataset('in_xsc', data=xsc)
        hf.create_dataset('in_ysc', data=ysc)
        hf.create_dataset('in_log10regparam', data=log10_regparam)
        hf.create_dataset('in_mscalethresh', data=mscale_thresh)
        hf.create_dataset('in_maskbefore', data=mask_before)
        # hf.create_dataset('in_filename', data=input_misc['fn_in'])
        # hf.create_dataset('in_meanremoved', data=input_misc['mean_removed'])
        # hf.create_dataset('in_meanwindow_t', data=input_misc['t_window']).attrs['units'] = 'hr'
        # hf.create_dataset('in_meanwindow_z', data=input_misc['z_window']).attrs['units'] = 'km'
        hf.create_dataset('in_outlierthresh', data=outlier_thresh)
        hf.create_dataset('in_rungcv', data=run_gcv)
        
        # Auxiliary info
        hf.create_dataset('t_file_creation', data=datetime.datetime.now().astimezone(pytz.utc).strftime('%Y-%m-%d %H:%M UT'))
    
    return fn

if __name__ == '__main__':
    
    from radar.specular_meteor_radars.SMR import SMRReader
    
    path = '/Users/mcordero/Data/IAP/SIMONe/Virtual2018'
    rpath = '/Users/mcordero/Data/IAP/SIMONe/Winds2018_3D'
    
    if not os.path.exists(rpath): os.mkdir(rpath)
        
    #Read meteor data in LLA coordinates
    meteor_obj = SMRReader(path)
    
    while True:
        
        info = meteor_obj.read_next_block()
        if info != 1: break
        
        df = meteor_obj.df
        filename = os.path.split( meteor_obj.filename )[1]
        filename = os.path.join(rpath, 'wind_'+filename)
        run_xy(df, filename, est_w=True)
        