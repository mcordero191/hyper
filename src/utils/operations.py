import numpy as np
from scipy.ndimage import convolve1d

def calc_mean_winds(x,
                    y,
                    *u,
                    x_width=4*60*60,
                    x_sigma=0.5,
                    y_width=4,
                    y_sigma=0.5,
                    x_axis=0,
                    y_axis=1):
    '''
    Inputs:
        times         :    array-like. Time vector in seconds from 01/01/1950 (ntimes)
        *u = [u0, v0, w0]    :    array-like. Mean winds (ntimes, naltitudes)
        width         :    window width in seconds
    '''
    
    u = np.squeeze(u)
    
    if len(u.shape) != 3:
        raise ValueError("The array must have at least three dimensions with more than one element, %s" %u.shape)
    
    #Filter in time
    dx = x[1] - x[0]
    N = int(x_width/dx)
    
    
    if N < 2:
        return(u)
    
    M = (N-1)/2
    
    k = np.arange(N)
    w = np.exp(-0.5*((k-M)/M*x_sigma)**2)
    
    w = np.ones(N)
    w /= np.sum(w)
    
    u4h = np.empty_like(u)
    
    for i in range(u.shape[0]):
        u4h[i] = convolve1d(u[i], w, axis=x_axis, mode='nearest')
    
    # v4h = convolve1d(v0, w, axis=x_axis, mode='nearest')
    # w4h = convolve1d(w0, w, axis=x_axis, mode='nearest')
    
    #Filter in altitude
    dy = y[1] - y[0]
    N = int(y_width/dy)
    
    if N<2:
        return(u4h)
    
    M = (N-1)/2
    
    k = np.arange(N)
    
    w = np.exp(-0.5*((k-M)/M*y_sigma)**2)
    
    # w = np.ones(N)
    w /= np.sum(w)
    
    for i in range(u.shape[0]):
        u4h[i] = convolve1d(u4h[i], w, axis=y_axis, mode='nearest')
        
    # u4h = convolve1d(u4h, w, axis=y_axis, mode='nearest')
    # v4h = convolve1d(v4h, w, axis=y_axis, mode='nearest')
    # w4h = convolve1d(w4h, w, axis=y_axis, mode='nearest')
    
    return(u4h)

def corr_factor_2D(v1, v2, axis=0):
    """
    Inputs:
    
        v1, v2    :    3d inputs (nx, ny, nz)
        axis      :    correlation along axis=axis
    """
    
    N       = v1.shape[axis]
    corr    = np.zeros(N)
    
    for k in range(N):
        vi = np.take(v1, k, axis=axis)
        vj = np.take(v2, k, axis=axis)
        
        valid = np.isfinite(vi) & np.isfinite(vj)
        
        vi = vi[valid] - np.mean(vi[valid])
        vj = vj[valid] - np.mean(vj[valid])
        
        corr_ij = np.dot(vi, vj)/np.sqrt(np.sum(vi**2)*np.sum(vj**2))
        
        corr[k] = corr_ij
        
    return(corr)

def calc_rmse(fields, fields_est, vmaxs, type='mean'):
    
    '''
    Inputs:
        [u,v,w,....]    :    dimension [nalt, nlat, nlon]
    
    Outputs:
    
        rmses    :    dimension [nfields, nalt]
        
    '''
    rmses_lon = []
    rmses_lat = []
    rmses_alt = []
    
    for i in range(len(fields)):
        
        v = fields[i]
        ve = fields_est[i]
        
        vmax = vmaxs[i]
        
        #RMSE
        f = (v - ve)**2
        # f = np.where(np.abs(v)<0.1*vmax, np.nan, f)
        
        if type == 'mean':
            rmse1 = np.sqrt(np.nanmean(f, axis=(1,2)))
            rmse2 = np.sqrt(np.nanmean(f, axis=(0,2)))
            rmse3 = np.sqrt(np.nanmean(f, axis=(0,1)))
        
        elif type == 'std':
            rmse1 = np.sqrt(np.nanstd(f, axis=(1,2)))
            rmse2 = np.sqrt(np.nanstd(f, axis=(0,2)))
            rmse3 = np.sqrt(np.nanstd(f, axis=(0,1)))
        elif type[:4] == 'corr':
            rmse1 = corr_factor_2D(v, ve, axis=0)
            rmse2 = corr_factor_2D(v, ve, axis=1)
            rmse3 = corr_factor_2D(v, ve, axis=2)
        else:
            raise ValueError('select a proper type: mean, std, corr')
        
        rmses_alt.append(rmse1)
        rmses_lat.append(rmse2)
        rmses_lon.append(rmse3)
                
    
    return(rmses_lon, rmses_lat, rmses_alt)

def profile_cut(u, axis=(1,2)):
    
    # sigma = [0]*3
    #
    # for i in axis: sigma[i] = 1
    #
    # u = gaussian_filter(u, sigma=sigma)
    #
    # # nz, ny, nx = u.shape
    #
    index = u.shape[axis[1]]//2
    u0 = np.take(u, index, axis=axis[1])
    
    index = u.shape[axis[0]]//2
    u0 = np.take(u0, index, axis=axis[0])
    
    # u0 = np.nanmean(u, axis=axis)
    
    return u0