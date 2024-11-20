'''
Created on 30 Jun 2024

@author: radar
'''

import numpy as np

from georeference.geo_coordinates import lat2km, lon2km
from utils.plotting import plot_mean_winds

def distance_weight(df, t, h, dt, dh, window="gaussian", sigma=0.5):
    
    if window == "rectangular":
        
        mask = (np.abs(df["times"] - t) <= dt/2) & (np.abs(df["heights"] - h) <= dh/2)
        weights = np.ones_like(np.where(mask)[0])
        
    elif window == "gaussian":
        
        mask = (np.abs(df["times"] - t) <= 3*dt/2) & (np.abs(df["heights"] - h) <= 3*dh/2)
        
        dfm = df[mask]
        
        sigma_t = sigma*dt/2
        sigma_h = sigma*dh/2
        
        weights = np.exp( -0.5*( (dfm["times"] - t)/sigma_t )**2 - 0.5*( (dfm["heights"] - h)/sigma_h )**2 )
        weights = weights.values
    else:
    
        raise ValueError("Selected window not available (%s)" %window)
    
    return (mask, weights )

def mean_wind_grad(df,
                   times=None,
                   alts=None,
                   dt = 60*60, #1h
                   dh = 1, #km
                   outlier_sigma=3.5,
                   gradients=False,
                   min_number_of_measurements=10,
                   debug=False,
                   overlapping = 4.,
                   window="rectangular"):
    
    df = df.copy()
    
    lat0 = df["lats"].median()
    lon0 = df["lons"].median()
    
    if alts is None:
        hmin = int(df["heights"].min()*2)//2
        hmax = np.ceil(df["heights"].max()*2)//2
        
        alts = np.arange(hmin, hmax + dh, dh/overlapping)

    if times is None:
        tmin = int(df["times"].min()/(30*60))*30*60
        tmax = np.ceil(df["times"].max()/(30*60))*30*60
        
        times = np.arange(tmin, tmax + dt, dt/overlapping)

    ntimes  = len(times)
    nalts   = len(alts)
    
    # if no gradients
    n_par = 2
    
    # six unknowns: zonal and meridional mean wind, and their zonal and meridional gradients
    if gradients:
        n_par = 6

    u   = np.zeros( (n_par, ntimes, nalts) ) + np.nan
    ue  = np.zeros( (n_par, ntimes, nalts) ) + np.nan
    res = np.zeros( (ntimes, nalts) ) + np.nan

    # valid_mask = np.zeros_like(df["dops"].values)
    # df["quality"] = valid_mask
    
    df.loc[:, "quality"] = 0.0
    
    # for each time step
    for ti, t in enumerate(times):

        # for each height interval
        for hi, h in enumerate(alts):
            
            # mask_i = (np.abs(df["times"] - t) <= dt/2) & (np.abs(df["heights"] - h) <= dh/2)
            mask_i, weights = distance_weight(df, t, h, dt, dh, window=window)
            
            n_meas = np.count_nonzero(mask_i)
            
            if n_meas < min_number_of_measurements:
                continue
            
            if np.sum(weights) < min_number_of_measurements:
                continue
            
            df_i = df[mask_i]
            
            A = np.empty( (n_meas, n_par) )
        
            # Doppler measurements in units of rad/s: 2*pi*f  
            m = -2.0*np.pi*df_i["dops"].values
            
            # mean winds. Bragg vectors in units of (rad/m)
            A[:,0] = df_i["braggs_x"].values
            A[:,1] = df_i["braggs_y"].values

            if gradients:
                #
                # gradients (how much does velocity change per kilometer
                # in the zonal and meridional directions: u_x, u_y, v_x, v_y
                latkm = lat2km(df_i["lats"].values, lat0)
                lonkm = lon2km(df_i["lons"].values, df_i["lats"].values, lon0)
                
                # zon lon grad
                A[:,2] = df_i["braggs_x"].values*lonkm
                # mer lon grad
                A[:,3] = df_i["braggs_x"].values*latkm
                # zon lat grad
                A[:,4] = df_i["braggs_y"].values*lonkm
                # mer lat grad
                A[:,5] = df_i["braggs_y"].values*latkm
            
            ###################################
            #transform A and m using weights:
            ###################################
            W = np.diag(np.sqrt(weights))
            A = W @ A
            m = W @ m
            
            try:
                uhat = np.linalg.lstsq(A, m, rcond=None)[0]
            except:
                continue
            
            resid = m - np.dot(A, uhat)
            
            # robust estimator for standard deviation
            resid_std = 0.7*np.median(np.abs(resid))

            # outlier rejection, when making the estimate of mean wind
            gidx = np.where(np.abs(resid) < outlier_sigma*resid_std)[0]

            # one more iteration with outliers removed
            if len(gidx) < min_number_of_measurements:
                continue
            
            A2 = A[gidx,:]
            m2 = m[gidx]

            try:
                uhat2 = np.linalg.lstsq(A2, m2, rcond=None)[0]
            except:
                continue
            
            # estimate stdev
            try:
                stdev = np.sqrt(np.diag(np.linalg.inv(np.dot(np.transpose(A2),A2))))*resid_std
            except:
                continue

            resid = m - np.dot(A, uhat2)
            
            res[ti, hi] = np.var(resid)
            resid_std = 0.7*np.median(np.abs(resid))
            
            valid = np.where(np.abs(resid) < outlier_sigma*resid_std, 1.0, 0.0)
            
            df.loc[mask_i, "quality"] += valid
            
            for pi in range(n_par):
                u[pi,ti,hi] = uhat2[pi]
                ue[pi,ti,hi] = stdev[pi]

    if debug:
        plot_mean_winds(times, alts, u[0], u[1], u[0]*0,
                        vmins=[-100,-100, -10],
                        vmaxs=[ 100, 100,  10],
                        histogram=True
                        )
    
    df_filtered = df[df["quality"]>=1.0]
    
    df_winds = {}
    df_winds["u0"] = u[0]
    df_winds["v0"] = u[1]
    df_winds["w0"] = u[0]*0
    
    df_winds["u0_err"] = ue[0]
    df_winds["v0_err"] = ue[1]
    df_winds["w0_err"] = ue[0]*0
    
    df_winds["times"] = times
    df_winds["alts"] = alts
    
    return(df_winds, df_filtered)
    