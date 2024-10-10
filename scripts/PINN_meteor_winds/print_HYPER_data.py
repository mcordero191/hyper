'''
Created on 30 Aug 2022

@author: mcordero
'''
import os, time, glob
import datetime
import numpy as np
import pandas as pd

import h5py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# from skimage.filters import gaussian
# from scipy.ndimage import gaussian_filter

from atmospheric_models.DNS import DNSReader
from georeference.geo_coordinates import lla2enu, lla2xyh

from pinn import hyper as pinn
from radar.smr.smr_file import SMRReader, filter_data

def save_ascii(times,
               x, y, z,
               u, v, w,
               u_dns, v_dns, w_dns,
               filename):
    
    df = {}
    
    df['times[s]'] = times
    df['x[m]'] = x
    df['y[m]'] = y
    df['z[m]'] = z
    df['u[m/s]'] = u
    df['v[m/s]'] = v
    df['w[m/s]'] = w
    df['u_true[m/s]'] = u_dns
    df['v_true[m/s]'] = v_dns
    df['w_true[m/s]'] = w_dns
    
    df = pd.DataFrame.from_dict(df)
    df.dropna(inplace=True)
    
    df.to_csv(filename, float_format='%g')
    
    # plt.subplot(311)
    plt.plot(u,u_dns,"rx", label="zonal")
    plt.plot(v,v_dns,"bh", label="meridional")
    plt.plot(w,w_dns,"go", label="vertical")
    
    plt.legend()
    
    plt.suptitle("Correlation")
    plt.xlabel("Estimates")
    plt.ylabel("Ground truth")
    
    plt.show()
    
def main(path_meteor_data,
         path_PINN,
         model_name,
         path_DNS=None,
         decS=1, decZ=1,
         zdecimation=1,
         vmins = [-10,-10,-5, None],
         vmaxs = [ 10, 10, 5, None],
         sevenfold=False
         ):
    
    figpath = os.path.join(path_PINN, '%s' %(os.path.splitext(model_name)[0]) )
    
    if not os.path.exists(figpath):
        os.mkdir(figpath)
        
    binpath = os.path.join(figpath, 'outs')
    
    if not os.path.exists(binpath):
        os.mkdir(binpath)
    
    
    
    meteor_sys = SMRReader(path_meteor_data)
    
    # meteor_sys.set_spatial_center(lat_center, lon_center, alt_center)
    #Get the updated center (in case lon, lat and alt_center were None
    lon_center, lat_center, alt_center = meteor_sys.get_spatial_center()
    
    meteor_sys.read_next_file(enu_coordinates=True)
    meteor_sys.filter(sevenfold=sevenfold)
    
    df_meteor = meteor_sys.df
    ini_time  = meteor_sys.get_initial_time()
    
    times = df_meteor['times'].values
    lats  = df_meteor['lats'].values
    lons  = df_meteor['lons'].values
    alts  = df_meteor['heights'].values
    
    x = df_meteor['x'].values
    y = df_meteor['y'].values
    z = df_meteor['z'].values
    
    
    
    
    filename = os.path.join(path_PINN, model_name)
    nn = pinn.restore(filename, log_index=log_index)
    
    lat_center = nn.lat_ref
    lon_center = nn.lon_ref
    alt_center = nn.alt_ref
    
    tmin, xmin, ymin, zmin = nn.lb
    tmax, xmax, ymax, zmax = nn.ub
        
    #Get winds at meteor locations
    outputs = nn.infer(times, lons, lats, alts, filter_output=True)
    
    u = outputs[:,0]
    v = outputs[:,1]
    w = outputs[:,2]
    
    u_dns = None
    v_dns = None
    w_dns = None
    
    if path_DNS is not None:
        
        u_dns = np.zeros_like(u) + np.nan
        v_dns = np.zeros_like(u) + np.nan
        w_dns = np.zeros_like(u) + np.nan
        
        #Read model data in LLA coordinates
        model_sys = DNSReader(path_DNS, decS=decS, decZ=decZ, dimZ=100)
        model_sys.set_meteor_initial_time(ini_time)
    
        x_dns = model_sys.x*1e3
        y_dns = model_sys.y*1e3
        z_dns = (model_sys.z + alt_center)*1e3
    
        while True:
            
            df = model_sys.read_next_block()
            if df is None: break
            
            if len(df.keys()) == 0:
                continue
            
            t_dns = int(df['time'])
            
            if t_dns < tmin:
                print('x', end=' ')
                continue
            
            if t_dns > tmax:
                print('x', end=' ')
                continue
            
            #Select grid points where there are meteors
            valid = np.where( np.abs(times-t_dns)<30 )[0]
            
            for ind, xi,yi,zi in zip(valid, x[valid], y[valid], z[valid]):
                
                try:
                    i = np.argwhere(np.isclose(x_dns,xi,atol=150))[0][0]
                    j = np.argwhere(np.isclose(y_dns,yi,atol=150))[0][0]
                    k = np.argwhere(np.isclose(z_dns,zi,atol=150))[0][0]
                except:
                    print(ind, end=", ")
                    continue
                
                u_dns[ind] = df['u'][k,j,i]
                v_dns[ind] = df['v'][k,j,i]
                w_dns[ind] = df['w'][k,j,i]
    
    save_ascii(times, x, y, z,
               u, v, w,
               u_dns, v_dns, w_dns,
               filename+".txt")
       
if __name__ == '__main__':
    
    
    path_meteor  = '/Users/radar/Data/IAP/SIMONe/Virtual/DNS_Simone2018/DNSx10_+12+53+91'
    path_DNS     = '/Users/radar/Data/IAP/Models/DNS/NonStratified'
    
    model_name = None
    subfolder = 'nnRESPINN_15.00'
    
    log_index       = None
    units           = 'm'
        
    junk = os.path.split( os.path.realpath(path_meteor) )[0]
    path_PINN = os.path.join(junk, "winds", subfolder)
        
    if model_name is None:
        models = glob.glob1(path_PINN, 'h*[!s].h5')
        models = sorted(models)
    else:
        models = [  model_name ]
    
    for model_name in models[:]:
        
        main(path_meteor,
             path_PINN,
             model_name,
             path_DNS=path_DNS,
             )