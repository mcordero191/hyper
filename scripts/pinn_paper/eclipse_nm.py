
import os
import glob
import h5py

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

def read_meteor_file(filename, keys=['div', 'vor', 'wz']):
    
    with h5py.File(filename,'r') as fp:

        meta = fp['metadata']
        data = fp['data']
        
        times  = meta['times'][()]
        alts   = meta['alts'][()]
        lons   = meta['lons'][()]
        lats   = meta['lats'][()]
        
        u   = data[keys[0]][()]
        v   = data[keys[1]][()]
        w   = data[keys[2]][()]
    
    ds = xr.Dataset(
                    {
                        "u" : (["time","altitude"], u),
                        "v" : (["time","altitude"], v),
                        "w" : (["time","altitude"], w),
                        },
                        coords={
                            "time": times,
                            "altitude": alts,
                        },
        
                    )
    
    return(ds)

def read_meteor_files(path, pattern='residuals*'):
    
    subfolders = glob.glob1(path, 'final_plot*')
    
    df = None
    
    for folder in subfolders:
        
        junk = os.path.join(path, folder)
        subf = glob.glob1(junk, pattern)[0]
        
        fullpath = os.path.join(junk, subf)
        
        filename = sorted(glob.glob1(fullpath, 'wind_grads*TvsZ*.h5'))[0]
        
        filename = os.path.join(fullpath, filename)
        
        dfi = read_meteor_file(filename)
        
        if df is None:
            df = dfi
        else:
            df = xr.concat([df, dfi], dim='time')
            # df.append(dfi, ignore_index=True)
        
    return df
        
def histogram(df, bins=20):
    
    u = df["u"].values
    v = df["v"].values
    w = df["w"].values
    
    u_hist, _ = np.histogram(u, bins)
    v_hist, _ = np.histogram(u, bins)
    
    return(u_hist, v_hist)
    
def plot_hist(df, bins=30):
    
    u = df["u"].values
    v = df["v"].values
    # w = df["w"].values
    
    u = np.ravel(u)
    v = np.ravel(v)
    
    plt.figure(figsize=(9,3))
    
    plt.subplot(121)
    plt.hist(u, bins=bins)
    plt.grid()
    plt.title('Divergence')
    
    plt.subplot(122)
    plt.hist(v, bins=bins)
    plt.grid()
    plt.title('Vorticity')
    
    plt.show()
    
    
if __name__ == '__main__':
    
    path = "/Users/radar/Data/IAP/SIMONe/NewMexico/winds/nnRESPINN_11.11"
    
    pattern = "residuals*"
    pattern = "full*"
    
    df = read_meteor_files(path, pattern)
    
    # hist = histogram(df)
    
    plot_hist(df)