'''
Created on 30 Aug 2022

@author: mcordero
'''
import os, glob
import time, datetime
import h5py

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pinn import hyper
from torch._inductor.ir import NoneLayout
   
def read_upleg(upleg = 1):
    
    if upleg:
        
        filename = '/Users/mcordero/Data/IAP/SIMONe/Norway/VorTex/UPLEG34_sigma0_1Bin_size_1km.txt'
        lat = 69.45
        lon = 15.75
        ref_label = 'UPLEG'
        dt = datetime.datetime(2023,3,23,21,0,0, tzinfo=datetime.timezone.utc)
        
    else:
        filename = '/Users/mcordero/Data/IAP/SIMONe/Norway/VorTex/DNLEGAVG_AVG.txt'
        lat = 70.35
        lon = 14.25
        ref_label = 'DWLEG'
        dt = datetime.datetime(2023,3,23,21,0,0, tzinfo=datetime.timezone.utc)
    
    epoch = dt.timestamp()    
    
    df = pd.read_csv(filename,
                     # sep=' ',
                     header=0,
                     skiprows=1,
                     delim_whitespace=True,
                     names=['alt', 'u', 'v','ue', 've'],
                     )

    df = df.assign(lat=lat, lon=lon, datetime=dt, epoch=epoch)
    
    return(df)

def create_grid(hmin = 80,
                hmax = 100,
                hstep = 0.25,
                epoch=None,
                lon0=None,
                lat0=None
                ):
    
    if hstep is None: hstep = 1
    
    h       = np.arange(hmin, hmax, hstep)
    
    TIMES, ALT, LON, LAT = np.meshgrid(epoch, h, lon0, lat0, indexing='ij')
    
    coords = {}
    coords["times"] = TIMES
    coords["lat"]   = LAT
    coords["lon"]   = LON
    coords["alt"]   = ALT
    
    return(coords)

def winds_from_model(exp_folder, coords):
    
    T = coords["times"]
    X = coords["lon"]
    Y = coords["lat"]
    Z = coords["alt"]

    t = T.ravel()
    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()
    
    shape_3D = T.shape
    # Map flat indices to 3D indices
    flat_indices = np.arange(t.size)
    ind_3d = np.unravel_index(flat_indices, shape_3D)
    
    #Create winds and variances variables 
    u = np.zeros(T.shape, dtype=np.float32)
    v = np.zeros(T.shape, dtype=np.float32)
    w = np.zeros(T.shape, dtype=np.float32)
    
    u_var = np.zeros(T.shape, dtype=np.float32)
    v_var = np.zeros(T.shape, dtype=np.float32)
    w_var = np.zeros(T.shape, dtype=np.float32)
    
    #Mean to compute std deviation using Welford's method
    u0 = np.zeros(T.shape, dtype=np.float32)
    v0 = np.zeros(T.shape, dtype=np.float32)
    w0 = np.zeros(T.shape, dtype=np.float32)
    
    file_ensembles = glob.glob1(exp_folder, "h*.h5")
    
    N =  len(file_ensembles)
    
    for i, ifile in enumerate(file_ensembles):
    
        filename = os.path.join(exp_folder, ifile)
        
        nn = hyper.restore(filename)
    
        outputs = nn.infer(t, x, y, z, filter_output=False)
        
        u[ind_3d] = outputs[:,0]
        v[ind_3d] = outputs[:,1]
        w[ind_3d] = outputs[:,2]
        
        u0      += u/N
        v0      += v/N
        w0      += w/N
    
    u_std = np.zeros(T.shape, dtype=np.float32) 
    v_std = np.zeros(T.shape, dtype=np.float32) 
    w_std = np.zeros(T.shape, dtype=np.float32) 
    
    df = {}
    
    df["u"] = u0
    df["v"] = v0
    df["w"] = w0
    
    df["u_std"] = u_std
    df["v_std"] = v_std
    df["w_std"] = w_std
    
    df["times"] = coords["times"]
    df["lat"] = coords["lat"]
    df["lon"] = coords["lon"]
    df["alt"] = coords["alt"]
    
    if N < 2:
        return(df)
        
    for i, ifile in enumerate(file_ensembles):
    
        filename = os.path.join(exp_folder, ifile)
        
        nn = hyper.restore(filename)
    
        outputs = nn.infer(t, x, y, z, filter_output=False)
        
        u[ind_3d] = outputs[:,0]
        v[ind_3d] = outputs[:,1]
        w[ind_3d] = outputs[:,2]
        
        # Welford's method
        u_std   += np.square(u - u0)/(N-1)
        v_std   += np.square(v - v0)/(N-1)
        w_std   += np.square(w - w0)/(N-1)
    
    u_std = np.sqrt(u_std)
    v_std = np.sqrt(v_std)
    w_std = np.sqrt(w_std)
    
    df["u_std"] = u_std
    df["v_std"] = v_std
    df["w_std"] = w_std
    
    return(df)


def save_winds(df, path):
    
    times = df["times"]
    
    lat = df["lat"]
    lon = df["lon"]
    alt = df["alt"]
    
    dt = datetime.datetime.utcfromtimestamp(times[0,0,0,0])
    
    output_file = os.path.join(path, "winds_%s.h5" %dt.strftime("%Y%m%d") )
    
    # Open the file in write mode
    with h5py.File(output_file, "a") as fp:
        
        if "lat" not in fp: fp.create_dataset("lat", data=lat[0,0,0,0])
        if "lon" not in fp: fp.create_dataset("lon", data=lon[0,0,0,0])
        if "alt" not in fp: fp.create_dataset("alt", data=alt[0,:,0,0])
        
        for i, t in enumerate(times[:,0,0,0]):  # Iterate over the time dimension
            
            group_name = f"{t}"
            
            if group_name in fp:
                continue
                
              # Define a group name for each timestamp
            group = fp.create_group(group_name)  # Create a group for the timestamp
            
            group.create_dataset("u", data=df["u"][i])  # Save the 3D slice
            group.create_dataset("v", data=df["v"][i])
            group.create_dataset("w", data=df["w"][i])
            
            group.create_dataset("u_std", data=df["u_std"][i])  # Save the 3D slice
            group.create_dataset("v_std", data=df["v_std"][i])
            group.create_dataset("w_std", data=df["w_std"][i])

def plot_profiles(df, figpath,
                  df_ref = None,
                  ext = "png",
                  ref_label = 'UPLEG',
                  prefix = 'wind',
                  xlabel = 'Velocity (m/s)',
                  labels = ['u', 'v', 'w'],
                  vmin = -100,
                  vmax = 100,
                ):
    
    times = df["times"]
    
    lats = df["lat"]
    lons = df["lon"]
    alts = df["alt"]
    
    u = df["u"]
    v = df["v"]
    w = df["w"]
    
    u_std = df["u_std"]
    v_std = df["v_std"]
    w_std = df["w_std"]
        
    nlons = len(lons)
    nlats = len(lats)
    
    dt = datetime.datetime.utcfromtimestamp(times[0,0,0,0])
    
    figname = os.path.join(figpath, '%s_profile_%s_%3.2f_%3.2f.%s' %( prefix, dt.strftime("%Y%m%d_%H%M%S"), lons[0,0,0,0], lats[0,0,0,0], ext) )
    
    fig, axs = plt.subplots(nlons, nlats, figsize=(nlats*7, nlons*7), squeeze=False)
    
    newline = "\n"
    fig.suptitle(r'SIMONe Norway: %s @ %3.2f$^\circ$N, %3.2f$^\circ$E' %( dt.strftime("%Y-%m-%d"), lats[0,0,0,0], lons[0,0,0,0]) )
    
    linestyles = ['-', '--', '-.']
    
    if df_ref is not None:
        alt_ref = df_ref['alt'].values
        u_ref   = df_ref['u'].values 
        v_ref   = df_ref['v'].values
    
    for j in range(nlons):
        for k in range(nlats):
            
            ax = axs[j,k]
            
            for i, time in enumerate(times[:,0,0,0]):
                
                alpha = 0.2
                linestyle = linestyles[i]
                
                dt = datetime.datetime.utcfromtimestamp(time)
                h_str = dt.strftime("%H:%M UT")
                
                label_u = "(%s)" %h_str
                label_v = None
                label_w = None
                
                if i == 0:
                    alpha = 1
                    linestyle = '-'
                    
                    label_u = "SIMONe: %s (%s)" %(labels[0], h_str)
                    label_v = "SIMONe: %s (%s)" %(labels[1], h_str)
                    label_w = 'SIMONe: %s*10 (%s)' %(labels[2], h_str)
                
                # ax.plot(u[i,:,j,k], alts[i,:,j,k], 'k', linestyle=linestyle, alpha=alpha, label=label_u)
                # ax.plot(v[i,:,j,k], alts[i,:,j,k], 'b', linestyle=linestyle, alpha=alpha, label=label_v)
                
                ax.errorbar(u[i,:,j,k], alts[i,:,j,k], xerr=u_std[i,:,j,k], color='k', linestyle=linestyle, alpha=alpha, label=label_u)
                ax.errorbar(v[i,:,j,k], alts[i,:,j,k], xerr=v_std[i,:,j,k], color='b', linestyle=linestyle, alpha=alpha, label=label_v)
                
                if w is not None:
                    # ax.plot(10*w[i,:,j,k], alts[i,:,j,k], 'g', linestyle=linestyle, alpha=alpha, label=label_w)
                    
                    ax.errorbar(10*w[i,:,j,k], alts[i,:,j,k], xerr=w_std[i,:,j,k], color='g', linestyle=linestyle, alpha=alpha, label=label_w)
                
                
            if df_ref is not None:
                ax.plot(u_ref, alt_ref, 'ko', linestyle='-', alpha=0.4, label='%s: %s' %(ref_label, labels[0]) )
                ax.plot(v_ref, alt_ref, 'bo', linestyle='-', alpha=0.4, label='%s: %s' %(ref_label, labels[1]) )
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Height (km)')
            ax.grid(True)
            ax.set_ylim(80,100)
    
    plt.legend()
    
    ax.set_xlim(vmin,vmax)
    # ax.set_ylim(80,100)
    
    plt.tight_layout()
    plt.savefig(figname)
    plt.close('all')
    
    return

if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script to estimate 3D wind fields')
    
    parser.add_argument('-d', '--dpath', dest='path', default="/Users/mcordero/Data/IAP/SIMONe/Norway/VorTex/winds/c20230323", help='Data path')
    parser.add_argument('-r', '--rpath', dest='rpath', default=None, help='Data path')
    parser.add_argument('-e', '--extension', dest='ext', default='png', help='figures extension')
    parser.add_argument('-u', '--upleg', dest='upleg', default=1, help='')
                       
    args = parser.parse_args()
    
    path   = args.path
    rpath  = args.rpath
    ext    = args.ext
    upleg  = args.upleg
    
    if rpath is None:
        rpath = os.path.join(path, "plots")
        
    if not os.path.isdir(rpath):
        os.mkdir(rpath)
        
    df_vortex = read_upleg(upleg=upleg)
    
    print("Creating GRID ...")
    coords = create_grid(
                        epoch = df_vortex["epoch"][0],
                        lon0  = df_vortex["lon"][0],
                        lat0  = df_vortex["lat"][0],
                        )
    
    print("Generating winds ...")
    #Produce u, v, w, and std(u), std(v), std(w)
    df = winds_from_model(path, coords)
    
    print("Saving winds ...")
    save_winds(df, rpath)
    
    print("Plotting winds ...")
    plot_profiles(df, rpath, df_ref=df_vortex, ext=ext)