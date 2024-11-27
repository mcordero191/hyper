'''
Created on Nov 10, 2024

@author: Miguel Urco
'''
import os, glob
import time, datetime
import numpy as np

import h5py

from pinn import hyper
from georeference import geo_coordinates
from utils import plotting

def create_3d_grid(exp_folfer,
                    tstep=None, 
                    xstep=None,
                    ystep=None,
                    zstep=None,
                    xrange=None,
                    yrange=None,
                    zrange=None,
                    lon0=None,
                    lat0=None
                    ):
    
    file_models = glob.glob1(exp_folfer, "h*.h5")
    
    filename = os.path.join(exp_folder, file_models[0])
    
    nn = hyper.restore(filename)
    
    lon_ref = nn.lon_ref
    lat_ref = nn.lat_ref
    alt_ref = nn.alt_ref
    
    tmin = nn.lb[0] + 30*60
    tmax = nn.ub[0] - 30*60
    
    trange = tmax - tmin
    
    x_lb = nn.lb[2]
    y_lb = nn.lb[3]
    z_lb = nn.lb[1]
    
    x_ub = nn.ub[2]
    y_ub = nn.ub[3]
    z_ub = nn.ub[1]
    
    if tstep is None: tstep = trange/(24*2)
    if xstep is None: xstep = 10
    if ystep is None: ystep = 10
    if zstep is None: zstep = 1
    
    if xrange is None: xrange = (x_ub - x_lb)*1e-3
    if yrange is None: yrange = (y_ub - y_lb)*1e-3
    if zrange is None: zrange = (z_ub - z_lb)*1e-3
    
    xmin = np.floor( -xrange/2.)
    xmax = np.ceil( xrange/2.)
    
    ymin = np.floor(-yrange/2.)
    ymax = np.ceil( yrange/2.)
    
    zmin = np.floor(alt_ref - zrange/2.)
    zmax = np.ceil(alt_ref + zrange/2.)
    
    times   = np.arange(tmin, tmax, tstep)
    x       = np.arange(xmin, xmax, xstep)
    y       = np.arange(ymin, ymax, ystep)
    z       = np.arange(zmin, zmax, zstep)
    
    T, X, Y, Z = np.meshgrid(times, x, y, z, indexing='ij')
    
    lat, lon, alt = geo_coordinates.xyh2lla(X, Y, Z,
                                            lat_ref=lat_ref,
                                            lon_ref=lon_ref,
                                            alt_ref=alt_ref)
    
    coords = {}
    
    coords["T"] = T
    
    coords["X"] = X*1e3 #save in meters
    coords["Y"] = Y*1e3
    coords["Z"] = Z*1e3
    
    coords["lat"] = lat
    coords["lon"] = lon
    coords["alt"] = alt
    
    return(coords)

def winds_from_model(exp_folder, coords):
    
    T = coords["T"]
    X = coords["X"]
    Y = coords["Y"]
    Z = coords["Z"]

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
    
        outputs = nn.infer(t, x=x, y=y, z=z, filter_output=False)
        
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
    
    df["T"] = T
    df["X"] = X #save in meters
    df["Y"] = Y
    df["Z"] = Z
    
    df["lat"] = coords["lat"]
    df["lon"] = coords["lon"]
    df["alt"] = coords["alt"]
    
    if N < 2:
        return(df)
        
    for i, ifile in enumerate(file_ensembles):
    
        filename = os.path.join(exp_folder, ifile)
        
        nn = hyper.restore(filename)
    
        outputs = nn.infer(t, x=x, y=y, z=z, filter_output=False)
        
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
    
    T = df["T"]
    
    X = df["X"]
    Y = df["Y"]
    Z = df["Z"]
    
    lat = df["lat"]
    lon = df["lon"]
    alt = df["alt"]
    
    dt = datetime.datetime.utcfromtimestamp(T[0,0,0,0])
    
    output_file = os.path.join(path, "winds3D_%s.h5" %dt.strftime("%Y%m%d") )
    
    # Open the file in write mode
    with h5py.File(output_file, "a") as fp:
        
        if "X" not in fp: fp.create_dataset("X", data=X[0])
        if "Y" not in fp: fp.create_dataset("Y", data=Y[0])
        if "Z" not in fp: fp.create_dataset("Z", data=Z[0])
        
        if "lat" not in fp: fp.create_dataset("lat", data=lat[0])
        if "lon" not in fp: fp.create_dataset("lon", data=lon[0])
        if "alt" not in fp: fp.create_dataset("alt", data=alt[0])
        
        for i, t in enumerate(T[:,0,0,0]):  # Iterate over the time dimension
            
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

def plot_winds(df, path, ext="png", lat0=None, lon0=None, h0=None):
    
    nt, nx, ny, nz = df["T"].shape
    
    if lat0 is None: ix = nx//2
    else: ix = np.abs(df["lon"][0,:,0,0] - lon0).argmin()
    
    if lon0 is None: iy = ny//2
    else: iy = np.abs(df["lat"][0,0,:,0] - lat0).argmin()
    
    if h0 is None: iz = nz//2
    else: iz = np.abs(df["alt"][0,0,0,:] - h0).argmin()
    
    t   = df["T"][:,ix, iy, iz]
    lon = df["lon"][0,:,iy,iz]
    lat = df["lat"][0,ix,:,iz]
    h   = df["alt"][0,ix,iy,:]
    
    u = df["u"]
    v = df["v"]
    w = df["w"]
    
    u_std = df["u_std"]
    v_std = df["v_std"]
    w_std = df["w_std"]
    
    dt = datetime.datetime.utcfromtimestamp(t[0])
    
    #Keogram X
    lat_h = "[%3.2fN,%3.2fkm]" %(lat[iy], h[iz])
    
    figfile1 = os.path.join(path, "keox_%s_%s.%s" %(dt.strftime("%Y%m%d"), lat_h, ext) )
    figfile2 = os.path.join(path, "keox_std_%s_%s.%s" %(dt.strftime("%Y%m%d"), lat_h, ext) )
    
    plotting.plot_mean_winds(t, lon, u[:,:,iy,iz], v[:,:,iy,iz], w[:,:,iy,iz],
                             figfile1,
                             titles=["u", "v", "w"],
                             vmins=[-100,-100,-5],
                             vmaxs=[ 100, 100, 5],
                             ylabel = "Latitude",
                             )
    
    plotting.plot_mean_winds(t, lon, u_std[:,:,iy,iz], v_std[:,:,iy,iz], w_std[:,:,iy,iz],
                             figfile2,
                             titles=["std(u)", "std(v)", "std(w)"],
                             vmins=[0,0,0],
                             vmaxs=[ 10, 10, 1],
                             cmap='jet',
                             ylabel = "Latitude",
                             )
    
    #Keogram Y
    lon_h = "[%3.2fE,%3.2fkm]" %(lon[ix], h[iz])
    
    figfile1 = os.path.join(path, "keoy_%s_%s.%s" %(dt.strftime("%Y%m%d"), lon_h, ext) )
    figfile2 = os.path.join(path, "keoy_std_%s_%s.%s" %(dt.strftime("%Y%m%d"), lon_h, ext) )
    
    plotting.plot_mean_winds(t, lat, u[:,ix,:,iz], v[:,ix,:,iz], w[:,ix,:,iz],
                             figfile1,
                             titles=["u", "v", "w"],
                             vmins=[-100,-100,-5],
                             vmaxs=[ 100, 100, 5],
                             ylabel = "Longitude",
                             )
    
    plotting.plot_mean_winds(t, lat, u_std[:,ix,:,iz], v_std[:,ix,:,iz], w_std[:,ix,:,iz],
                             figfile2,
                             titles=["std(u)", "std(v)", "std(w)"],
                             vmins=[0,0,0],
                             vmaxs=[ 10, 10, 1],
                             cmap='jet',
                             ylabel = "Longitude",
                             )
    
    #Keogram Z
    lat_lon = "[%3.2fE,%3.2fN]" %(lon[ix], lat[iy])
    
    figfile1 = os.path.join(path, "keoz_%s_%s.%s" %(dt.strftime("%Y%m%d"), lat_lon, ext) )
    figfile2 = os.path.join(path, "keoz_std_%s_%s.%s" %(dt.strftime("%Y%m%d"), lat_lon, ext) )
    
    plotting.plot_mean_winds(t, h, u[:,ix,iy,:], v[:,ix,iy,:], w[:,ix,iy,:],
                             figfile1,
                             titles=["u", "v", "w"],
                             vmins=[-100,-100,-5],
                             vmaxs=[ 100, 100, 5],
                             )
    
    plotting.plot_mean_winds(t, h, u_std[:,ix,iy,:], v_std[:,ix,iy,:], w_std[:,ix,iy,:],
                             figfile2,
                             titles=["std(u)", "std(v)", "std(w)"],
                             vmins=[0,0,0],
                             vmaxs=[ 10, 10, 1],
                             cmap='jet'
                             )
    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script to produce 3D wind outputs')
    
    parser.add_argument('-m', '--mpath', dest='mpath', default="/Users/mcordero/Data/IAP/SIMONe/Norway/VorTex/winds", help='Path where the model weights are')
    parser.add_argument('-r', '--rpath', dest='rpath', default=None, help='Path where the wind data will be saved')
    
    parser.add_argument('-g', '--gradients', dest='ena_gradients', default=0, help='Generate gradients too')
    
    parser.add_argument('-p', '--plotting', dest='ena_plotting', default=1, help='enable plotting')
    parser.add_argument('-e', '--extension', dest='ext', default='png', help='figure extension')
    
    parser.add_argument('--time-step', dest='tstep', default=30*60, help='in seconds')
    parser.add_argument('--x-step', dest='xstep', default=None, help='in km')
    parser.add_argument('--y-step', dest='ystep', default=None, help='in km')
    parser.add_argument('--z-step', dest='zstep', default=None, help='in km')
    
    parser.add_argument('--x-range', dest='xrange', default=None, help='in km')
    parser.add_argument('--y-range', dest='yrange', default=None, help='in km')
    parser.add_argument('--z-range', dest='zrange', default=None, help='in km')
    
    parser.add_argument('--lat-ref', dest='lat0', default=None, help='')
    parser.add_argument('--lon-ref', dest='lon0', default=None, help='')
    
    args = parser.parse_args()
    
    mpath = args.mpath
    rpath = args.rpath
    
    ena_grads   = args.ena_gradients
    
    ena_plot    = args.ena_plotting
    ext         = args.ext
    
    tstep       = args.tstep
    xstep       = args.xstep
    ystep       = args.ystep
    zstep       = args.zstep
    
    xrange      = args.xrange
    yrange      = args.yrange
    zrange      = args.zrange
    
    lat0        = args.lat0
    lon0        = args.lon0
    
    cmap = 'seismic'
    
    exp_folders = glob.glob1(mpath, "*")
    
    if len(exp_folders) < 1:
        print("No model folders found in %s" %mpath)
            
    for exp_folder in exp_folders:
        
        exp_folder = os.path.join(mpath, exp_folder)
        
        print("Processing experiment %s" %exp_folder)
        
        if rpath is None:
            rpath_ = exp_folder
        
        print("Creating GRID ...")
        coords_3d = create_3d_grid(exp_folder,
                                    tstep=tstep, 
                                    xstep=xstep,
                                    ystep=ystep,
                                    zstep=zstep,
                                    xrange=xrange,
                                    yrange=yrange,
                                    zrange=zrange,
                                    lon0=lon0,
                                    lat0=lat0,
                                    )
        
        print("Generating winds ...")
        #Produce u, v, w, and std(u), std(v), std(w)
        df = winds_from_model(exp_folder, coords_3d)
        
        print("Saving winds ...")
        save_winds(df, rpath_)
        
        print("Plotting winds ...")
        plot_winds(df, rpath_, ext=ext)
        