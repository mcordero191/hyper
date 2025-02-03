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

class Grid4D():
    
    def __init__(self, 
                    tstep=None, 
                    xstep=None,
                    ystep=None,
                    zstep=None,
                    xrange=None,
                    yrange=None,
                    zrange=None,
                    ):
        
        self.tstep = tstep
        self.xstep = xstep
        self.ystep = ystep
        self.zstep = zstep
        
        self.trange = None
        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange
    
    def read_coords(self, model_file):
        
        nn = hyper.restore(model_file)
    
        self.lon_ref = nn.lon_ref
        self.lat_ref = nn.lat_ref
        self.alt_ref = 90#nn.alt_ref
        
        x_lb = nn.lb[2]
        y_lb = nn.lb[3]
        z_lb = nn.lb[1]
        
        x_ub = nn.ub[2]
        y_ub = nn.ub[3]
        z_ub = nn.ub[1]
        
        tmin = nn.lb[0] + 30*60
        tmax = nn.ub[0] - 30*60
        
        trange = tmax - tmin
        
        if self.tstep is None: self.tstep = trange/(24*2)
        
        self.tmin = tmin
        self.tmax = tmax
        
        if self.xrange is None: self.xrange = np.ceil( (x_ub - x_lb)*1e-3 )
        if self.yrange is None: self.yrange = np.ceil( (y_ub - y_lb)*1e-3 )
        if self.zrange is None: self.zrange = np.ceil( (z_ub - z_lb)*1e-3 )
        
        if self.xstep is None: self.xstep = np.ceil(self.xrange/15)
        if self.ystep is None: self.ystep = np.ceil(self.yrange/15)
        if self.zstep is None: self.zstep = np.ceil(self.zrange/10)
        
    def set_spatial_grid(self):
        
        x = np.arange(0, self.xrange, self.xstep) - self.xrange/2
        y = np.arange(0, self.yrange, self.ystep) - self.yrange/2
        z = np.arange(0, self.zrange, self.zstep) - self.zrange/2 + self.alt_ref
        
        self.x_grid = x
        self.y_grid = y
        self.z_grid = z
        
    def set_temporal_grid(self):
        
        times = np.arange(self.tmin, self.tmax, self.tstep)
        
        self.t_grid = times
    
    def update(self, model_file):
        
        self.read_coords(model_file)
        self.set_spatial_grid()
        self.set_temporal_grid()
        
        X, Y, Z = np.meshgrid(self.x_grid, self.y_grid, self.z_grid, indexing='ij')
        
        lat, lon, alt = geo_coordinates.xyh2lla(X, Y, Z,
                                                lat_ref=self.lat_ref,
                                                lon_ref=self.lon_ref,
                                                alt_ref=self.alt_ref)
        
        coords = {}
    
        coords["t"] = self.t_grid
        coords["x"] = self.x_grid*1e3 #save in meters
        coords["y"] = self.y_grid*1e3
        coords["z"] = self.z_grid*1e3
        
        coords["lat_3D"] = lat
        coords["lon_3D"] = lon
        coords["alt_3D"] = alt
        
        return(coords)

class TimeAltitudePlot():
    
    def __init__(self, lon0=None, lat0=None, alt0=None):
        """
        Initialize the plot with the given parameters.
        
        Args:
            total_hours (int): Total time span for the plot (e.g., 24 hours).
            chunk_hours (int): Time span for each chunk (e.g., 3 hours).
            altitude_levels (int): Number of altitude levels.
        """
        
        self.lon0 = lon0
        self.lat0 = lat0
        self.alt0 = alt0
        
        self.latitudes = []  # Altitude levels
        self.longitudes = []  # Altitude levels
        self.altitudes = []  # Altitude levels
        
        self.times = []  # To accumulate time data
        
        self.datax_u = []  # To accumulate 2D data
        self.datax_v = []
        self.datax_w = []
        
        self.datay_u = []  # To accumulate 2D data
        self.datay_v = []
        self.datay_w = []
        
        self.dataz_u = []  # To accumulate 2D data
        self.dataz_v = []
        self.dataz_w = []

    def update_chunk(self, df):
        """
        Update the plot with a new chunk of data.
        
        Args:
            time_chunk (array-like): 1D array of time points for the chunk.
            data_chunk (array-like): 2D array of data for the chunk (time x altitude).
        """
        
        t   = df["t"]
        
        u = df["u"]
        v = df["v"]
        w = df["w"]
        
        u_std = df["u_std"]
        v_std = df["v_std"]
        w_std = df["w_std"]
        
        nt, nx, ny, nz = u.shape
        
        if self.lat0 is None: ix = nx//2
        else: ix = np.abs(df["lon_3D"][:,0,0] - self.lon0).argmin()
        
        if self.lon0 is None: iy = ny//2
        else: iy = np.abs(df["lat_3D"][0,:,0] - self.lat0).argmin()
        
        if self.alt0 is None: iz = nz//2
        else: iz = np.abs(df["alt_3D"][0,0,:] - self.alt0).argmin()
    
        lon = df["lon_3D"][:,iy,iz]
        lat = df["lat_3D"][ix,:,iz]
        alt = df["alt_3D"][ix,iy,:]
        
        keox_u = u[:,:,iy,iz]
        keox_v = v[:,:,iy,iz]
        keox_w = w[:,:,iy,iz]
        
        keoy_u = u[:,ix,:,iz]
        keoy_v = v[:,ix,:,iz]
        keoy_w = w[:,ix,:,iz]
        
        keoz_u = u[:,ix,iy,:]
        keoz_v = v[:,ix,iy,:]
        keoz_w = w[:,ix,iy,:]
        
        
        # Validate input dimensions
        if (len(t) != nt) or (nx != len(lon)) or (ny != len(lat)) or (nz != len(alt)):
            raise ValueError("Dimensions of (time, lon, lat, alt) and data do not match. (%d,%d,%d,%d)= %s" %(nt, nx, ny, nz, df["lon_3D"].shape))
        
        #Grid
        self.times.extend(t)
        
        self.longitudes = lon
        self.latitudes  = lat
        self.altitudes  = alt
        
        #Center
        self.lon0 = lon[ix]
        self.lat0 = lat[iy]
        self.alt0 = alt[iz]
        
        # Accumulate data
        self.datax_u.append(keox_u)
        self.datax_v.append(keox_v)
        self.datax_w.append(keox_w)
        
        self.datay_u.append(keoy_u)
        self.datay_v.append(keoy_v)
        self.datay_w.append(keoy_w)
        
        self.dataz_u.append(keoz_u)
        self.dataz_v.append(keoz_v)
        self.dataz_w.append(keoz_w)
        

    def save_plot(self, path,
                  vmins=[-100,-100,-5],
                  vmaxs=[ 100, 100, 5],
                  ):
        
        """
        Save the final plot as an image file.
        
        Args:
            filename (str): The name of the file to save.
        """
        
        dt = datetime.datetime.utcfromtimestamp(np.mean(self.times))
        
        
        times       = np.ravel(self.times)
        longitudes  = np.ravel(self.longitudes)
        latitudes   = np.ravel(self.latitudes)
        altitudes   = np.ravel(self.altitudes)
        
        data_u = np.vstack(self.datax_u)
        data_v = np.vstack(self.datax_v)
        data_w = np.vstack(self.datax_w)
        
        #Keogram X
        lat_h = r"[%3.2f$^\circ$N, %3.2fkm]" %(self.lat0, self.alt0)
        
        figfile = os.path.join(path, "keox_%s.%s" %(dt.strftime("%Y%m%d"), ext) )
        
        plotting.plot_mean_winds(times,
                                 longitudes,
                                 data_u, data_v, data_w,
                                 figfile,
                                 # titles=["u", "v", "w"],
                                 figtitle=lat_h,
                                 vmins=vmins,
                                 vmaxs=vmaxs,
                                 ylabel = "Longitude",
                                 )
        
        #Keogram Y
        
        data_u = np.vstack(self.datay_u)
        data_v = np.vstack(self.datay_v)
        data_w = np.vstack(self.datay_w)
        
        
        lon_h = r"[%3.2f$^\circ$E, %3.2fkm]" %(self.lon0, self.alt0)
        
        figfile = os.path.join(path, "keoy_%s.%s" %(dt.strftime("%Y%m%d"), ext) )
        
        plotting.plot_mean_winds(times,
                                 latitudes,
                                 data_u, data_v, data_w,
                                 figfile,
                                 # titles=["u", "v", "w"],
                                 figtitle=lon_h,
                                 vmins=vmins,
                                 vmaxs=vmaxs,
                                 ylabel = "Latitude",
                                 )
        
        #Keogram Z
        
        data_u = np.vstack(self.dataz_u)
        data_v = np.vstack(self.dataz_v)
        data_w = np.vstack(self.dataz_w)
        
        lat_lon = r"[%3.2f$^\circ$E, %3.2f$^\circ$N]" %(self.lon0, self.lat0)
        
        print(lat_lon, lon_h)
        
        figfile = os.path.join(path, "keoz_%s.%s" %(dt.strftime("%Y%m%d"), ext) )
        
        plotting.plot_mean_winds(times,
                                 altitudes,
                                 data_u, data_v, data_w,
                                 figfile,
                                 # titles=["u", "v", "w"],
                                 figtitle=lat_lon,
                                 vmins=vmins,
                                 vmaxs=vmaxs,
                                 )
    

def winds_from_model(ensemble_files, coords):
    
    t = coords["t"]
    x = coords["x"]
    y = coords["y"]
    z = coords["z"]
    
    # T,X,Y,Z = np.meshgrid(t,x,y,z, indexing="ij")
    
    X_3D = coords["lon_3D"]
    Y_3D = coords["lat_3D"]
    Z_3D = coords["alt_3D"]
    
    Nx, Ny, Nz = X_3D.shape
    
    # Expand to 4D by adding the time dimension
    T_4D = t[:, np.newaxis, np.newaxis, np.newaxis]  # Shape: (T, 1, 1, 1)
    X_4D = X_3D[np.newaxis, :, :, :]  # Shape: (1, Nx, Ny, Nz)
    Y_4D = Y_3D[np.newaxis, :, :, :]  # Shape: (1, Nx, Ny, Nz)
    Z_4D = Z_3D[np.newaxis, :, :, :]  # Shape: (1, Nx, Ny, Nz)
    
    # Use broadcasting to create a full 4D grid (T, Nx, Ny, Nz)
    T_mesh = np.broadcast_to(T_4D, (len(t), Nx, Ny, Nz))
    X_mesh = np.broadcast_to(X_4D, (len(t), Nx, Ny, Nz))
    Y_mesh = np.broadcast_to(Y_4D, (len(t), Nx, Ny, Nz))
    Z_mesh = np.broadcast_to(Z_4D, (len(t), Nx, Ny, Nz))

    shape_4D = T_mesh.shape
    
    # Map flat indices to 3D indices
    flat_indices = np.arange(T_mesh.size)
    ind_4d = np.unravel_index(flat_indices, shape_4D)
    
    #Single realization
    u = np.zeros(shape_4D, dtype=np.float32)
    v = np.zeros(shape_4D, dtype=np.float32)
    w = np.zeros(shape_4D, dtype=np.float32)
    
    #Mean to compute std deviation using Welford's method
    u0 = np.zeros(shape_4D, dtype=np.float32)
    v0 = np.zeros(shape_4D, dtype=np.float32)
    w0 = np.zeros(shape_4D, dtype=np.float32)
    
    u_std = np.zeros(shape_4D, dtype=np.float32) 
    v_std = np.zeros(shape_4D, dtype=np.float32) 
    w_std = np.zeros(shape_4D, dtype=np.float32) 
    
    N =  len(ensemble_files)
    
    for i, ifile in enumerate(ensemble_files):
        
        nn = hyper.restore(ifile)
    
        outputs, mask = nn.infer(T_mesh, X_mesh, Y_mesh, Z_mesh,
                           filter_output=False,
                           return_valid_mask=True)
        
        u[ind_4d] = outputs[:,0]
        v[ind_4d] = outputs[:,1]
        w[ind_4d] = outputs[:,2]
        
        u0      += u/N
        v0      += v/N
        w0      += w/N
    
    df = {}
    
    df["u"] = u0
    df["v"] = v0
    df["w"] = w0
    
    df["u_std"] = u_std
    df["v_std"] = v_std
    df["w_std"] = w_std
    
    df["t"] = t
    
    df["x"] = x #save in meters
    df["y"] = y
    df["z"] = z
    
    df["lat_3D"] = coords["lat_3D"]
    df["lon_3D"] = coords["lon_3D"]
    df["alt_3D"] = coords["alt_3D"]
    
    df["mask"] = mask.reshape(shape_4D)
    
    if N < 2:
        return(df)
        
    for i, ifile in enumerate(ensemble_files):
        
        nn = hyper.restore(ifile)
    
        outputs = nn.infer(T, x=X, y=Y, z=Z, filter_output=False)
        
        u[ind_4d] = outputs[:,0]
        v[ind_4d] = outputs[:,1]
        w[ind_4d] = outputs[:,2]
        
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
    
    #Include a mask to indicate good or bad meteor counts
    ####
    ####
    
    return(df)
    
def save_winds(df, output_file):
    
    t = df["t"]
    
    x = df["x"]
    y = df["y"]
    z = df["z"]
    
    lat = df["lat_3D"]
    lon = df["lon_3D"]
    alt = df["alt_3D"]
    
    mask = df["mask"][0]
    
    dt = datetime.datetime.utcfromtimestamp(np.mean(t))
    
    # Open the file in write mode
    with h5py.File(output_file, "a") as fp:
        
        # Add metadata to the file
        fp.attrs["Author"] = "Miguel Urco"
        fp.attrs["Email"] = "miguelcordero191@gmail.com"
        fp.attrs["Description"] = "4D wind data generated with HYPER from meteor radar measurements"
        fp.attrs["Reference"] = "Urco et al., 2024;  https://doi.org/10.1029/2024JH000162"
        
        if "x" not in fp: fp.create_dataset("x", data=x)
        if "y" not in fp: fp.create_dataset("y", data=y)
        if "z" not in fp: fp.create_dataset("z", data=z)
        
        if "lat" not in fp: fp.create_dataset("lat", data=lat)
        if "lon" not in fp: fp.create_dataset("lon", data=lon)
        if "alt" not in fp: fp.create_dataset("alt", data=alt)
        
        if "mask" not in fp: fp.create_dataset("mask", data=mask)
            
        for i, t in enumerate(t):  # Iterate over the time dimension
            
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
    
def find_ensemble_files(hourly_file):
    
    ensemble_files = sorted( glob.glob(hourly_file[:-6]+"*.h5") )
    
    return(ensemble_files)

if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script to produce 3D wind outputs')
    
    parser.add_argument('-m', '--mpath', dest='mpath', default="/Users/mcordero/Data/IAP/SIMONe/Germany/Simone2023/hyper24", help='Path where the model weights are')
    parser.add_argument('-r', '--rpath', dest='rpath', default=None, help='Path where the wind data will be saved')
    
    parser.add_argument('-g', '--gradients', dest='ena_gradients', default=0, help='Generate gradients too')
    
    parser.add_argument('-p', '--plotting', dest='ena_plotting', default=1, help='enable plotting')
    parser.add_argument('-e', '--extension', dest='ext', default='png', help='figure extension')
    
    parser.add_argument('--time-step', dest='tstep', default=5*60, help='in seconds')
    parser.add_argument('--x-step', dest='xstep', default=20, help='in km')
    parser.add_argument('--y-step', dest='ystep', default=20, help='in km')
    parser.add_argument('--z-step', dest='zstep', default=1, help='in km')
    
    parser.add_argument('--x-range', dest='xrange', default=None, help='in km')
    parser.add_argument('--y-range', dest='yrange', default=None, help='in km')
    parser.add_argument('--z-range', dest='zrange', default=None, help='in km')
    
    parser.add_argument('--lat-ref', dest='lat0', default=53.9, help='')
    parser.add_argument('--lon-ref', dest='lon0', default=12.6, help='')
    parser.add_argument('--alt-ref', dest='alt0', default=None, help='')
    
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
    alt0        = args.alt0
    
    cmap = 'seismic'
    
    if rpath is None:
        rpath_ = os.path.join(mpath, "winds")
    else:
        rpath_ = rpath
    
    if not os.path.isdir(rpath_):
        os.mkdir(rpath_)
        
    if rpath is None:
        figpath_ = os.path.join(mpath, "plots")
    else:
        figpath_ = rpath
    
    if not os.path.isdir(figpath_):
        os.mkdir(figpath_)
            
    day_folders = sorted(glob.glob1(mpath, "*"))
    
    if len(day_folders) < 1:
        print("No model folders found in %s" %mpath)
    
    #Create a new grid per day
    # print("Creating GRID ...")
    grid_4d = Grid4D(tstep=tstep, 
                    xstep=xstep,
                    ystep=ystep,
                    zstep=zstep,
                    xrange=xrange,
                    yrange=yrange,
                    zrange=zrange,
                    )
    
    for day_folder in day_folders:
        
        day_folder = os.path.join(mpath, day_folder)
        
        hourly_files = sorted( glob.glob( os.path.join(day_folder, "*_i000.h5") ) )
        
        if len(hourly_files) < 1:
            print("No model files found in %s" %day_folder)
            continue
            
        print("Processing experiment %s" %day_folder)
        
        plotData = TimeAltitudePlot(lon0=lon0, lat0=lat0, alt0=alt0)
        
        print("Generating winds ...")
        for hourly_file in hourly_files:
            
            #Update the temporal and spatial grid if required
            coords = grid_4d.update(hourly_file)
            
            dt = datetime.datetime.utcfromtimestamp(np.mean(coords['t']))
            
            output_file = os.path.join(rpath_, "winds3D_%s.h5" %dt.strftime("%Y%m%d") )
            
            # if os.path.exists(output_file):
            #     continue
            
            ensemble_files = find_ensemble_files(hourly_file)
            
            
            #Produce u, v, w, and std(u), std(v), std(w)
            df = winds_from_model(ensemble_files, coords)
            
            print(".", end='', flush=True)
            
            ##Directories are wrong!
            save_winds(df, output_file)
            
            plotData.update_chunk(df)
            # plot_winds(df, figpath_, ext=ext)
        
        print("Plotting winds ...")
        plotData.save_plot(path=figpath_)