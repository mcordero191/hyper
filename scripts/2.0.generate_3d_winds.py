#!/usr/bin/env python
"""
Created on Nov 10, 2024

@author: Miguel Urco
Modified to use geographic (lon, lat, alt) grid parameters.
"""

import os, glob
import time, datetime
import numpy as np
import h5py
import netCDF4

from pinn import hyper
from utils import plotting, version_history

version_history_file = "hyper_history.json"

class Grid4D():
    
    def __init__(self, 
                 tstep=None, 
                 lon_step=None,
                 lat_step=None,
                 alt_step=None,
                 lon_range=None,
                 lat_range=None,
                 alt_range=None,
                 lon_ref=None,
                 lat_ref=None,
                 alt_ref=None,
                 # Optional direct coordinate inputs:
                 t_coords=None,
                 lon_coords=None,
                 lat_coords=None,
                 alt_coords=None,
                 ):
        
        self.tstep = tstep
        self.lon_step = lon_step
        self.lat_step = lat_step
        self.alt_step = alt_step
        
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.alt_range = alt_range
        
        self.lon_ref = lon_ref
        self.lat_ref = lat_ref
        self.alt_ref = alt_ref
        
        # Direct grid input (if provided, these should be lists of floats)
        self.t_coords = t_coords
        self.lon_coords = lon_coords
        self.lat_coords = lat_coords
        self.alt_coords = alt_coords

    def read_coords(self, model_file):
        
        nn = hyper.restore(model_file)
        
        # Set reference coordinates if not provided
        if self.lon_ref is None:
            self.lon_ref = nn.lon_ref
            
        if self.lat_ref is None:
            self.lat_ref = nn.lat_ref
            
        if self.alt_ref is None:
            self.alt_ref = nn.alt_ref  # Using model altitude reference
        
        # Determine temporal bounds from the model file
        
        xmin = nn.lb[2]
        xmax = nn.ub[2]
        
        ymin = nn.lb[3]
        ymax = nn.ub[3]
        
        zmin = nn.lb[1]
        zmax = nn.ub[1]
        
        tmin = int(nn.lb[0]/300)*300 + 30*60
        tmax = int(nn.ub[0]/300)*300 - 30*60
        
        trange = tmax - tmin
        xrange = xmax - xmin
        yrange = ymax - ymin
        zrange = zmax - zmin
        
        if self.tstep is None:
            self.tstep = max(int(trange/300)*300/(24*2), 300) #Minimun 5min = 300seconds
            
        self.tmin = tmin
        self.tmax = tmax
        
        def m_to_deg_lon(meters, lat0):
            """Convert meters to degrees longitude at a given center latitude (lat0 in degrees)."""
            return meters / (111.32e3 * np.cos(np.radians(lat0)))
        
        def m_to_deg_lat(meters):
            """Convert meters to degrees latitude."""
            return meters / 111.32e3

        # Set default spatial ranges if not provided (lon/lat in degrees, alt in km)
        if self.lon_range is None:
            self.lon_range = m_to_deg_lon(xrange, self.lat_ref)   # default: 1 degree longitude range
            
        if self.lat_range is None:
            self.lat_range = m_to_deg_lat(yrange)   # default: 1 degree latitude range
            
        if self.alt_range is None:
            self.alt_range = int(zrange/2000)*2  # default: 10 km altitude range

        if self.lon_step is None:
            self.lon_step = self.lon_range/20.0
        
        if self.lat_step is None:
            self.lat_step = self.lat_range/20.0
        
        if self.alt_step is None:
            self.alt_step = self.alt_range/40.0
        
    def set_spatial_grid(self):
        
        if self.lon_coords is not None and self.lat_coords is not None and self.alt_coords is not None:
            
            self.lon_grid = np.sort(self.lon_coords)
            self.lat_grid = np.sort(self.lat_coords)
            self.alt_grid = np.sort(self.alt_coords)
            
        else:
            # Compute grid boundaries centered at the reference coordinates
            lon_min = self.lon_ref - self.lon_range / 2
            lon_max = self.lon_ref + self.lon_range / 2
            
            lat_min = self.lat_ref - self.lat_range / 2
            lat_max = self.lat_ref + self.lat_range / 2
            
            alt_min = self.alt_ref - self.alt_range / 2
            alt_max = self.alt_ref + self.alt_range / 2
    
            # Create uniform grids in geographic coordinates
            self.lon_grid = np.arange(lon_min, lon_max, self.lon_step)
            self.lat_grid = np.arange(lat_min, lat_max, self.lat_step)
            self.alt_grid = np.arange(alt_min, alt_max, self.alt_step)
    
            # Create a 3D mesh grid (lon, lat, alt)
        self.lon_3D, self.lat_3D, self.alt_3D = np.meshgrid(self.lon_grid, self.lat_grid, self.alt_grid, indexing='ij')
    
    def set_temporal_grid(self):
        
        if self.t_coords is not None:
            times = np.array(self.t_coords)
            times = times[(times >= self.tmin) & (times <= self.tmax)]
        else:
            times = np.arange(self.tmin, self.tmax, self.tstep)
            
        self.t_grid = times

    def update(self, model_file):
        
        self.read_coords(model_file)
        self.set_spatial_grid()
        self.set_temporal_grid()

        coords = {}
        
        coords["t"]   = self.t_grid
        coords["lon"] = self.lon_grid
        coords["lat"] = self.lat_grid
        coords["alt"] = self.alt_grid
        coords["lon_3D"] = self.lon_3D
        coords["lat_3D"] = self.lat_3D
        coords["alt_3D"] = self.alt_3D
        
        return coords

class TimeAltitudePlot():
    
    def __init__(self, lon0=None, lat0=None, alt0=None):
        
        """
        Initialize the plot with the given parameters.
        """
        self.lon0 = lon0
        self.lat0 = lat0
        self.alt0 = alt0
        
        self.latitudes = []  
        self.longitudes = []  
        self.altitudes = []  
        self.times = []  
        
        self.datax_u = []  
        self.datax_v = []
        self.datax_w = []
        
        self.datay_u = []  
        self.datay_v = []
        self.datay_w = []
        
        self.dataz_u = []  
        self.dataz_v = []
        self.dataz_w = []

    def update_chunk(self, df, plot_std=False):
        t   = df["t"]
        u = df["u"]
        v = df["v"]
        w = df["w"]
        
        if plot_std:
            u = df["u_std"]
            v = df["v_std"]
            w = df["w_std"]
        
        nt, nx, ny, nz = u.shape
        
        if self.lat0 is None: 
            ix = nx // 2
        else: 
            ix = np.abs(df["lon_3D"][:,0,0] - self.lon0).argmin()
        
        if self.lon0 is None: 
            iy = ny // 2
        else: 
            iy = np.abs(df["lat_3D"][0,:,0] - self.lat0).argmin()
        
        if self.alt0 is None: 
            iz = nz // 2
        else: 
            iz = np.abs(df["alt_3D"][0,0,:] - self.alt0).argmin()
    
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
        
        if (len(t) != nt) or (nx != len(lon)) or (ny != len(lat)) or (nz != len(alt)):
            raise ValueError("Dimensions of (time, lon, lat, alt) and data do not match. (%d,%d,%d,%d)= %s" % (nt, nx, ny, nz, df["lon_3D"].shape))
        
        self.times.extend(t)
        self.longitudes = lon
        self.latitudes  = lat
        self.altitudes  = alt
        
        # Update center coordinates
        self.lon0 = lon[ix]
        self.lat0 = lat[iy]
        self.alt0 = alt[iz]
        
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
                  cmap='seismic',
                  suffix="",
                  ):
        
        
        dt = datetime.datetime.utcfromtimestamp(np.min(self.times))
        times       = np.ravel(self.times)
        longitudes  = np.ravel(self.longitudes)
        latitudes   = np.ravel(self.latitudes)
        altitudes   = np.ravel(self.altitudes)
        
        data_u = np.vstack(self.datax_u)
        data_v = np.vstack(self.datax_v)
        data_w = np.vstack(self.datax_w)
        
        lat_h = r"[%3.2f$^\circ$N, %3.2fkm]" % (self.lat0, self.alt0)
        figfile = os.path.join(path, "keox_%s_%s.%s" % (dt.strftime("%Y%m%d_%H%M%S"), suffix, ext))
        
        plotting.plot_mean_winds(times,
                                 longitudes,
                                 data_u, data_v, data_w,
                                 figfile,
                                 figtitle=lat_h,
                                 vmins=vmins,
                                 vmaxs=vmaxs,
                                 cmap=cmap,
                                 ylabel = "Longitude")
        
        data_u = np.vstack(self.datay_u)
        data_v = np.vstack(self.datay_v)
        data_w = np.vstack(self.datay_w)
        
        lon_h = r"[%3.2f$^\circ$E, %3.2fkm]" % (self.lon0, self.alt0)
        figfile = os.path.join(path, "keoy_%s_%s.%s" % (dt.strftime("%Y%m%d_%H%M%S"), suffix, ext))
        
        plotting.plot_mean_winds(times,
                                 latitudes,
                                 data_u, data_v, data_w,
                                 figfile,
                                 figtitle=lon_h,
                                 vmins=vmins,
                                 vmaxs=vmaxs,
                                 cmap=cmap,
                                 ylabel = "Latitude")
        
        data_u = np.vstack(self.dataz_u)
        data_v = np.vstack(self.dataz_v)
        data_w = np.vstack(self.dataz_w)
        
        lat_lon = r"[%3.2f$^\circ$E, %3.2f$^\circ$N]" % (self.lon0, self.lat0)
        # print(lat_lon, lon_h)
        
        figfile = os.path.join(path, "keoz_%s_%s.%s" % (dt.strftime("%Y%m%d_%H%M%S"), suffix, ext))
        
        plotting.plot_mean_winds(times,
                                 altitudes,
                                 data_u, data_v, data_w,
                                 figfile,
                                 figtitle=lat_lon,
                                 vmins=vmins,
                                 vmaxs=vmaxs,
                                 cmap=cmap)

class TwoDPlanesPlot():
    def __init__(self, lon0=None, lat0=None, alt0=None):
        self.lon0 = lon0
        self.lat0 = lat0
        self.alt0 = alt0
        
        self.latitudes = []
        self.longitudes = []
        self.altitudes = []
        self.times = []
        
        self.datax_u = []
        self.datax_v = []
        self.datax_w = []
        
        self.datay_u = []
        self.datay_v = []
        self.datay_w = []
        
        self.dataz_u = []
        self.dataz_v = []
        self.dataz_w = []

    def update_chunk(self, df, plot_std=False):
        t = df["t"]
        u = df["u"]
        v = df["v"]
        w = df["w"]
        
        if plot_std:
            u = df["u_std"]
            v = df["v_std"]
            w = df["w_std"]
        
        nt, nx, ny, nz = u.shape
        
        if self.lat0 is None: 
            ix = nx // 2
        else: 
            ix = np.abs(df["lon_3D"][:,0,0] - self.lon0).argmin()
        
        if self.lon0 is None: 
            iy = ny // 2
        else: 
            iy = np.abs(df["lat_3D"][0,:,0] - self.lat0).argmin()
        
        if self.alt0 is None: 
            iz = nz // 2
        else: 
            iz = np.abs(df["alt_3D"][0,0,:] - self.alt0).argmin()
    
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
        
        if (len(t) != nt) or (nx != len(lon)) or (ny != len(lat)) or (nz != len(alt)):
            raise ValueError("Dimensions of (time, lon, lat, alt) and data do not match. (%d,%d,%d,%d)= %s" % (nt, nx, ny, nz, df["lon_3D"].shape))
        
        self.times.extend(t)
        self.longitudes = lon
        self.latitudes  = lat
        self.altitudes  = alt
        
        self.lon0 = lon[ix]
        self.lat0 = lat[iy]
        self.alt0 = alt[iz]
        
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
                  cmap='seismic',
                  sufix=""):
        
        dt = datetime.datetime.utcfromtimestamp(np.mean(self.times))
        times       = np.ravel(self.times)
        longitudes  = np.ravel(self.longitudes)
        latitudes   = np.ravel(self.latitudes)
        altitudes   = np.ravel(self.altitudes)
        
        data_u = np.vstack(self.datax_u)
        data_v = np.vstack(self.datax_v)
        data_w = np.vstack(self.datax_w)
        
        lat_h = r"[%3.2f$^\circ$N, %3.2fkm]" % (self.lat0, self.alt0)
        figfile = os.path.join(path, "keox_%s_%s.%s" % (dt.strftime("%Y%m%d"), sufix, ext))
        
        plotting.plot_mean_winds(times,
                                 longitudes,
                                 data_u, data_v, data_w,
                                 figfile,
                                 figtitle=lat_h,
                                 vmins=vmins,
                                 vmaxs=vmaxs,
                                 cmap=cmap,
                                 ylabel = "Longitude")
        
        data_u = np.vstack(self.datay_u)
        data_v = np.vstack(self.datay_v)
        data_w = np.vstack(self.datay_w)
        
        lon_h = r"[%3.2f$^\circ$E, %3.2fkm]" % (self.lon0, self.alt0)
        figfile = os.path.join(path, "keoy_%s_%s.%s" % (dt.strftime("%Y%m%d"), sufix, ext))
        
        plotting.plot_mean_winds(times,
                                 latitudes,
                                 data_u, data_v, data_w,
                                 figfile,
                                 figtitle=lon_h,
                                 vmins=vmins,
                                 vmaxs=vmaxs,
                                 cmap=cmap,
                                 ylabel = "Latitude")
        
        data_u = np.vstack(self.dataz_u)
        data_v = np.vstack(self.dataz_v)
        data_w = np.vstack(self.dataz_w)
        
        lat_lon = r"[%3.2f$^\circ$E, %3.2f$^\circ$N]" % (self.lon0, self.lat0)
        print(lat_lon, lon_h)
        
        figfile = os.path.join(path, "keoz_%s_%s.%s" % (dt.strftime("%Y%m%d"), sufix, ext))
        
        plotting.plot_mean_winds(times,
                                 altitudes,
                                 data_u, data_v, data_w,
                                 figfile,
                                 figtitle=lat_lon,
                                 vmins=vmins,
                                 vmaxs=vmaxs,
                                 cmap=cmap)

            
def find_ensemble_files(hourly_file):
    
    idx = hourly_file.find("i000")
    
    ensemble_files = sorted(glob.glob( "%s*%s" %(hourly_file[:idx], hourly_file[idx+4:]) ) )
    
    return ensemble_files
        
def winds_from_model(ensemble_files, coords):
    
    t = coords["t"]
    
    lon_3D = coords["lon_3D"]
    lat_3D = coords["lat_3D"]
    alt_3D = coords["alt_3D"]
    
    Nx, Ny, Nz = lon_3D.shape
    
    T_4D = t[:, np.newaxis, np.newaxis, np.newaxis]
    X_4D = lon_3D[np.newaxis, :, :, :]
    Y_4D = lat_3D[np.newaxis, :, :, :]
    Z_4D = alt_3D[np.newaxis, :, :, :]
    
    # Use broadcasting to create a full 4D grid (T, Nx, Ny, Nz)
    T_mesh = np.broadcast_to(T_4D, (len(t), Nx, Ny, Nz))
    X_mesh = np.broadcast_to(X_4D, (len(t), Nx, Ny, Nz))
    Y_mesh = np.broadcast_to(Y_4D, (len(t), Nx, Ny, Nz))
    Z_mesh = np.broadcast_to(Z_4D, (len(t), Nx, Ny, Nz))
    
    shape_4D = T_mesh.shape
    flat_indices = np.arange(T_mesh.size)
    ind_4d = np.unravel_index(flat_indices, shape_4D)
    
    u = np.zeros(shape_4D, dtype=np.float32)
    v = np.zeros(shape_4D, dtype=np.float32)
    w = np.zeros(shape_4D, dtype=np.float32)
    
    u0 = np.zeros(shape_4D, dtype=np.float32)
    v0 = np.zeros(shape_4D, dtype=np.float32)
    w0 = np.zeros(shape_4D, dtype=np.float32)
    
    u_std = np.zeros(shape_4D, dtype=np.float32) 
    v_std = np.zeros(shape_4D, dtype=np.float32) 
    w_std = np.zeros(shape_4D, dtype=np.float32) 
    
    N = len(ensemble_files)
    
    for i, ifile in enumerate(ensemble_files):
        nn = hyper.restore(ifile)
        outputs, mask = nn.infer(T_mesh, X_mesh, Y_mesh, Z_mesh,
                                 filter_output=False,
                                 return_valid_mask=True)
        u[ind_4d] = outputs[:,0]
        v[ind_4d] = outputs[:,1]
        w[ind_4d] = outputs[:,2]
        
        u0 += u / N
        v0 += v / N
        w0 += w / N
    
    df = {}
    
    df["version"] = nn.version
    
    df["u"] = u0
    df["v"] = v0
    df["w"] = w0
    
    df["u_std"] = u_std
    df["v_std"] = v_std
    df["w_std"] = w_std
    
    df["t"] = t
    df["lon"] = coords["lon"]
    df["lat"] = coords["lat"]
    df["alt"] = coords["alt"]
    
    df["lon_3D"] = lon_3D
    df["lat_3D"] = lat_3D
    df["alt_3D"] = alt_3D
    
    df["mask"] = mask.reshape(shape_4D)
    
    if N < 2:
        return df
        
    for i, ifile in enumerate(ensemble_files):
        nn = hyper.restore(ifile)
        outputs = nn.infer(T_mesh, X_mesh, Y_mesh, Z_mesh,
                           filter_output=False)
        u[ind_4d] = outputs[:,0]
        v[ind_4d] = outputs[:,1]
        w[ind_4d] = outputs[:,2]
        
        u_std += np.square(u - u0) / (N-1)
        v_std += np.square(v - v0) / (N-1)
        w_std += np.square(w - w0) / (N-1)
    
    u_std = np.sqrt(u_std)
    v_std = np.sqrt(v_std)
    w_std = np.sqrt(w_std)
    
    df["u_std"] = u_std
    df["v_std"] = v_std
    df["w_std"] = w_std
    
    return df

def save_winds(df, output_file):
    
    t = df["t"]
    lon = df["lon"]
    lat = df["lat"]
    alt = df["alt"]
    
    mask = df["mask"][0]
    dt = datetime.datetime.utcfromtimestamp(np.mean(t))
    
    with h5py.File(output_file, "a") as fp:
        
        fp.attrs["Author"] = "Miguel Urco"
        fp.attrs["Email"] = "miguelcordero191@gmail.com"
        fp.attrs["Description"] = "4D wind data generated with HYPER from meteor radar measurements"
        fp.attrs["Reference"] = "Urco et al., 2024; https://doi.org/10.1029/2024JH000162"
        
        if "lon" not in fp: 
            fp.create_dataset("lon", data=lon)
        if "lat" not in fp: 
            fp.create_dataset("lat", data=lat)
        if "alt" not in fp: 
            fp.create_dataset("alt", data=alt)
        
        if "mask" not in fp: 
            fp.create_dataset("mask", data=mask)
            
        for i, t_val in enumerate(t):
            group_name = f"{t_val}"
            if group_name in fp:
                continue
            group = fp.create_group(group_name)
            group.create_dataset("u", data=df["u"][i])
            group.create_dataset("v", data=df["v"][i])
            group.create_dataset("w", data=df["w"][i])
            group.create_dataset("u_std", data=df["u_std"][i])
            group.create_dataset("v_std", data=df["v_std"][i])
            group.create_dataset("w_std", data=df["w_std"][i])

def append_winds_hdf5(df, output_file, overwrite=False):
    
    # Extract time and spatial coordinate arrays from the input dictionary
    t = np.atleast_1d(df["t"])  # Expecting a 1D array (or single value) for time
    
    lat = df["lat_3D"]
    lon = df["lon_3D"]
    alt = df["alt_3D"]
    
    # Use the first mask slice (assumed constant over time)
    mask = df["mask"][0]
    
    # For metadata purposes, compute a representative datetime (not used for indexing here)
    dt = datetime.datetime.utcfromtimestamp(np.mean(t))
    
    # Read version and history info from external JSON file
    current_version = df['version'] #version_history.get_current_version(version_history_file)
    history_text = version_history.get_version_history(version_history_file)
    
    filename = output_file + "_v%s.h5" %current_version
    
    # If clean is True and file exists, remove it.
    if overwrite and os.path.exists(filename):
        os.remove(filename)
        
    with h5py.File(filename, "a") as fp:
        # --- File-level metadata ---
        fp.attrs["Author"] = "Miguel Urco Cordero"
        fp.attrs["Email"] = "miguelcordero191@gmail.com"
        fp.attrs["Description"] = "4D wind data generated with HYPER from meteor radar measurements"
        fp.attrs["Reference"] = "Urco et al., 2024; https://doi.org/10.1029/2024JH000162"
        fp.attrs["Date"] = dt.strftime("%Y-%m-%d")
        fp.attrs["version"] = current_version
        fp.attrs["history"] = history_text
        
        # --- Static datasets (spatial coordinates and mask) --
        if "lat" not in fp:
            dset = fp.create_dataset("lat", data=lat)
            dset.attrs["units"] = "Degrees"
        if "lon" not in fp:
            dset = fp.create_dataset("lon", data=lon)
            dset.attrs["units"] = "Degrees"
        if "alt" not in fp:
            dset = fp.create_dataset("alt", data=alt)
            dset.attrs["units"] = "km"
        
        if "mask" not in fp:
            fp.create_dataset("mask", data=mask)
        
        # --- Time coordinate dataset ---
        if "times" not in fp:
            t_dset = fp.create_dataset("times", data=np.array(t), 
                                       maxshape=(None,), chunks=True)
            t_dset.attrs["units"] = "seconds since 1970/01/01"  # Adjust time units as needed
        else:
            t_dset = fp["times"]
            current_len = t_dset.shape[0]
            new_len = current_len + len(t)
            t_dset.resize((new_len,))
            t_dset[current_len:] = t
        
        # --- Append new data for each wind component ---
        # List of variable names with desired units.
        for var, unit in zip(["u", "v", "w"], ["m/s", "m/s", "m/s"]):
            if var not in fp:
                # Create new dataset using the new data slice.
                data = df[var]  # expected shape: (n_time, n_lon, n_lat, n_alt) (usually n_time==1)
                init_shape = data.shape
                max_shape = (None,) + init_shape[1:]
                dset = fp.create_dataset(var, data=data, maxshape=max_shape, chunks=True)
                dset.attrs["units"] = unit
            else:
                dset = fp[var]
                current_len = dset.shape[0]
                new_data = df[var]
                n_new = new_data.shape[0]
                dset.resize((current_len + n_new,) + dset.shape[1:])
                dset[current_len:current_len + n_new, ...] = new_data
        
        # --- Append new data for the standard deviation datasets ---
        for var, unit in zip(["u_std", "v_std", "w_std"], ["m/s", "m/s", "m/s"]):
            if var not in fp:
                data = df[var]  # expected shape: (n_time, n_lon, n_lat, n_alt)
                init_shape = data.shape
                max_shape = (None,) + init_shape[1:]
                dset = fp.create_dataset(var, data=data, maxshape=max_shape, chunks=True)
                dset.attrs["units"] = unit
            else:
                dset = fp[var]
                current_len = dset.shape[0]
                new_data = df[var]
                n_new = new_data.shape[0]
                dset.resize((current_len + n_new,) + dset.shape[1:])
                dset[current_len:current_len + n_new, ...] = new_data

def append_winds_netcdf(df, output_file, overwrite=False):
    """
    Append wind data and associated variables to a NetCDF file.
    If the file does not exist, create it with the proper dimensions,
    static spatial variables, and global attributes.
    If overwrite is True, any existing file will be removed before writing new data.

    Parameters:
        df : dict
            Dictionary containing:
                - "t": Time values (1D array or single value).
                - "lat_3D", "lon_3D", "alt_3D": 3D arrays with static spatial coordinates.
                  The 1D coordinate arrays are extracted as:
                      longitude = lon_3D[:, 0, 0]
                      latitude  = lat_3D[0, :, 0]
                      altitude  = alt_3D[0, 0, :]
                - "mask": Mask array (assumed constant in time; the first slice is used).
                - "version": Version information.
                - Wind components "u", "v", "w" and standard deviations "u_std", "v_std", "w_std"
                  with shape (n_time, n_longitude, n_latitude, n_altitude).
        output_file : str
            Base name for the output file (version and file extension will be appended).
        overwrite : bool, optional
            If True, any existing NetCDF file with the same name will be deleted before writing.

    Returns:
        None
    """

    # Extract data from the input dictionary.
    t = np.atleast_1d(df["t"])  # Ensure time is a 1D array
    lat_3D = df["lat_3D"]
    lon_3D = df["lon_3D"]
    alt_3D = df["alt_3D"]
    mask = df["mask"][0]  # Use the first slice (assumed constant over time)

    # Extract coordinate vectors from 3D arrays.
    longitudes = lon_3D[:, 0, 0]
    latitudes  = lat_3D[0, :, 0]
    altitudes  = alt_3D[0, 0, :]

    # Compute representative datetime for metadata purposes.
    dt = datetime.datetime.utcfromtimestamp(np.mean(t))
    
    # Retrieve version and history information.
    current_version = df["version"]
    history_text = version_history.get_version_history(version_history_file)
    
    # Define the filename (with version information in the name).
    filename = output_file + "_v%s.nc" % current_version

    # Remove the file if overwrite is True.
    if overwrite and os.path.exists(filename):
        os.remove(filename)
    
    # Open file: create it if it does not exist; otherwise, open in append mode.
    if os.path.exists(filename):
        ds = netCDF4.Dataset(filename, "r+")
        # Verify that the "time" dimension is unlimited.
        if not ds.dimensions["time"].isunlimited():
            ds.close()
            raise ValueError("The file exists, but its time dimension is not unlimited. "
                             "Please use overwrite=True to recreate the file with an unlimited time dimension.")
        # Get the current length of the time dimension from the time variable.
        base_length = ds.variables["time"].shape[0] if "time" in ds.variables else 0
    else:
        ds = netCDF4.Dataset(filename, "w", format="NETCDF4")
        
        # --- Create dimensions ---
        ds.createDimension("time", None)  # Unlimited time dimension
        ds.createDimension("longitude", len(longitudes))
        ds.createDimension("latitude",  len(latitudes))
        ds.createDimension("altitude",  len(altitudes))
        
        # --- Global attributes ---
        ds.author = "Miguel Urco Cordero"
        ds.email = "miguelcordero191@gmail.com"
        ds.description = "4D wind data generated with HYPER from meteor radar measurements"
        ds.reference = "Urco et al., 2024; https://doi.org/10.1029/2024JH000162"
        ds.date = dt.strftime("%Y-%m-%d")
        ds.version = current_version
        ds.history = history_text
        
        # --- Create spatial coordinate variables ---
        lon_var = ds.createVariable("longitude", "f4", ("longitude",))
        lon_var[:] = longitudes
        lon_var.units = "Degrees"
        
        lat_var = ds.createVariable("latitude", "f4", ("latitude",))
        lat_var[:] = latitudes
        lat_var.units = "Degrees"
        
        alt_var = ds.createVariable("altitude", "f4", ("altitude",))
        alt_var[:] = altitudes
        alt_var.units = "km"
        
        # Create the mask variable (dimensions: longitude, latitude, altitude)
        mask_var = ds.createVariable("mask", "i1", ("longitude", "latitude", "altitude"))
        mask_var[:] = mask
        
        # For a new file, the base length is initially zero.
        base_length = 0

    n_new = len(t)  # Number of new time steps

    # --- Append or create the time coordinate variable using base_length ---
    if "time" not in ds.variables:
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since 1970-01-01"
        time_var[:] = t
    else:
        time_var = ds.variables["time"]
        time_var[base_length:base_length + n_new] = t

    # Helper function to append or create a variable along the time dimension using base_length.
    def append_variable(var_name, data, units):
        # Expected data shape: (n_time, longitude, latitude, altitude)
        n_new_local = data.shape[0]
        if var_name not in ds.variables:
            var = ds.createVariable(
                var_name, "f4",
                ("time", "longitude", "latitude", "altitude"),
                zlib=True,
            )
            var.units = units
            var[:] = data
        else:
            var = ds.variables[var_name]
            # Use the stored base_length to ensure that the new data are appended at the proper index.
            var[base_length:base_length + n_new_local, ...] = data

    # --- Append new data for wind components ---
    for var_name, unit in zip(["u", "v", "w"], ["m/s", "m/s", "m/s"]):
        append_variable(var_name, df[var_name], unit)
    
    # --- Append new data for standard deviation variables ---
    for var_name, unit in zip(["u_std", "v_std", "w_std"], ["m/s", "m/s", "m/s"]):
        append_variable(var_name, df[var_name], unit)

    ds.close()
    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script to produce 3D wind outputs')
    
    parser.add_argument('-m', '--mpath', dest='mpath', default="/Users/radar/Data/IAP/SIMONe/Germany/SpaceX/hyper_5.256_48", help='Path where the model weights are')
    parser.add_argument('-r', '--rpath', dest='rpath', default=None, help='Path where the wind data will be saved')
    
    parser.add_argument('-g', '--gradients', dest='ena_gradients', default=0, help='Generate gradients too')
    parser.add_argument('-p', '--plotting', dest='ena_plotting', default=1, help='Enable plotting')
    parser.add_argument('--plot-std', dest='plot_std', default=1, help='Plot uncertainties')
    parser.add_argument('-e', '--extension', dest='ext', default='png', help='Figure extension')
    
    parser.add_argument('--output-format', dest='output_format', default='ncdf', help='File format of wind files: either "ncdf" or "hdf5"')
    
    parser.add_argument('--time-step', dest='tstep', type=float, default=15*60, help='Time step in seconds')
    
    # New geographic grid arguments
    parser.add_argument('--lon-step', dest='lon_step', type=float, default=None, help='Longitude step in degrees')
    parser.add_argument('--lat-step', dest='lat_step', type=float, default=None, help='Latitude step in degrees')
    parser.add_argument('--alt-step', dest='alt_step', type=float, default=1.0, help='Altitude step in km')
    
    parser.add_argument('--lon-range', dest='lon_range', type=float, default=None, help='Longitude range in degrees')
    parser.add_argument('--lat-range', dest='lat_range', type=float, default=None, help='Latitude range in degrees')
    parser.add_argument('--alt-range', dest='alt_range', type=float, default=None, help='Altitude range in km')
    
    parser.add_argument('--lat-ref', dest='lat0', type=float, default=None, help='Reference latitude')
    parser.add_argument('--lon-ref', dest='lon0', type=float, default=None, help='Reference longitude')
    parser.add_argument('--alt-ref', dest='alt0', type=float, default=None, help='Reference altitude (km)')
    
    # New optional direct coordinate inputs (as comma-separated lists)
    parser.add_argument('--t-coords', dest='t_coords', default=None)
    parser.add_argument('--lon-coords', dest='lon_coords', default=None, type=str, help='Comma-separated list of longitude coordinates')
    parser.add_argument('--lat-coords', dest='lat_coords', default=None, type=str, help='Comma-separated list of latitude coordinates')
    parser.add_argument('--alt-coords', dest='alt_coords', default=None, type=str, help='Comma-separated list of altitude coordinates')
    
    # parser.add_argument('--t-coords', dest='t_coords', default="1739923200.0, 1739925000.0, 1739926800.0, 1739928600.0, 1739930400.0, 1739932200.0, 1739934000.0, 1739935800.0, 1739937600.0, 1739939400.0, 1739941200.0, 1739943000.0, 1739944800.0, 1739946600.0, 1739948400.0, 1739950200.0, 1739952000.0, 1739953800.0, 1739955600.0, 1739957400.0, 1739959200.0, 1739961000.0, 1739962800.0, 1739964600.0, 1739966400.0, 1739968200.0, 1739970000.0, 1739971800.0, 1739973600.0, 1739975400.0, 1739977200.0, 1739979000.0, 1739980800.0, 1739982600.0, 1739984400.0, 1739986200.0, 1739988000.0, 1739989800.0, 1739991600.0, 1739993400.0, 1739995200.0, 1739997000.0, 1739998800.0, 1740000600.0, 1740002400.0, 1740004200.0, 1740006000.0, 1740007800.0, 1740009600.0",
    #                     type=str, help='Comma-separated list of time coordinates')
    # parser.add_argument('--lon-coords', dest='lon_coords', default="9.00000000e+00,  9.18000000e+00,  9.36000000e+00,  9.54000000e+00, 9.72000000e+00,  9.90000000e+00,  1.00800000e+01,  1.02600000e+01, 1.04400000e+01,  1.06200000e+01,  1.08000000e+01,  1.09800000e+01, 1.11600000e+01,  1.13400000e+01,  1.15200000e+01,  1.17000000e+01, 1.18800000e+01,  1.20600000e+01,  1.22400000e+01,  1.24200000e+01, 1.26000000e+01,  1.27800000e+01,  1.29600000e+01,  1.31400000e+01, 1.33200000e+01,  1.35000000e+01,  1.36800000e+01,  1.38600000e+01, 1.40400000e+01,  1.42200000e+01,  1.44000000e+01,  1.45800000e+01, 1.47600000e+01,  1.49400000e+01,  1.51200000e+01,  1.53000000e+01, 1.54800000e+01,  1.56600000e+01,  1.58400000e+01,  1.60200000e+01",
    #                     type=str, help='Comma-separated list of longitude coordinates')
    # parser.add_argument('--lat-coords', dest='lat_coords', default="51.93, 52.11, 52.29, 52.47, 52.65, 52.83, 53.01, 53.19, 53.37, 53.55, 53.73, 53.91, 54.09, 54.27, 54.45, 54.63, 54.81, 54.99, 55.17, 55.35, 55.53, 55.71, 55.89, 56.07, 56.25, 56.43, 56.61, 56.79",
    #                     type=str, help='Comma-separated list of latitude coordinates')
    # parser.add_argument('--alt-coords', dest='alt_coords', default="100.07407025,  98.67341399,  97.28674999,  95.91398446,94.55503898,  93.20983589,  91.87828907,  90.56032043,89.25584373,  87.96478917,  86.68707801,  85.42264065,84.17139057,  82.93326558,  81.70818754,  80.49608593", 
    #                     type=str, help='Comma-separated list of altitude coordinates')
    
    args = parser.parse_args()
    
    mpath       = args.mpath
    rpath       = args.rpath
    ena_grads   = args.ena_gradients
    ena_plot    = args.ena_plotting
    plot_std    = args.plot_std
    ext         = args.ext
    tstep       = args.tstep
    lon_step    = args.lon_step
    lat_step    = args.lat_step
    alt_step    = args.alt_step
    lon_range   = args.lon_range
    lat_range   = args.lat_range
    alt_range   = args.alt_range
    lat0        = args.lat0
    lon0        = args.lon0
    alt0        = args.alt0
    output_format = args.output_format
    
    # Helper to parse comma-separated lists into arrays of floats
    def parse_coords(coord_str):
        coord = np.sort( [float(item) for item in coord_str.split(',')] ) if coord_str is not None else None
        return coord

    t_coords = parse_coords(args.t_coords)
    lon_coords = parse_coords(args.lon_coords)
    lat_coords = parse_coords(args.lat_coords)
    alt_coords = parse_coords(args.alt_coords)
    
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
        print("No model folders found in %s" % mpath)
    
    grid_4d = Grid4D(tstep=tstep, 
                     lon_step=lon_step,
                     lat_step=lat_step,
                     alt_step=alt_step,
                     lon_range=lon_range,
                     lat_range=lat_range,
                     alt_range=alt_range,
                     lon_ref=lon0,
                     lat_ref=lat0,
                     alt_ref=alt0,
                     t_coords=t_coords,
                     lon_coords=lon_coords,
                     lat_coords=lat_coords,
                     alt_coords=alt_coords,
                     )
    
    for day_folder in day_folders:
        
        day_folder = os.path.join(mpath, day_folder)
        hourly_files = sorted(glob.glob(os.path.join(day_folder, "*_i000*.h5")))
        
        if len(hourly_files) < 1:
            print("No model files found in %s" % day_folder)
            continue
            
        print("Processing experiment %s" % day_folder)
        
        plotData = TimeAltitudePlot(lon0=lon0, lat0=lat0, alt0=alt0)
        
        if plot_std:
            plotStd = TimeAltitudePlot(lon0=lon0, lat0=lat0, alt0=alt0)
        
        print("Generating winds ...")
        for i, hourly_file in enumerate(hourly_files):
            
            coords = grid_4d.update(hourly_file)
            
            if len(coords['t']) < 1:
                continue
            
            dt = datetime.datetime.utcfromtimestamp(np.min(coords['t']))
            
            
            ensemble_files = find_ensemble_files(hourly_file)
            df = winds_from_model(ensemble_files, coords)
            
            print(".", end='', flush=True)
            
            output_file = os.path.join(rpath_, "winds3D_%s" % dt.strftime("%Y%m%d_%H%M%S"))
            
            overwrite = False
            if i == 0: overwrite = True
            
            if output_format == "hdf5":
                append_winds_hdf5(df, output_file, overwrite=overwrite)
            elif output_format == "ncdf":
                append_winds_netcdf(df, output_file, overwrite=overwrite)
            else:
                raise ValueError("File format not supported")
            
            plotData.update_chunk(df)
            
            if plot_std:
                plotStd.update_chunk(df, plot_std=True)
        
        print("Plotting winds ...")
        plotData.save_plot(path=figpath_, suffix=df["version"])
        
        if plot_std:
            
            plotStd.save_plot(path=figpath_,
                              suffix="std_%s" %df["version"],
                              cmap='inferno',
                              vmins=[0,0,0],
                              vmaxs=[30,30,5],
                              )