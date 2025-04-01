import h5py
import numpy as np
import xarray as xr
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from netCDF4 import Dataset, num2date

from scipy.signal import correlate

def read_wind_data(file_path):
    """
    Reads 3D wind data from an HDF5 file and returns an xarray dataset.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        xarray.Dataset: Dataset containing u, v, w wind components along with time and spatial coordinates.
    """
    with h5py.File(file_path, "r") as hdf:
        # Extract time groups (each group name represents seconds since 1970)
        time_groups = sorted(hdf.keys())

        # Read spatial grid (assuming these are stored at the root level)
        lon = hdf["lon"][:] if "lon" in hdf else None  # Longitude (3D)
        lat = hdf["lat"][:] if "lat" in hdf else None  # Latitude (3D)
        alt = hdf["alt"][:] if "alt" in hdf else None  # Altitude (3D)

        # Check if spatial grids exist
        if lon is None or lat is None or alt is None:
            raise ValueError("Missing spatial grid variables (Lon, Lat, Alt) in the file.")

        # Create empty lists to store wind data
        time_list = []
        u_list, v_list, w_list = [], [], []

        # Read wind data for each time step
        for time_str in time_groups:
            
            try:
                time_value = int( float(time_str) )
            except:
                #If the time is not numeric, continue with the next value
                continue
            
            group = hdf[time_str]

            # Read wind components (assuming 3D arrays)
            u = group["u"][:] if "u" in group else None
            v = group["v"][:] if "v" in group else None
            w = group["w"][:] if "w" in group else None

            # Ensure all wind components are present
            if u is None or v is None or w is None:
                raise ValueError(f"Missing wind component in time group {time_str}.")

            u_list.append(u)
            v_list.append(v)
            w_list.append(w)
            
            time_list.append(time_value)
            

        # Convert lists to 4D numpy arrays (time, alt, lat, lon)
        u_data = np.stack(u_list, axis=0)
        v_data = np.stack(v_list, axis=0)
        w_data = np.stack(w_list, axis=0)
        
        epochs = np.stack(time_list, axis=0)
        datetimes = np.array([datetime.datetime.utcfromtimestamp(t) for t in time_list])
        
        # Create xarray dataset
        ds = xr.Dataset(
            {
                "u": (["time", "lon", "lat", "alt"], u_data),
                "v": (["time", "lon", "lat", "alt"], v_data),
                "w": (["time", "lon", "lat", "alt"], w_data),
            },
            coords={
                "time": epochs,  # Time in seconds
                "datetime": ("time", datetimes),  # Time in datetime format
                "alt": (["lon", "lat", "alt"], alt),
                "lat": (["lon", "lat", "alt"], lat),
                "lon": (["lon", "lat", "alt"], lon),
            },
        )
        
    return ds

def read_icon2_file(filename):
    
    with Dataset(filename, mode='r') as nc:
        
        time_values = nc.variables['time'][:]
        time_units = nc.variables['time'].units
        time_dt = num2date(time_values, units=time_units).data
        
        epoch = datetime.datetime(1970, 1, 1)
        time_seconds = [t.total_seconds() for t in (time_dt - epoch)]
        
        alt = nc.variables['altitude'][:] / 1000  # Remember to convert altitude from meters to kilometers
        lat = nc.variables['latitude'][:] 
        lon = nc.variables['longitude'][:] 
        
        u = nc.variables['u'][:]  # Zonal wind
        v = nc.variables['v'][:]  # Meridional wind
    
    # Create xarray dataset
    ds = xr.Dataset(
        {
            "u": (["time", "lon", "lat", "alt"], u),
            "v": (["time", "lon", "lat", "alt"], v),
            "w": (["time", "lon", "lat", "alt"], u*0),
        },
        coords={
            "time": time_seconds,  # Time in seconds
            "datetime": ("time", time_dt),  # Time in datetime format
            "alt": (["lon", "lat", "alt"], alt),
            "lat": (["lon", "lat", "alt"], lat),
            "lon": (["lon", "lat", "alt"], lon),
        },
    )
    
    return ds

def read_icon_file(filename):
    
    with Dataset(filename, mode='r') as nc:
        
        time_values = nc.variables['time'][:]
        time_units = nc.variables['time'].units
        time_dt = num2date(time_values, units=time_units).data
        
        epoch = datetime.datetime(1970, 1, 1)
        time_seconds = [t.total_seconds() for t in (time_dt - epoch)]
        
        alt = nc.variables['z_ifc'][:] / 1000  # Remember to convert altitude from meters to kilometers
        lat = nc.variables['lat'][:] 
        lon = nc.variables['lon'][:] 
        
        u = nc.variables['u'][:]  # Zonal wind
        v = nc.variables['v'][:]  # Meridional wind
    
    ALT, LAT, LON = np.meshgrid(alt, lat, lon, indexing="ij")
    
    # Create xarray dataset
    ds = xr.Dataset(
        {
            "u": (["time", "alt", "lat", "lon"], u),
            "v": (["time", "alt", "lat", "lon"], v),
            "w": (["time", "alt", "lat", "lon"], u*0),
        },
        coords={
            "time": time_seconds,  # Time in seconds
            "datetime": ("time", time_dt),  # Time in datetime format
            "alt": (["alt", "lat", "lon"], ALT),
            "lat": (["alt", "lat", "lon"], LAT),
            "lon": (["alt", "lat", "lon"], LON),
        },
    )
    
    return ds

def plot_horizontal_wind(ds, time_dt, altitudes_km, dec=1):
    """
    Plots horizontal wind vectors (u and v) for multiple altitude levels at a specified datetime.
    
    Parameters:
        ds (xarray.Dataset): Dataset containing wind components with dimensions
                             ("time", "lon", "lat", "alt") and corresponding coordinates.
        time_dt (datetime.datetime): Target time as a datetime object.
        altitudes_km (list of float): List of target altitudes in kilometers.
        
    Returns:
        None: Displays a figure with multiple subplots (one per altitude) showing wind vectors.
        
    Notes:
        - It is assumed that the dataset contains a 'datetime' coordinate for time-based selection.
        - The altitude coordinate is assumed to be constant across longitude and latitude. The function
          uses the altitude values from the first longitude and latitude indices to compute the nearest index.
        - If your dataset stores altitude in meters, convert altitudes_km to meters (i.e., altitude_km * 1000)
          before calling the function.
    """
    # Select the nearest time step using the 'datetime' coordinate.
    epoch = datetime.datetime(1970, 1, 1)
    time_seconds = (time_dt - epoch).total_seconds()
    
    # Select the nearest time step using the 'time' coordinate.
    ds_time = ds.sel(time=time_seconds, method="nearest")
    
    # Extract the altitude values along the alt dimension (assumed constant across lon and lat).
    alt_values = ds_time['alt'].isel(lon=0, lat=0).values  # 1D array of altitude values
    
    # Determine the number of altitudes to plot.
    n_alt = len(altitudes_km)
    
    # Create subplots: one per requested altitude.
    if n_alt == 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_alt, figsize=(4 * n_alt, 5))
    
    # Ensure axes is iterable.
    if n_alt == 1:
        axes = [axes]
    
    plt.suptitle('Horizontal winds: %s' %time_dt.strftime('%Y-%m-%d %H:%M:%S') )
    
    # Iterate over each target altitude and plot the corresponding wind vectors.
    for idx, target_alt in enumerate(altitudes_km):
        # Find the altitude index closest to the target altitude.
        alt_index = np.argmin(np.abs(alt_values - target_alt))
        
        # Extract u and v horizontal wind components and the corresponding longitude and latitude.
        u = ds_time['u'].isel(alt=alt_index)[::dec, ::dec]
        v = ds_time['v'].isel(alt=alt_index)[::dec, ::dec]
        lon = ds_time['lon'].isel(alt=alt_index)[::dec, ::dec]
        lat = ds_time['lat'].isel(alt=alt_index)[::dec, ::dec]
        
        # Create the quiver plot on the corresponding subplot.
        ax = axes[idx]
        ax.quiver(lon, lat, u, v, scale=1000)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title( f'at {alt_values[alt_index]:.2f} km')
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        textstr = '\n'.join((
            r'$u_0=%2.1f$m/s' %np.nanmean(u),
            r'$v_0=%2.1f$m/s' %np.nanmean(v),
            ))
        
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

    
    plt.tight_layout()
    # plt.show()

def plot_mean_winds(*datasets, cmap="RdBu", vmin=-100, vmax=100):
    
    nplots = len(datasets)
    
    fig, axes = plt.subplots(nplots, 2, figsize=(12, nplots*3), sharex=True, sharey=True)
    
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    
    for i in range(nplots):
        
        ds = datasets[i]
        
        time = ds.time.values
        alt = ds['alt'].isel(lon=0, lat=0).values
        mean = ds.mean(dim=["lon","lat"])
        
        # Convert time to matplotlib date format
        time_datetime = [datetime.datetime.utcfromtimestamp(t) for t in time]
        time_num = mdates.date2num(time_datetime)  # Convert to numeric format for plotting


        u = mean.u.values
        v = mean.v.values
        
        ax = axes[i,0]
        im = ax.pcolormesh(time_num, alt, u.T, cmap=cmap, vmin=vmin, vmax=vmax)#, shading='gouraud')
        
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        ax.set_title("Zonal velocity")
        ax.set_ylabel('Altitude (km)')
        ax.set_xlabel('Universal time')
        
        ax = axes[i,1]
        im =ax.pcolormesh(time_num, alt, v.T, cmap=cmap, vmin=vmin, vmax=vmax)#, shading='gouraud')
        
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        ax.set_title("Meridional velocity")
        ax.set_ylabel('Altitude (km)')
        ax.set_xlabel('Universal time')
    
    plt.tight_layout()
    
    fig.subplots_adjust(left=0.05, right=0.93, bottom=0.1, top=0.95)  # Make room for colorbar

    # Add colorbar in a separate axis
    cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, pad=0.005, label="m/s")
    
    plt.show()

def estimate_time_delay(data1, data2, time_resolution=1.0):
    """
    Estimates the time delay between two 2D temperature datasets (time x altitude).
    
    Parameters:
      data1, data2 : 2D numpy arrays
          Temperature data from instrument 1 and instrument 2.
      time_resolution : float
          Time interval between consecutive measurements.
    
    Returns:
      delay_steps : int
          Estimated time delay in time steps.
      delay_time  : float
          Estimated time delay in the same units as time_resolution.
    """
    # Average over altitude to get a 1D time series for each dataset
    signal1 = np.mean(data1, axis=1)
    signal2 = np.mean(data2, axis=1)
    
    # Compute full cross-correlation between the two signals
    corr = correlate(signal1, signal2, mode='full')
    
    # Create an array of lag indices
    lags = np.arange(-len(signal1) + 1, len(signal1))
    
    # Identify the lag that maximizes the cross-correlation
    delay_steps = lags[np.argmax(corr)]
    delay_time = delay_steps * time_resolution
    
    # Plot the cross-correlation for visualization
    plt.figure(figsize=(8, 4))
    plt.plot(lags*time_resolution/60, corr, marker='o')
    plt.xlabel('Lag (minutes)')
    plt.ylabel('Cross-correlation')
    plt.title('Mean wind cross-correlation between HYPER and ICON')
    plt.grid(True)
    plt.show()
    
    return delay_steps, delay_time

def estimate_time_delay_by_altitude(data1, data2, time_resolution=1.0):
    """
    Estimates the time delay between two 2D temperature datasets by computing
    the cross-correlation at each altitude level and averaging the results.
    
    Parameters:
      data1, data2 : 2D numpy arrays (time x altitude)
      time_resolution : float
          The time interval between consecutive measurements.
          
    Returns:
      delay_steps : int
          Estimated delay in time steps.
      delay_time  : float
          Estimated delay in same units as time_resolution.
    """
    T, A = data1.shape
    lags = np.arange(-(T - 1), T)
    
    # Accumulate the cross-correlation from each altitude level
    corr_sum = np.zeros(len(lags))
    
    for a in range(A):
        signal1 = data1[:, a]
        signal2 = data2[:, a]
        # Compute cross-correlation for the altitude level
        corr = correlate(signal1, signal2, mode='full')
        corr_sum += corr
        
    # Average over altitudes
    corr_mean = corr_sum / A
    
    # Find the lag corresponding to the maximum average correlation
    delay_steps = lags[np.argmax(corr_mean)]
    delay_time = delay_steps * time_resolution
    
    # Plot the averaged cross-correlation function
    plt.figure(figsize=(8, 4))
    plt.plot(lags, corr_mean, marker='o')
    plt.xlabel("Time lag (steps)")
    plt.ylabel("Average cross-correlation")
    plt.title("Average Cross-correlation Across Altitudes")
    plt.grid(True)
    plt.show()
    
    return delay_steps, delay_time

if __name__=="__main__":

    # Load the wind dataset
    file_path = "/Users/mcordero/Data/IAP/SIMONe/Germany/SpaceX/hyper24/winds/winds3D_20250219.h5"
    # Select the desired time step (e.g., the first time index)
    time_dt = datetime.datetime(2025,2,19,19,0,0)
    # Select the altitude index for the cut (modify as needed)
    altitudes_km = [90, 95, 100]
    
    ds0 = read_wind_data(file_path)
    ds0 = ds0.where( (ds0.lon>10) & (ds0.lon<15) & (ds0.lat>52.5) & (ds0.lat<55.5), drop=True)
    
    plot_horizontal_wind(ds0, time_dt, altitudes_km)
    
    # file1 = "/Users/mcordero/Data/IAP/SIMONe/Germany/SpaceX/hyper24/winds/longitudes"
    # file2 = "/Users/mcordero/Data/IAP/SIMONe/Germany/SpaceX/hyper24/winds/latitudes"
    # file3 = "/Users/mcordero/Data/IAP/SIMONe/Germany/SpaceX/hyper24/winds/altitudes"
    #
    # np.save(file1, ds['lon'].values)
    # np.save(file2, ds['lat'].values)
    # np.save(file3, ds['alt'].values)
    
    
    file_path = "/Users/mcordero/Data/IAP/SIMONe/Germany/SpaceX/ICON/UA-ICON_SIMONe_falcon_cb_20250219-20.nc"
    file_path = "/Users/mcordero/Data/IAP/SIMONe/Germany/SpaceX/ICON/UA-ICON_NWP_atm_DOM01_80-100km_20250219-20.nc"
    
    ds = read_icon_file(file_path)
    ds = ds.where( (ds.lon>ds0.lon.min()) & (ds.lon<ds0.lon.max()) & (ds.lat>ds0.lat.min()) & (ds.lat<ds0.lat.max()), drop=True)
    
    # plot_horizontal_wind(ds, time_dt, altitudes_km, dec=1)
    
    # plot_mean_winds( ds0, ds )

    ds0 = ds0.where( (ds0.alt>92) & (ds0.alt<97), drop=True)
    ds = ds.where( (ds.alt>92) & (ds.alt<97), drop=True)
    
    mean0 = ds0.mean(dim=["lon","lat"])
    mean1 = ds.mean(dim=["lon","lat"])
    
    estimate_time_delay(mean0.v.values[::6], mean1.v.values[1:-1], time_resolution=1800)
