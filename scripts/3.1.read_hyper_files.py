import os
import numpy as np
import xarray as xr
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import h5py
import netCDF4

from scipy.signal import correlate, correlate2d

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
                "lon": lon,
                "lat": lat,
                "alt": alt,
            },
        )
        
    return ds

def read_winds_hdf5(input_file):
    """
    Read wind data from an HDF5 file and create an xarray.Dataset.
    
    Parameters
    ----------
    input_file : str
        Path to the HDF5 file containing the wind data.
    
    Returns
    -------
    xr.Dataset
        An xarray Dataset with the following structure:
        
        - Data variables:
            "u": Wind component with dimensions (time, lon, lat, alt)
            "v": Wind component with dimensions (time, lon, lat, alt)
            "w": Wind component with dimensions (time, lon, lat, alt)
        
        - Coordinates:
            "time": Time in seconds since the epoch.
            "datetime": Time converted to numpy.datetime64 format.
            "lon": Spatial coordinate with dimensions (lon, lat, alt)
            "lat": Spatial coordinate with dimensions (lon, lat, alt)
            "alt": Spatial coordinate with dimensions (lon, lat, alt)
    
    Notes
    -----
    The HDF5 file is expected to have the following datasets:
      - "times": 1D array of time values (seconds since 1970/01/01)
      - "u", "v", "w": Wind component data arrays with shape (time, lon, lat, alt)
      - "lon", "lat", "alt": Coordinate arrays with shape (lon, lat, alt)
      
    References
    ----------
    - xarray documentation: https://xarray.pydata.org/en/stable/
    - h5py documentation: https://www.h5py.org/
    """
    
    with h5py.File(input_file, "r") as fp:
        # Read the time data as an array of epochs (seconds)
        times = fp["times"][:]
        
        # Convert epochs (seconds since 1970-01-01) to datetime64 objects (1-second resolution)
        datetimes = np.array(times, dtype='datetime64[s]')
        
        # Read wind component data
        u_data = fp["u"][:]
        v_data = fp["v"][:]
        w_data = fp["w"][:]
        
        # Read spatial coordinate data
        lon = fp["lon"][:]
        lat = fp["lat"][:]
        alt = fp["alt"][:]
    
    # Create the xarray.Dataset with the specified data variables and coordinates
    ds = xr.Dataset(
        data_vars={
            "u": (["time", "lon", "lat", "alt"], u_data),
            "v": (["time", "lon", "lat", "alt"], v_data),
            "w": (["time", "lon", "lat", "alt"], w_data),
        },
        coords={
            "time": times,
            "datetime": ("time", datetimes),
            "lon": lon,
            "lat": lat,
            "alt": alt,
        },
        attrs={
            "Description": "4D wind data loaded from HDF5 file",
            "Reference": "Urco et al., 2024; https://doi.org/10.1029/2024JH000162"
        }
    )
    
    return ds

def read_winds_netcdf(input_file):
    """
    Read wind data from a NetCDF file and create an xarray.Dataset.
    
    Parameters
    ----------
    input_file : str
        Path to the NetCDF file containing the wind data.
    
    Returns
    -------
    xr.Dataset
        An xarray Dataset with the following structure:
        
        - Data variables:
            "u": Wind component with dimensions (time, lon, lat, alt)
            "v": Wind component with dimensions (time, lon, lat, alt)
            "w": Wind component with dimensions (time, lon, lat, alt)
        
        - Coordinates:
            "time": Time in seconds since 1970-01-01.
            "datetime": Time converted to numpy.datetime64 format.
            "lon": 3D array of longitude values.
            "lat": 3D array of latitude values.
            "alt": 3D array of altitude values.
    
    Notes
    -----
    The NetCDF file is expected to contain:
      - A "time" variable (1D array) with units "seconds since 1970-01-01".
      - Spatial coordinate variables "longitude", "latitude", and "altitude"
        (each a 1D array). These are converted into 3D arrays via meshgrid.
      - Wind component variables "u", "v", and "w", each with dimensions 
        (time, longitude, latitude, altitude).
    
    References
    ----------
    - xarray documentation: https://xarray.pydata.org/en/stable/
    - netCDF4 documentation: https://unidata.github.io/netcdf4-python/
    """
    # Open the NetCDF file for reading.
    # nc = netCDF4.Dataset(input_file, "r")
    with netCDF4.Dataset(input_file, mode='r') as nc:
        
        # Read the time variable and convert it to datetime64 format.
        times = nc.variables["time"][:]
        datetimes = np.array(times, dtype="datetime64[s]")
        
        # Read wind component data.
        u_data = nc.variables["u"][:]
        v_data = nc.variables["v"][:]
        w_data = nc.variables["w"][:]
        
        # Read the spatial coordinate variables.
        lon_1d = nc.variables["longitude"][:]
        lat_1d = nc.variables["latitude"][:]
        alt_1d = nc.variables["altitude"][:]
    
    # nc.close()
    
    # Create 3D coordinate arrays using meshgrid.
    # The ordering 'ij' ensures that longitude, latitude, and altitude dimensions 
    # are mapped to the first, second, and third axes, respectively.
    # lon_3d, lat_3d, alt_3d = np.meshgrid(lon_1d, lat_1d, alt_1d, indexing="ij")
    
    # Construct the xarray.Dataset.
    ds = xr.Dataset(
        data_vars={
            "u": (["time", "lon", "lat", "alt"], u_data),
            "v": (["time", "lon", "lat", "alt"], v_data),
            "w": (["time", "lon", "lat", "alt"], w_data),
        },
        coords={
            "time": times,
            "datetime": ("time", datetimes),
            "lon": lon_1d,
            "lat": lat_1d,
            "alt": alt_1d,
        },
        attrs={
            "Description": "4D wind data loaded from NetCDF file",
        }
    )
    
    return ds

def read_icon2_file(filename):
    
    with netCDF4.Dataset(filename, mode='r') as nc:
        
        time_values = nc.variables['time'][:]
        time_units = nc.variables['time'].units
        time_dt = netCDF4.num2date(time_values, units=time_units).data
        
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
            "lon": lon,
            "lat": lat,
            "alt": alt,
        },
    )
    
    return ds

def read_icon_file(filename):
    
    with netCDF4.Dataset(filename, mode='r') as nc:
        
        time_values = nc.variables['time'][:]
        time_units = nc.variables['time'].units
        time_dt = netCDF4.num2date(time_values, units=time_units).data
        
        epoch = datetime.datetime(1970, 1, 1)
        time_seconds = [t.total_seconds() for t in (time_dt - epoch)]
        
        try:
            alt = nc.variables['z_ifc'][:] / 1000  # Remember to convert altitude from meters to kilometers
        except:
            alt = nc.variables['z_mc'][:] / 1000
        
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
            "lon": lon,
            "lat": lat,
            "alt": alt,
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
    alt_values = ds_time['alt'].values#.isel(lon=0, lat=0).values  # 1D array of altitude values
    
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
    
    lon = ds_time['lon'].values[::dec]#.isel(alt=alt_index)[::dec, ::dec]
    lat = ds_time['lat'].values[::dec]#.isel(alt=alt_index)[::dec, ::dec]
    
    lon3D, lat3D = np.meshgrid(lon, lat, indexing="ij")
    
    # Iterate over each target altitude and plot the corresponding wind vectors.
    for idx, target_alt in enumerate(altitudes_km):
        # Find the altitude index closest to the target altitude.
        alt_index = np.argmin(np.abs(alt_values - target_alt))
        
        # Extract u and v horizontal wind components and the corresponding longitude and latitude.
        u = ds_time['u'].isel(alt=alt_index)[::dec, ::dec]
        v = ds_time['v'].isel(alt=alt_index)[::dec, ::dec]
        
        # Create the quiver plot on the corresponding subplot.
        ax = axes[idx]
        ax.quiver(lon3D, lat3D, u, v, scale=1000)
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
    plt.show()

def plot_mean_winds(*datasets, cmap="seismic", vmin=-100, vmax=100, version="", rpath=None):
    
    nplots = len(datasets)
    
    fig, axes = plt.subplots(nplots, 2, figsize=(12, nplots*3), sharex=True, sharey=True)
    
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    
    titles = ["HYPER", "ICON (%s)" %version]
    
    for i in range(nplots):
        
        ds = datasets[i]
        
        time = ds.time.values
        alt = ds['alt'].values#.isel(lon=0, lat=0).values
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
        
        ax.set_title("%s: Zonal velocity" %titles[i])
        ax.set_ylabel('Altitude (km)')
        ax.set_xlabel('Universal time')
        
        ax = axes[i,1]
        im =ax.pcolormesh(time_num, alt, v.T, cmap=cmap, vmin=vmin, vmax=vmax)#, shading='gouraud')
        
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        ax.set_title("%s: Meridional velocity" %titles[i])
        ax.set_ylabel('Altitude (km)')
        ax.set_xlabel('Universal time')
    
    plt.tight_layout()
    
    fig.subplots_adjust(left=0.05, right=0.93, bottom=0.1, top=0.95)  # Make room for colorbar

    # Add colorbar in a separate axis
    cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, pad=0.005, label="m/s")
    
    if rpath is None:
        plt.show()
    else:
        filename = os.path.join(rpath, "winds_%s.pdf" %version)
        plt.savefig(filename)
        plt.close()

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

def compute_2d_corr_for_lag(data1, data2, lag):
    """
    Compute the normalized 2D cross-correlation between two 2D arrays 
    (with dimensions [times, altitudes]) for a specified time lag using 
    scipy.signal.correlate2d (without flattening the arrays).
    
    For a positive lag, data2 is shifted forward relative to data1.  
    The correlation is computed over the overlapping region.
    
    Parameters
    ----------
    data1 : np.ndarray
        2D array of shape (times, altitudes).
    data2 : np.ndarray
        2D array of shape (times, altitudes).
    lag : int
        The time lag (in time steps) to apply.
    
    Returns
    -------
    float
        Normalized correlation coefficient.
    """
    # Select the overlapping region according to the lag.
    if lag > 0:
        s1 = data1[lag:, :, :]
        s2 = data2[:-lag, :, :]
    elif lag < 0:
        s1 = data1[:lag, :, :]
        s2 = data2[-lag:, :, :]
    else:
        s1 = data1
        s2 = data2

    # Subtract the mean from each array (demean)
    s1_mean = np.mean(s1)
    s2_mean = np.mean(s2)
    s1_demean = s1 #- s1_mean
    s2_demean = s2 #- s2_mean

    # Compute the unnormalized 2D cross-correlation over the overlapping region.
    # Since s1 and s2 have the same shape, using mode='valid' returns a single value.
    corr_matrix = correlate2d(s1_demean, s2_demean, mode='valid')
    numerator = corr_matrix[0, 0]

    # Compute normalization factors (without explicitly flattening the arrays).
    norm1 = np.sqrt(np.sum(s1_demean * s1_demean))
    norm2 = np.sqrt(np.sum(s2_demean * s2_demean))
    norm_factor = norm1 * norm2

    if norm_factor == 0:
        return np.nan
    return numerator / norm_factor

def plot_2d_cross_correlation(data1, data2, max_lag=10):
    """
    Compute and plot the normalized 2D cross-correlation (using scipy.signal.correlate2d)
    as a function of time lag, and print the time lag corresponding to the maximum correlation.
    
    Parameters
    ----------
    data1 : np.ndarray
        First 2D data array with shape (times, altitudes).
    data2 : np.ndarray
        Second 2D data array with shape (times, altitudes).
    max_lag : int, optional
        Maximum time lag (in time steps) to consider (default is 10).
    
    Returns
    -------
    tuple
        A tuple (corr_values, lags) where:
          - corr_values is a 1D array of correlation coefficients for each lag.
          - lags is an array of lag values.
    """
    
    lags = np.arange(-max_lag, max_lag + 1)
    corr_values = np.empty(len(lags))
    
    for i, lag in enumerate(lags):
        corr_values[i] = compute_2d_corr_for_lag(data1, data2, lag)
    
    # Plotting the correlation versus time lag
    plt.figure(figsize=(8, 6))
    plt.plot(lags, corr_values, marker='o')
    plt.xlabel("Time lag (time steps)")
    plt.ylabel("Normalized 2D cross-correlation")
    plt.title("2D Normalized Cross-Correlation vs Time Lag")
    plt.grid(True)
    plt.show()
    
    # Identify the best lag (maximum correlation)
    best_index = np.nanargmax(corr_values)
    best_lag = lags[best_index]
    print("Estimated time shift (lag):", best_lag)
    
    return corr_values, lags

def compute_vector_corr_for_lag(data1, data2, lag):
    """
    Compute the normalized vector correlation between two 2D vector fields 
    (shape: [time, altitude, 2]) for a given time lag.
    
    The vector correlation is computed as:
    
        ρ = [ Σ ( (X - X̄) · (Y - Ȳ) ) ] / [ sqrt( Σ ||X - X̄||² * Σ ||Y - Ȳ||² ) ]
    
    where the summations extend over the overlapping region (time and altitude).
    
    Parameters
    ----------
    data1 : np.ndarray
        First vector field of shape (time, altitude, 2), e.g. [u, v].
    data2 : np.ndarray
        Second vector field of shape (time, altitude, 2).
    lag : int
        The time lag (in time steps) to apply. A positive lag means data2 is shifted 
        forward relative to data1.
    
    Returns
    -------
    float
        The vector correlation coefficient (normalized), or NaN if undefined.
    """
    
    # Get original shapes
    nt1, nh1, _ = data1.shape
    nt2, nh2, _ = data2.shape

    # Overlapping altitude: take minimum number of altitude levels.
    n_overlap_h = min(nh1, nh2)

    # Determine time indices for the overlapping region.
    if lag >= 0:
        # When lag is positive:
        # data1: from index lag to end
        # data2: from index 0 to end-lag
        t1_start = lag
        t1_end = nt1
        t2_start = 0
        t2_end = nt2 #- lag
    else:
        # When lag is negative (lag < 0):
        # data1: from index 0 to end+lag
        # data2: from index -lag to end
        t1_start = 0
        t1_end = nt1 + lag  # note: lag is negative so this subtracts from nt1.
        t2_start = -lag
        t2_end = nt2

    # Compute the number of overlapping time steps
    n_overlap_t = min(t1_end - t1_start, t2_end - t2_start)
    if n_overlap_t <= 0:
        print("No overlapping time region given the lag and array sizes.")
        return np.NaN, np.NaN

    # Extract overlapping region for both arrays.
    s1 = data1[t1_start: t1_start + n_overlap_t, :n_overlap_h, :]
    s2 = data2[t2_start: t2_start + n_overlap_t, :n_overlap_h, :]

    diff = s1 - s2
    # Compute dispersion as the root mean square difference over time, altitude, and vector components.
    dispersion = np.sqrt(np.mean(diff**2))
    
    # Compute numerator: sum of the dot products over all overlapping points.
    numerator = np.sum(s1 * s2)  # sums over all axes
    
    # Compute denominator: product of the vector norms for the two demeaned fields.
    norm1 = np.sqrt(np.sum(s1 ** 2))
    norm2 = np.sqrt(np.sum(s2 ** 2))
    
    if norm1 * norm2 == 0:
        return np.nan
    
    return numerator / (norm1 * norm2), dispersion


# def compute_vector_dispersion_for_lag(data1, data2, lag):
#     """
#     Compute the dispersion between two 2D vector fields for a given time lag.
#
#     Here, the dispersion is defined as the root-mean-square (RMS) difference 
#     (Euclidean norm of the difference) over the overlapping region (time and altitude).
#
#     Parameters
#     ----------
#     data1 : np.ndarray
#         First vector field with shape (time, altitude, 2).
#     data2 : np.ndarray
#         Second vector field with shape (time, altitude, 2).
#     lag : int
#         Time lag to apply (positive lag shifts data2 forward relative to data1).
#
#     Returns
#     -------
#     float
#         The RMS difference (dispersion) between the overlapping regions of data1 and data2.
#     """
#     if lag > 0:
#         s1 = data1[:-lag, :, :]
#         s2 = data2[lag:, :, :]
#     elif lag < 0:
#         s1 = data1[-lag:, :, :]
#         s2 = data2[:lag, :, :]
#     else:
#         s1 = data1
#         s2 = data2
#
#     diff = s1 - s2
#     # Compute dispersion as the root mean square difference over time, altitude, and vector components.
#     dispersion = np.sqrt(np.mean(diff**2))
#
#     return dispersion

def plot_vector_corr_vs_lag(ds1, ds2, max_lag=10, time_resolution=1.0):
    """
    Compute and plot the normalized vector correlation between two 2D vector fields 
    across a range of time lags.
    
    Parameters
    ----------
    data1 : np.ndarray
        First vector field of shape (time, altitude, 2), e.g. [u, v].
    data2 : np.ndarray
        Second vector field of shape (time, altitude, 2).
    max_lag : int, optional
        Maximum lag (in time steps) to consider in both directions (default is 10).
    
    Returns
    -------
    tuple
        (corr_values, lags), where corr_values is a 1D array of correlation coefficients 
        for each lag and lags is the corresponding array of lag values.
    """
    
    
    data1 = np.stack((ds1.u.values, ds1.v.values), axis=-1)
    data2 = np.stack((ds2.u.values, ds2.v.values), axis=-1)
    
    lags = np.arange(-max_lag, max_lag + 1)
    corr_values = np.empty(len(lags))
    
    for i, lag in enumerate(lags):
        corr_values[i], _ = compute_vector_corr_for_lag(data1, data2, lag)
    
    time_lags = lags*time_resolution
    
    # Plot the correlation versus time lag.
    plt.figure(figsize=(8, 6))
    plt.plot(lags, corr_values, marker='o')
    plt.xlabel("Time lag (min)")
    plt.ylabel("Normalized vector correlation")
    plt.title("Vector Correlation vs. Time Lag")
    plt.grid(True)
    plt.show()
    
    # Determine and print the lag with maximum correlation.
    best_index = np.nanargmax(corr_values)
    best_lag = lags[best_index]
    print("Estimated time shift (lag):", best_lag)
    
    return corr_values, lags

def plot_vector_corr_and_dispersion_vs_lag(ds1, ds2, max_lag=10, version="", rpath=None):
    """
    Compute and plot the normalized vector correlation and dispersion between two 
    2D vector fields (shape: (time, altitude, 2)) as a function of time lag.
    
    A dual-axis plot is created where the left y-axis shows the normalized vector 
    correlation and the right y-axis shows the RMS dispersion.
    
    Parameters
    ----------
    data1 : np.ndarray
        First vector field with shape (time, altitude, 2).
    data2 : np.ndarray
        Second vector field with shape (time, altitude, 2).
    max_lag : int, optional
        Maximum time lag (in time steps) to consider (default is 10).
    
    Returns
    -------
    tuple
        A tuple (lags, corr_values, dispersion_values) where:
         - lags: array of time lags considered.
         - corr_values: array of normalized vector correlation coefficients.
         - dispersion_values: array of RMS dispersion values.
    """
    
    time1 = ds1.time.values
    time2 = ds2.time.values
    
    time_resolution  = np.mean( np.diff(time1) )
    time_resolution2 = np.mean( np.diff(time1) )
    
    if time_resolution != time_resolution2:
        raise ValueError("The datasets must have the same time resolution")
    
    time_shift = time2[0] - time1[0]
    lag_shift = int(time_shift/time_resolution)
    
    data1 = np.stack((ds1.u.values, ds1.v.values), axis=-1)
    data2 = np.stack((ds2.u.values, ds2.v.values), axis=-1)
    
    lags = np.arange(-max_lag, max_lag + 1) + lag_shift
    time_lags = (lags*time_resolution - time_shift)/60
    
    corr_values = np.empty(len(lags))
    dispersion_values = np.empty(len(lags))
    
    for i, lag in enumerate(lags):
        corr_values[i], dispersion_values[i] = compute_vector_corr_for_lag(data1, data2, lag)
        # dispersion_values[i] = compute_vector_dispersion_for_lag(data1, data2, lag)
    
    # Create a dual-axis plot.
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    color1 = 'tab:blue'
    ax1.set_xlabel("Time lag (min)")
    ax1.set_ylabel("Normalized vector correlation", color=color1)
    ax1.plot(time_lags, corr_values, marker='o', color=color1, label="Correlation")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0,0.5)
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel("root-mean-square deviation (RMSD)", color=color2)
    ax2.plot(time_lags, dispersion_values, marker='x', linestyle='--', color=color2, label="RMSD")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(35,55)
    # ax2.grid(True)
    
    plt.title("Correlation HYPER vs ICON (%s)" %version)
    fig.tight_layout()
    
    if rpath is None:
        plt.show()
    else:
        filename = os.path.join(rpath, "corr_%s.pdf" %version)
        plt.savefig(filename)
        plt.close()
    
    # Identify the time lag with the maximum correlation.
    best_index = np.nanargmax(corr_values)
    best_lag = time_lags[best_index]
    
    print("Estimated time shift (best correlation lag):", best_lag)
    print("Dispersion at best correlation lag:", dispersion_values[best_index])
    
    return lags, corr_values, dispersion_values

if __name__=="__main__":

    # Load the wind dataset
    file_path = "/Users/radar/Data/IAP/SIMONe/Germany/SpaceX/hyper24/winds/winds3D_20250219.h5"
    file_path = "/Users/radar/Data/IAP/SIMONe/Germany/SpaceX/hyper72/winds/winds3D_20250218_003000_v1.1.0.nc"
    
    rpath = os.path.split(file_path)[0]
    
    version = "old"
    # Select the desired time step (e.g., the first time index)
    time_dt = datetime.datetime(2025,2,19,19,0,0)
    # Select the altitude index for the cut (modify as needed)
    altitudes_km = [90, 95, 100]
    
    _, ext = os.path.splitext(file_path)
    
    if ext == ".h5":
        ds0 = read_wind_data(file_path)
    elif ext == ".nc":
        ds0 = read_winds_netcdf(file_path)
    else:
        raise ValueError("File format not supported")
    
    
    ds0 = ds0.where( (ds0.lon>=10) & (ds0.lon<=15) & (ds0.lat>=52.5) & (ds0.lat<=55.5), drop=True)
    
    # plot_horizontal_wind(ds0, time_dt, altitudes_km)
    
    # file1 = "/Users/mcordero/Data/IAP/SIMONe/Germany/SpaceX/hyper24/winds/longitudes"
    # file2 = "/Users/mcordero/Data/IAP/SIMONe/Germany/SpaceX/hyper24/winds/latitudes"
    # file3 = "/Users/mcordero/Data/IAP/SIMONe/Germany/SpaceX/hyper24/winds/altitudes"
    #
    # np.save(file1, ds['lon'].values)
    # np.save(file2, ds['lat'].values)
    # np.save(file3, ds['alt'].values)
    
    if version == "new":
        file_path = "/Users/radar/Data/IAP/SIMONe/Germany/SpaceX/ICON/UA-ICON_NWP_atm_ML_DOM01_falcon2_20250219-20.nc"
    else:
        file_path = "/Users/radar/Data/IAP/SIMONe/Germany/SpaceX/ICON/UA-ICON_NWP_atm_DOM01_80-100km_20250219-20.nc"
    
    ds = read_icon_file(file_path)
    
    ds = ds.where( (ds.time>= (ds0.time.min()) ) & (ds.time<=ds0.time.max()), drop=True)
                   
    ds = ds.where( (ds.lon>=ds0.lon.min()) & (ds.lon<=ds0.lon.max()) & (ds.lat>=ds0.lat.min()) & (ds.lat<=ds0.lat.max()), drop=True)
    
    # plot_horizontal_wind(ds, time_dt, altitudes_km, dec=1)

    ds0 = ds0.where( (ds0.alt>=89) & (ds0.alt<=101), drop=True)
    ds = ds.where( ( ds.alt>=ds0.alt.min()-0.01) & (ds.alt<=ds0.alt.max()+0.01), drop=True)
    
    plot_mean_winds( ds0, ds, version=version, rpath=rpath)
    
    mean0 = ds0.mean(dim=["lon","lat"])
    mean1 = ds.mean(dim=["lon","lat"])
    
    time = mean0.time
    delta_time = np.mean(np.diff(time))
    
    # estimate_time_delay(mean0.v.values, mean1.v.values, time_resolution=delta_time)
    #
    # corr_values, lags = plot_2d_cross_correlation(mean0.v.values, mean1.v.values, max_lag=30)
    #
    # corr_values, lags = plot_vector_corr_vs_lag(mean0, mean1, max_lag=30)
    
    corr_values, lags, dispersion = plot_vector_corr_and_dispersion_vs_lag(mean0, mean1, max_lag=30, version=version, rpath=rpath)
    
    
