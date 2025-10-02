import os
import numpy as np
import pandas as pd
import xarray as xr
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import h5py
import netCDF4

from scipy.signal import correlate, correlate2d

from matplotlib.colors import LogNorm

def read_meanwinds_to_4d(h5_file):
    """
    Convierte archivo HDF5 de vientos medios (time, alt) a formato 4D (time, lon, lat, alt).
    """
    with h5py.File(h5_file, "r") as f:
        time_s = f["times"][:]             # segundos desde epoch
        alt = f["alts"][:].astype("float64")
        u = f["u0"][:]
        v = f["v0"][:]
        w = f["w0"][:]

    # Expandir a lon=1, lat=1
    u4 = u[:, np.newaxis, np.newaxis, :]  # (time, 1, 1, alt)
    v4 = v[:, np.newaxis, np.newaxis, :]
    w4 = w[:, np.newaxis, np.newaxis, :]

    ds = xr.Dataset(
        data_vars={
            "u": (["time", "lon", "lat", "alt"], u4),
            "v": (["time", "lon", "lat", "alt"], v4),
            "w": (["time", "lon", "lat", "alt"], w4),
        },
        coords={
            "time_s": ("time", time_s),
            "time": ("time", pd.to_datetime(time_s, unit="s", origin="unix")),
            "lon": [12.0],     # dummy
            "lat": [54.0],     # dummy
            "alt": alt,
        },
        attrs={
            "Description": "Converted mean winds (time, alt) to 4D structure (time, lon, lat, alt).",
            "Note": "lon/lat are dummy single points because original file is horizontally averaged."
        }
    )

    # Ordenar altitud
    ds = ds.assign_coords(alt=np.round(ds.alt, 2)).sortby("alt")

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
            "time_s": times,
            "time": pd.to_datetime(times, unit="s", origin="unix"),
            "datetime": ("time", datetimes),
            "lon": lon_1d,
            "lat": lat_1d,
            "alt": alt_1d,
        },
        attrs={
            "Description": "4D wind data loaded from NetCDF file",
        }
    )
    
    ds = ds.assign_coords(alt=ds.alt.astype("float64"))
    ds = ds.assign_coords(alt=np.round(ds.alt, 2))
    ds = ds.sortby("alt")
    
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
            "time_s": time_seconds,  # Time in seconds
            "time": pd.to_datetime(time_seconds, unit="s", origin="unix"),
            "datetime": ("time", time_dt),  # Time in datetime format
            "lon": lon,
            "lat": lat,
            "alt": alt,
        },
    )
    
    ds = ds.assign_coords(alt=ds.alt.astype("float64"))
    ds = ds.assign_coords(alt=np.round(ds.alt, 2))
    ds = ds.sortby("alt")
    
    return ds

def plot_mean_winds(*datasets, cmap="seismic", vmin=-100, vmax=100, version="", rpath=None):
    
    nplots = len(datasets)
    
    fig, axes = plt.subplots(nplots, 2, figsize=(12, nplots*3), sharex=True, sharey=True)
    
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    
    titles = ["HYPER", "ICON"]
    
    for i in range(nplots):
        
        ds = datasets[i]
        
        time = ds.time.values
        alt = ds['alt'].values#.isel(lon=0, lat=0).values
        # mean = ds.mean(dim=["lon","lat"])
        
        # Convert time to matplotlib date format
        # time_datetime = [datetime.datetime.utcfromtimestamp(t) for t in time]
        time_num = mdates.date2num(time)  # Convert to numeric format for plotting


        u = ds.u.values
        v = ds.v.values
        
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
        
# ============================================================
# 1) Igualar resolución temporal SIN interpolar y mantener cobertura
# ============================================================

def _to_datetime64ns(ds):
    """Asegura que ds.time sea datetime64[ns] (necesario para resampling/orden)."""
    if not np.issubdtype(ds.time.dtype, np.datetime64):
        ds = ds.assign_coords(time=pd.to_datetime(ds.time.values, unit="s", origin="unix"))
    else:
        ds = ds.assign_coords(time=pd.to_datetime(ds.time.values))  # fuerza ns
    return ds

def _median_dt_seconds(time_values):
    """Δt típico (mediana) en segundos; requiere tiempos ordenados y únicos."""
    t = np.asarray(time_values).astype("datetime64[ns]")
    if t.size < 2:
        return np.nan
    dtns = np.diff(t).astype("timedelta64[ns]").astype(np.int64)
    # Ignora posibles ceros u outliers muy grandes al calcular la mediana
    dtns = dtns[dtns > 0]
    if dtns.size == 0:
        return np.nan
    return float(np.median(dtns) / 1e9)

def _decimate_to_min_step(ds, step_seconds):
    """
    Devuelve un subconjunto de ds tal que los tiempos consecutivos difieren
    ≥ step_seconds (greedy, sin crear tiempos nuevos ni interpolar).
    Mantiene cobertura total de ds.
    """
    t = np.asarray(ds.time.values).astype("datetime64[ns]").astype("int64")  # ns
    if t.size == 0:
        return ds
    step_ns = int(round(step_seconds * 1e9))
    t0 = t[0]
    # Índices de "bins" enteros desde t0 con tamaño step_ns
    bins = (t - t0) // step_ns
    # Conserva la primera ocurrencia de cada bin
    _, first_idx = np.unique(bins, return_index=True)
    first_idx.sort()
    return ds.isel(time=first_idx)

def regrid_to_d1(mean1: xr.Dataset, mean2: xr.Dataset, method="linear"):
    """
    Interpola mean2 a la grilla (time, alt) de mean1, sin extrapolar.

    Parámetros
    ----------
    mean1 : xr.Dataset
        Dataset de referencia con dims (time, alt).
    mean2 : xr.Dataset
        Dataset a interpolar con dims (time, alt).
    method : str
        Método de interpolación: 'linear' o 'nearest'.

    Retorna
    -------
    xr.Dataset
        mean2 interpolado a la grilla de mean1.
        Los puntos fuera del rango de mean2 quedan como NaN.
    """
    # Ordenar y asegurar tipos
    mean1 = mean1.sortby(["time", "alt"])
    mean2 = mean2.sortby(["time", "alt"])
    mean1 = mean1.assign_coords(
        alt=mean1.alt.astype("float64"),
        time=mean1.time.astype("datetime64[ns]")
    )
    mean2 = mean2.assign_coords(
        alt=mean2.alt.astype("float64"),
        time=mean2.time.astype("datetime64[ns]")
    )

    # Interpolación (sin extrapolación automática)
    mean2_interp = mean2.interp_like(mean1, method=method)

    # Máscara: solo dentro del rango de mean2
    # tmin = np.datetime64(mean2.time.min().values, "ns")
    # tmax = np.datetime64(mean2.time.max().values, "ns")
    # zmin, zmax = float(mean2.alt.min()), float(mean2.alt.max())
    #
    # mask_time = (mean1.time >= tmin) & (mean1.time <= tmax)
    # mask_alt = (mean1.alt >= zmin) & (mean1.alt <= zmax)
    #
    # mask = xr.DataArray(
    #     (mask_time.values[:, None]) & (mask_alt.values[None, :]),
    #     dims=("time", "alt"),
    #     coords={"time": mean1.time, "alt": mean1.alt},
    # )

    return mean2_interp#.where(mask, drop=True)

def equalize_temporal_resolution(ds1, ds2, prefer="coarser"):
    """
    Acepta dos Datasets (vars: 'u','v'; coords: ('time','alt')).
    - Detecta resoluciones temporales (mediana Δt).
    - Reduce (decimate) el de mayor resolución para igualarlo al de menor resolución,
      SIN interpolar y SIN recortar cobertura temporal.
    - Verifica que no haya gaps evidentes en cada resultado.
    
    Parámetros
    ----------
    prefer : {"coarser","ds1","ds2"}
        - "coarser": iguala al paso más grueso entre ds1 y ds2 (recomendado).
        - "ds1"/"ds2": fuerza a igualar a la resolución del dataset elegido.

    Retorna
    -------
    ds1o, ds2o, info
      - ds1o, ds2o: datasets con la misma resolución (por decimación cuando aplica)
      - info: dict con dt1, dt2, target_dt, notes
    """
    # Normaliza tiempo (ns) y orden
    ds1 = _to_datetime64ns(ds1).sortby("time")
    ds2 = _to_datetime64ns(ds2).sortby("time")

    # Normaliza alt tipo/orden para facilitar cruces posteriores
    ds1 = ds1.assign_coords(alt=ds1.alt.astype("float64")).sortby("alt")
    ds2 = ds2.assign_coords(alt=ds2.alt.astype("float64")).sortby("alt")

    dt1 = _median_dt_seconds(ds1.time.values)
    dt2 = _median_dt_seconds(ds2.time.values)

    if np.isnan(dt1) or np.isnan(dt2):
        raise ValueError("No se pudo inferir Δt en alguno de los datasets.")

    if prefer == "coarser":
        target_dt = max(dt1, dt2)
        target_owner = "ds1" if dt1 == target_dt else "ds2"
    elif prefer == "ds1":
        target_dt = dt1
        target_owner = "ds1"
    elif prefer == "ds2":
        target_dt = dt2
        target_owner = "ds2"
    else:
        raise ValueError("prefer debe ser 'coarser', 'ds1' o 'ds2'.")

    # Decimación: iguala el dataset más fino al paso objetivo
    if abs(dt1 - target_dt) / target_dt > 1e-6:
        ds1o = _decimate_to_min_step(ds1, target_dt)
    else:
        ds1o = ds1
    if abs(dt2 - target_dt) / target_dt > 1e-6:
        ds2o = _decimate_to_min_step(ds2, target_dt)
    else:
        ds2o = ds2

    # Verificación rápida de gaps (deltas mucho mayores al paso objetivo)
    def _check_gaps(ds, name, tol=1.5):
        t = np.asarray(ds.time.values).astype("datetime64[ns]")
        if t.size < 2:
            return f"{name}: <2 timestamps"
        diffs = np.diff(t).astype("timedelta64[ns]").astype(np.int64) / 1e9
        if diffs.size == 0:
            return f"{name}: OK (1 timestamp)"
        max_gap = np.max(diffs)
        note = f"{name}: maxΔt={max_gap:.3f}s vs target={target_dt:.3f}s"
        if max_gap > tol * target_dt:
            note += " [GAP detectado]"
        return note

    info = {
        "dt1_sec": dt1,
        "dt2_sec": dt2,
        "target_dt_sec": target_dt,
        "target_from": target_owner,
        "notes": (_check_gaps(ds1o, "ds1o"), _check_gaps(ds2o, "ds2o")),
    }
    
    return ds1o, ds2o#, info


# ============================================================
# 2) Correlación vectorial (u,v) por altitud con igual resolución
# ============================================================
def _to_datetime64ns(ds):
    """Asegura time en datetime64[ns] y ordena por tiempo."""
    if not np.issubdtype(ds.time.dtype, np.datetime64):
        ds = ds.assign_coords(time=pd.to_datetime(ds.time.values, unit="s", origin="unix"))
    else:
        ds = ds.assign_coords(time=pd.to_datetime(ds.time.values))
    return ds.sortby("time")

def _median_dt_seconds(time_vals):
    t = np.asarray(time_vals).astype("datetime64[ns]")
    if t.size < 2: 
        raise ValueError("No se puede inferir Δt con <2 tiempos.")
    dtns = np.diff(t).astype("timedelta64[ns]").astype(np.int64) / 1e9
    dtns = dtns[dtns > 0]
    return float(np.median(dtns))

def _align_alt(ds):
    """Normaliza tipo/orden de alt para intersecciones limpias."""
    return ds.assign_coords(alt=ds.alt.astype("float64")).sortby("alt")

def vector_correlation_by_altitude_and_lag(ds1, ds2, max_lag_steps=15, dt_seconds=None):
    """
    Correlación vectorial (u,v) por altitud y lag, manteniendo fijo ds2.
    Para cada lag L, se usa la intersección de tiempos entre ds2.time y (ds1.time + L*dt).

    Parámetros
    ----------
    ds1, ds2 : xr.Dataset con 'u','v' y dims ('time','alt')
        Deben tener la MISMA resolución temporal (usa tu función de igualar/decimar antes).
    max_lag_steps : int
        Explora lags en [-max_lag_steps, ..., +max_lag_steps].
    dt_seconds : float | None
        Paso temporal en segundos. Si None, se infiere de ds2.

    Retorna
    -------
    corr_da : xr.DataArray (alt, lag_minutes)
        Correlación vectorial por altitud y lag.
    """
    # 1) Normaliza tiempo y altitud
    ds1 = _align_alt(_to_datetime64ns(ds1))
    ds2 = _align_alt(_to_datetime64ns(ds2))
    # Alinea SOLO altitud por intersección; tiempo lo tratamos manualmente
    ds1, ds2 = xr.align(ds1, ds2, join="inner", exclude=["time"])

    # 2) Paso temporal
    if dt_seconds is None:
        dt_seconds = _median_dt_seconds(ds2.time.values)
    dt_ns = np.int64(round(dt_seconds * 1e9))  # nanosegundos por paso

    # 3) Preparativos
    t1 = np.asarray(ds1.time.values).astype("datetime64[ns]").astype("int64")  # ns
    t2 = np.asarray(ds2.time.values).astype("datetime64[ns]").astype("int64")
    lags = np.arange(-max_lag_steps, max_lag_steps + 1)
    lag_minutes = lags * dt_seconds / 60.0

    Z = ds1.dims["alt"]
    corr = np.full((Z, lags.size), np.nan)

    # 4) Bucle por lag y altitud
    for i, L in enumerate(lags):
        # ds1 desplazado en tiempo
        t1_shift = t1 + (L * dt_ns)
        # intersección exacta de marcas (en ns)
        common_ns = np.intersect1d(t2, t1_shift, assume_unique=False)
        if common_ns.size == 0:
            continue

        # tiempos originales equivalentes en ds1 y ds2
        time2_sel = common_ns.view("datetime64[ns]")
        time1_sel = (common_ns - (L * dt_ns)).view("datetime64[ns]")

        # recortes sincronizados sin reindexar (evita introducir NaNs)
        ds2_c = ds2.sel(time=time2_sel)
        ds1_c = ds1.sel(time=time1_sel)

        # matrices (time_common, alt)
        U1, V1 = ds1_c["u"].values, ds1_c["v"].values
        U2, V2 = ds2_c["u"].values, ds2_c["v"].values

        # por altitud
        for j in range(Z):
            u1j, v1j = U1[:, j], V1[:, j]
            u2j, v2j = U2[:, j], V2[:, j]
            # enmascara NaNs pareados
            m = (~np.isnan(u1j)) & (~np.isnan(v1j)) & (~np.isnan(u2j)) & (~np.isnan(v2j))
            if m.sum() == 0:
                continue
            uu1, vv1, uu2, vv2 = u1j[m], v1j[m], u2j[m], v2j[m]
            num = np.sum(uu1 * uu2 + vv1 * vv2)
            den = np.sqrt(np.sum(uu1**2 + vv1**2) * np.sum(uu2**2 + vv2**2))
            corr[j, i] = num / den if den > 0 else np.nan

    corr_da = xr.DataArray(
        corr,
        coords={"alt": ds1.alt.values, "lag": lag_minutes},
        dims=("alt", "lag"),
        name="vec_corr",
        attrs={"description": "Vector correlation (u,v) vs lag (ds1 shifted, ds2 fixed)"},
    )
    return corr_da

def plot_correlation_altitude_lag(corr_da, title="Vector correlation vs lag and altitude", cmap="jet", rpath=None):
    """
    Mapa 2D: x=lag (min), y=alt, color=correlación.
    """
    if not {"alt", "lag"}.issubset(corr_da.dims):
        raise ValueError("El DataArray debe tener dims 'alt' y 'lag'.")

    fig, ax = plt.subplots(figsize=(10, 6))
    pcm = ax.pcolormesh(corr_da["lag"].values, corr_da["alt"].values, corr_da.values,
                        shading="auto", cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(pcm, ax=ax, label="Vector correlation")
    ax.set_xlabel("Lag (min)")
    ax.set_ylabel("Altitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    if rpath is None:
        plt.show()
    else:
        fig.savefig(rpath, format="pdf")
        plt.close(fig)


def vector_correlation_all_altitudes_vs_lag(ds1, ds2, max_lag_steps=30, dt_seconds=None):
    """
    Calcula la correlación vectorial (u,v) considerando todas las alturas
    como una sola muestra vectorial, para distintos lags.

    Parámetros
    ----------
    ds1, ds2 : xr.Dataset
        Con variables 'u' y 'v' y dims ('time','alt').
        Deben tener misma resolución temporal (usar función de decimación antes).
        ds1 puede cubrir rango temporal más amplio.
    max_lag_steps : int
        Número máximo de pasos de lag (positivos y negativos).
    dt_seconds : float | None
        Paso temporal en segundos. Si None, se infiere de ds2.time.

    Retorna
    -------
    xr.DataArray con dims ('lag',), nombre 'vec_corr'.
    """
    # Normalizar
    def _to_datetime64ns(ds):
        if not np.issubdtype(ds.time.dtype, np.datetime64):
            ds = ds.assign_coords(time=pd.to_datetime(ds.time.values, unit="s", origin="unix"))
        else:
            ds = ds.assign_coords(time=pd.to_datetime(ds.time.values))
        return ds.sortby("time")
    
    ds1 = _to_datetime64ns(ds1).assign_coords(alt=ds1.alt.astype("float64")).sortby("alt")
    ds2 = _to_datetime64ns(ds2).assign_coords(alt=ds2.alt.astype("float64")).sortby("alt")
    ds1, ds2 = xr.align(ds1, ds2, join="inner", exclude=["time"])  # intersectar altitud
    
    # Paso temporal
    if dt_seconds is None:
        t = np.asarray(ds2.time.values).astype("datetime64[ns]")
        diffs = np.diff(t).astype("timedelta64[ns]").astype(np.int64) / 1e9
        dt_seconds = float(np.median(diffs))
    dt_ns = np.int64(round(dt_seconds * 1e9))
    
    t1 = np.asarray(ds1.time.values).astype("datetime64[ns]").astype("int64")
    t2 = np.asarray(ds2.time.values).astype("datetime64[ns]").astype("int64")
    
    lags = np.arange(-max_lag_steps, max_lag_steps+1)
    corr = np.full(lags.size, np.nan)
    
    for i, L in enumerate(lags):
        # tiempos comunes entre ds2 y ds1 desplazado
        t1_shift = t1 + (L * dt_ns)
        common_ns = np.intersect1d(t2, t1_shift, assume_unique=False)
        if common_ns.size == 0:
            continue
        
        t2_sel = common_ns.view("datetime64[ns]")
        t1_sel = (common_ns - (L * dt_ns)).view("datetime64[ns]")
        
        ds2_c = ds2.sel(time=t2_sel)
        ds1_c = ds1.sel(time=t1_sel)
        
        # aplanar todas las alturas y tiempos
        u1 = ds1_c["u"].values.ravel()
        v1 = ds1_c["v"].values.ravel()
        u2 = ds2_c["u"].values.ravel()
        v2 = ds2_c["v"].values.ravel()
        
        mask = (~np.isnan(u1)) & (~np.isnan(v1)) & (~np.isnan(u2)) & (~np.isnan(v2))
        if mask.sum() == 0:
            continue
        
        u1, v1, u2, v2 = u1[mask], v1[mask], u2[mask], v2[mask]
        num = np.sum(u1*u2 + v1*v2)
        den = np.sqrt(np.sum(u1**2+v1**2) * np.sum(u2**2+v2**2))
        corr[i] = num/den if den > 0 else np.nan
    
    lag_minutes = lags * dt_seconds / 60.0
    da = xr.DataArray(corr, coords={"lag": lag_minutes}, dims=("lag",), name="vec_corr")
    da.attrs["description"] = "Vector correlation (u,v) across all altitudes vs lag (ds1 shifted, ds2 fixed)"
    return da

def plot_correlation_vs_lag(corr_da, title="Vector correlation vs lag", rpath=None):
    """
    Grafica correlación total (todas las alturas) en función del lag.
    """
    if "lag" not in corr_da.dims:
        raise ValueError("El DataArray debe tener la dimensión 'lag'.")
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(corr_da["lag"].values, corr_da.values, marker="o", linestyle="-", color="blue")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Lag (min)")
    ax.set_ylabel("Vector correlation (all altitudes)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    
    if rpath is None:
        plt.show()
    else:
        fig.savefig(rpath, format="pdf")
        plt.close(fig)

def plot_density_at_best_lag(ds1, ds2, corr_da, bins=100, cmap="inferno", rpath=None):
    """
    Produce figuras de calidad para publicación: densidad ds1 vs ds2 al lag de máxima correlación.
    Grafica u y v en subplots separados.
    """
    # 1. Identificar lag óptimo
    best_idx = int(np.nanargmax(corr_da.values))
    best_lag = float(corr_da["lag"].values[best_idx])
    print(f"Lag óptimo: {best_lag:.2f} min, correlación = {corr_da.values[best_idx]:.3f}")

    # Paso temporal (ns)
    def _to_datetime64ns(ds):
        return ds.assign_coords(time=pd.to_datetime(ds.time.values)).sortby("time")
    ds1 = _to_datetime64ns(ds1)
    ds2 = _to_datetime64ns(ds2)
    ds1, ds2 = xr.align(ds1, ds2, join="inner", exclude=["time"])

    t1 = np.asarray(ds1.time.values).astype("datetime64[ns]").astype("int64")
    t2 = np.asarray(ds2.time.values).astype("datetime64[ns]").astype("int64")
    dt_ns = int(np.median(np.diff(t1)))
    lag_steps = int(round(best_lag * 60 / (dt_ns / 1e9)))
    t1_shift = t1 + lag_steps * dt_ns
    common_ns = np.intersect1d(t2, t1_shift)
    if common_ns.size == 0:
        raise ValueError("No hay intersección de tiempos al lag óptimo.")
    t2_sel = common_ns.view("datetime64[ns]")
    t1_sel = (common_ns - lag_steps * dt_ns).view("datetime64[ns]")
    ds2_c = ds2.sel(time=t2_sel)
    ds1_c = ds1.sel(time=t1_sel)

    # 2. Configuración de la figura
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))#, constrained_layout=True)

    titles = ["Zonal", "Meridional"]
    for i, var in enumerate(["u", "v"]):
        
        x = ds1_c[var].values.ravel()
        y = ds2_c[var].values.ravel()
        mask = (~np.isnan(x)) & (~np.isnan(y))
        x, y = x[mask], y[mask]
        rmsd = np.sqrt(np.mean((x - y) ** 2))
        print(f"RMSE {var}: {rmsd:.2f}")

        ax = axs[i]
        title = titles[i]
        h = ax.hist2d(x, y, bins=bins, cmap=cmap, norm=LogNorm())
        cb = fig.colorbar(h[3], ax=ax, pad=0.03, shrink=0.6)
        cb.set_label("Counts")
        ax.axline((0, 0), slope=1, color="k", lw=1, linestyle="--")
        ax.set_xlabel(r"%s$_{HYPER}$ (m/s)" %var)
        ax.set_ylabel(r"%s$_{ICON}$ (m/s)" %var)
        ax.set_title(f"{title} velocity (RMSE={rmsd:.1f} (m/s))")
        ax.set_aspect("equal", "box")
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.grid(True, linestyle=":", alpha=0.5)

    if rpath is None:
        plt.show()
    else:
        figfile = os.path.join(rpath, "density_map.pdf")
        fig.savefig(figfile, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

def vector_correlation_by_altitude_vs_lag(ds1, ds2, max_lag_steps=10, dt_seconds=None):
    """
    Compute vector correlation (u,v) per altitude and lag.
    ds1 is shifted, ds2 fixed. Both must have same resolution and alt grid.
    
    Returns
    -------
    xr.DataArray with dims (alt, lag).
    """
    # Ensure sorted and aligned
    ds1 = ds1.sortby(["time", "alt"])
    ds2 = ds2.sortby(["time", "alt"])
    ds1, ds2 = xr.align(ds1, ds2, join="inner", exclude=["time"])

    # Time step
    if dt_seconds is None:
        t = np.asarray(ds2.time.values).astype("datetime64[ns]")
        diffs = np.diff(t).astype("timedelta64[ns]").astype(np.int64) / 1e9
        dt_seconds = float(np.median(diffs))
    dt_ns = np.int64(round(dt_seconds * 1e9))

    # Times as ns
    t1 = np.asarray(ds1.time.values).astype("datetime64[ns]").astype("int64")
    t2 = np.asarray(ds2.time.values).astype("datetime64[ns]").astype("int64")

    lags = np.arange(-max_lag_steps, max_lag_steps+1)
    lag_minutes = lags * dt_seconds / 60.0

    Z = ds1.dims["alt"]
    corr = np.full((Z, lags.size), np.nan)

    # Loop over lags
    for i, L in enumerate(lags):
        t1_shift = t1 + (L * dt_ns)
        common_ns = np.intersect1d(t2, t1_shift)
        if common_ns.size == 0:
            continue

        t2_sel = common_ns.view("datetime64[ns]")
        t1_sel = (common_ns - (L * dt_ns)).view("datetime64[ns]")

        ds2_c = ds2.sel(time=t2_sel)
        ds1_c = ds1.sel(time=t1_sel)

        U1, V1 = ds1_c["u"].values, ds1_c["v"].values
        U2, V2 = ds2_c["u"].values, ds2_c["v"].values

        for j in range(Z):
            u1, v1 = U1[:, j], V1[:, j]
            u2, v2 = U2[:, j], V2[:, j]
            mask = (~np.isnan(u1)) & (~np.isnan(v1)) & (~np.isnan(u2)) & (~np.isnan(v2))
            if mask.sum() == 0:
                continue
            num = np.sum(u1[mask]*u2[mask] + v1[mask]*v2[mask])
            den = np.sqrt(np.sum(u1[mask]**2+v1[mask]**2) * np.sum(u2[mask]**2+v2[mask]**2))
            corr[j, i] = num/den if den > 0 else np.nan

    da = xr.DataArray(
        corr,
        coords={"alt": ds1.alt.values, "lag": lag_minutes},
        dims=("alt", "lag"),
        name="vec_corr",
        attrs={"description": "Vector correlation per altitude and lag"}
    )
    return da

import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_correlation_vs_lag_with_altitudes(
    corr_total, corr_per_alt, title="Correlation HYPER vs ICON", cmap="Grays", rpath=None
):
    """
    Overlay total correlation curve with per-altitude correlation curves,
    color-coded by altitude.
    """
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })
    
    fig, ax = plt.subplots(figsize=(9, 6))

    # Set up colormap for altitude
    alts = corr_per_alt.alt.values
    norm = mcolors.Normalize(vmin=np.min(alts), vmax=np.max(alts))
    cmap_obj = cm.get_cmap(cmap)

    # Per-altitude curves with color gradient
    for z in alts:
        color = cmap_obj(norm(z))
        ax.plot(
            corr_per_alt.lag.values,
            corr_per_alt.sel(alt=z).values,
            "-",
            color=color,
            alpha=0.5,
            linewidth=1.0,
        )

    # Total curve (on top)
    ax.plot(
        corr_total.lag.values,
        corr_total.values,
        color="blue",
        linewidth=2.5,
        label="All altitudes",
    )

    # Colorbar legend
    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Altitude")

    # Axis labels and grid
    ax.axhline(0, color="k", lw=0.8, linestyle="--")
    ax.set_xlabel("Lag (min)")
    ax.set_ylabel("Vector correlation")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    if rpath is None:
        plt.show()
    else:
        figfile = os.path.join(rpath, "vector_corr.pdf")
        fig.savefig(figfile, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
if __name__=="__main__":

    mean_path = "/Users/radar/Data/IAP/SIMONe/Germany/SpaceX/mean_wind_20250218.hdf5"
    ds1 = read_meanwinds_to_4d(mean_path)
    
    # Load the wind dataset
    # file_path = "/Users/radar/Data/IAP/SIMONe/Germany/SpaceX/hyper24/winds/winds3D_20250219.h5"
    file_path = "/Users/radar/Data/IAP/SIMONe/Germany/SpaceX/hyper72/winds/winds3D_20250218_003000_v1.1.0.nc"
    
    rpath = None#os.path.split(mean_path)[0]
    
    version = "new"
    # Select the desired time step (e.g., the first time index)
    time_dt = datetime.datetime(2025,2,19,19,0,0)
    # Select the altitude index for the cut (modify as needed)
    altitudes_km = [90, 95, 100]
    
    ds0 = read_winds_netcdf(file_path)
    ds0 = ds0.where( (ds0.lon>=10) & (ds0.lon<=15) & (ds0.lat>=52.5) & (ds0.lat<=55.5), drop=True)
    
    if version == "new":
        file_path = "/Users/radar/Data/IAP/SIMONe/Germany/SpaceX/ICON/UA-ICON_NWP_atm_ML_DOM01_falcon2_20250219-20.nc"
    else:
        file_path = "/Users/radar/Data/IAP/SIMONe/Germany/SpaceX/ICON/UA-ICON_NWP_atm_DOM01_80-100km_20250219-20.nc"
    
    ds2 = read_icon_file(file_path)
    
    ds2 = ds2.where( (ds2.time>= (ds0.time.min()) ) & (ds2.time<=ds0.time.max()), drop=True)           
    ds2 = ds2.where( (ds2.lon>=ds0.lon.min()) & (ds2.lon<=ds0.lon.max()) & (ds2.lat>=ds0.lat.min()) & (ds2.lat<=ds0.lat.max()), drop=True)
    
    ds1 = ds1.where( (ds1.alt>=80) & (ds1.alt<=101), drop=True)
    ds2 = ds2.where( ( ds2.alt>=ds0.alt.min()-0.01) & (ds2.alt<=ds0.alt.max()+0.01), drop=True)
    
    # plot_mean_winds( ds0, ds, version=version)#, rpath=rpath)
    
    mean1 = ds1.mean(dim=["lon","lat"])
    mean2 = ds2.mean(dim=["lon","lat"])
    
    # dsm1, dsm2 = equalize_temporal_resolution(mean1, mean2)
    
    dsm1 = mean1
    dsm2 = regrid_to_d1(mean1, mean2)
    
    plot_mean_winds(dsm1, dsm2, rpath=rpath)
    # corr_da = vector_correlation_by_altitude_and_lag(dsm1, dsm2)
    # plot_correlation_altitude_lag(corr_da)
    
    corr_total = vector_correlation_all_altitudes_vs_lag(dsm1, dsm2, max_lag_steps=30)
    corr_per_alt = vector_correlation_by_altitude_vs_lag(dsm1, dsm2, max_lag_steps=30)

    plot_correlation_vs_lag_with_altitudes(corr_total, corr_per_alt, rpath=rpath)

    # plot_correlation_vs_lag(corr_da)
    
    # 3) Graficar densidad al lag óptimo
    plot_density_at_best_lag(dsm1, dsm2, corr_total, bins=15, rpath=rpath)
