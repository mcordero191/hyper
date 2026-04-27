#!/usr/bin/env python3
"""
wind_products_plot_dailyfiles.py

Wind-only workflow (u,v,w) across DAILY FILES:
  - raw: daily mean winds (daily-binned)
  - mean: running-window mean of daily winds
  - perturbation: raw - mean
  - residual_kinematic: mean (or additional low-pass)

Reads multiple daily files from a directory (NetCDF or HDF5), concatenates time,
then computes products and plots:
  A) Time–X planes (as before): time vs lon / lat / alt
  B) NEW: Lat–Lon maps at selected altitudes for selected day(s)
"""

import os
import glob
import argparse
import numpy as np
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import h5py
import netCDF4


# -----------------------------
# Utilities: daily + running mean
# -----------------------------
def bin_daily_mean(t_sec: np.ndarray, data: np.ndarray, seconds_per_day: float = 86400.0):
    t = np.asarray(t_sec, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if data.shape[0] != t.size:
        raise ValueError("data first dimension must match time length")

    day_index = np.floor(t / seconds_per_day).astype(np.int64)
    days = np.unique(day_index)

    t_out = np.empty(days.size, dtype=float)
    out = []

    for i, d in enumerate(days):
        m = (day_index == d)
        t_out[i] = np.nanmean(t[m])
        out.append(np.nanmean(data[m, ...], axis=0))

    return t_out, np.stack(out, axis=0)


def running_mean_time_nan_safe(x: np.ndarray, win: int, centered: bool = True, min_valid_frac: float = 0.5):
    x = np.asarray(x, dtype=float)
    nt = x.shape[0]
    if win <= 1 or nt == 0:
        return x.copy()
    win = int(win)

    valid = np.isfinite(x)
    x0 = np.where(valid, x, 0.0)

    csum = np.cumsum(x0, axis=0)
    ccnt = np.cumsum(valid.astype(np.float64), axis=0)

    def rsum(arr, i0, i1):
        if i0 == 0:
            return arr[i1]
        return arr[i1] - arr[i0 - 1]

    y = np.full_like(x, np.nan, dtype=float)

    if centered:
        half = win // 2
        for i in range(nt):
            i0 = max(0, i - half)
            i1 = min(nt - 1, i + half)
            s = rsum(csum, i0, i1)
            n = rsum(ccnt, i0, i1)
            required = min_valid_frac * (i1 - i0 + 1)
            ok = n >= required
            y[i] = np.where(ok, s / np.where(n == 0, np.nan, n), np.nan)
    else:
        for i in range(nt):
            i1 = i
            i0 = max(0, i - win + 1)
            s = rsum(csum, i0, i1)
            n = rsum(ccnt, i0, i1)
            required = min_valid_frac * (i1 - i0 + 1)
            ok = n >= required
            y[i] = np.where(ok, s / np.where(n == 0, np.nan, n), np.nan)

    return y


def compute_wind_products(t, u, v, w, running_days=21, centered=True, min_valid_frac=0.5, extra_lowpass_days=None):
    # raw = daily mean
    t_day, u_day = bin_daily_mean(t, u)
    _,    v_day  = bin_daily_mean(t, v)
    _,    w_day  = bin_daily_mean(t, w)

    # mean = running mean of daily
    u_mean = running_mean_time_nan_safe(u_day, running_days, centered=centered, min_valid_frac=min_valid_frac)
    v_mean = running_mean_time_nan_safe(v_day, running_days, centered=centered, min_valid_frac=min_valid_frac)
    w_mean = running_mean_time_nan_safe(w_day, running_days, centered=centered, min_valid_frac=min_valid_frac)

    # perturbation
    u_p = u_day - u_mean
    v_p = v_day - v_mean
    w_p = w_day - w_mean

    # residual kinematic
    if extra_lowpass_days is not None and int(extra_lowpass_days) > 1:
        lp = int(extra_lowpass_days)
        u_res = running_mean_time_nan_safe(u_mean, lp, centered=centered, min_valid_frac=min_valid_frac)
        v_res = running_mean_time_nan_safe(v_mean, lp, centered=centered, min_valid_frac=min_valid_frac)
        w_res = running_mean_time_nan_safe(w_mean, lp, centered=centered, min_valid_frac=min_valid_frac)
    else:
        u_res, v_res, w_res = u_mean, v_mean, w_mean

    return {
        "raw": {"t": t_day, "u": u_day, "v": v_day, "w": w_day},
        "mean": {"t": t_day, "u": u_mean, "v": v_mean, "w": w_mean},
        "perturbation": {"t": t_day, "u": u_p, "v": v_p, "w": w_p},
        "residual_kinematic": {"t": t_day, "u": u_res, "v": v_res, "w": w_res},
    }


# -----------------------------
# Reading: single file
# -----------------------------
def reduce_coords(lon, lat, alt):
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    alt = np.asarray(alt)

    if lon.ndim == 1 and lat.ndim == 1 and alt.ndim == 1:
        return lon.astype(float), lat.astype(float), alt.astype(float)

    if lon.ndim == 3 and lat.ndim == 3 and alt.ndim == 3:
        lon1d = lon[:, 0, 0]
        lat1d = lat[0, :, 0]
        alt1d = alt[0, 0, :]
        return lon1d.astype(float), lat1d.astype(float), alt1d.astype(float)

    raise ValueError(f"Unsupported coordinate shapes: lon{lon.shape}, lat{lat.shape}, alt{alt.shape}")


def read_winds_netcdf(path: str):
    ds = netCDF4.Dataset(path, "r")
    tname = "time" if "time" in ds.variables else ("times" if "times" in ds.variables else None)
    lonname = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
    latname = "latitude" if "latitude" in ds.variables else ("lat" if "lat" in ds.variables else None)
    altname = "altitude" if "altitude" in ds.variables else ("alt" if "alt" in ds.variables else None)
    if tname is None or lonname is None or latname is None or altname is None:
        ds.close()
        raise ValueError(f"Missing coordinate variables in NetCDF file: {path}")

    t = np.array(ds.variables[tname][:], dtype=float)
    lon = np.array(ds.variables[lonname][:], dtype=float)
    lat = np.array(ds.variables[latname][:], dtype=float)
    alt = np.array(ds.variables[altname][:], dtype=float)

    u = np.array(ds.variables["u"][:], dtype=float)
    v = np.array(ds.variables["v"][:], dtype=float)
    w = np.array(ds.variables["w"][:], dtype=float)
    ds.close()

    if u.ndim != 4:
        raise ValueError(f"Expected u to be 4D (time,lon,lat,alt). Got {u.shape} in {path}")

    return t, lon, lat, alt, u, v, w


def read_winds_hdf5(path: str):
    with h5py.File(path, "r") as fp:
        # Stacked format
        if all(k in fp for k in ["times", "u", "v", "w"]):
            t = np.array(fp["times"][:], dtype=float)
            lon = np.array(fp["lon"][:] if "lon" in fp else fp["longitude"][:], dtype=float)
            lat = np.array(fp["lat"][:] if "lat" in fp else fp["latitude"][:], dtype=float)
            alt = np.array(fp["alt"][:] if "alt" in fp else fp["altitude"][:], dtype=float)

            u = np.array(fp["u"][:], dtype=float)
            v = np.array(fp["v"][:], dtype=float)
            w = np.array(fp["w"][:], dtype=float)

            lon1d, lat1d, alt1d = reduce_coords(lon, lat, alt)
            return t, lon1d, lat1d, alt1d, u, v, w

        # Group-per-time format
        lon = np.array(fp["lon"][:], dtype=float) if "lon" in fp else None
        lat = np.array(fp["lat"][:], dtype=float) if "lat" in fp else None
        alt = np.array(fp["alt"][:], dtype=float) if "alt" in fp else None
        if lon is None or lat is None or alt is None:
            raise ValueError(f"HDF5 group-per-time format requires lon,lat,alt at root: {path}")

        group_keys = []
        for k in fp.keys():
            if k in ["lon", "lat", "alt", "mask", "times"]:
                continue
            obj = fp[k]
            if isinstance(obj, h5py.Group) and all(x in obj for x in ["u", "v", "w"]):
                group_keys.append(k)

        if not group_keys:
            raise ValueError(f"No time groups found in HDF5 file: {path}")

        group_keys = sorted(group_keys, key=lambda s: float(s))
        t = np.array([float(k) for k in group_keys], dtype=float)

        u0 = np.array(fp[group_keys[0]]["u"][:], dtype=float)
        shape = (len(t),) + u0.shape
        u = np.empty(shape, dtype=float)
        v = np.empty(shape, dtype=float)
        w = np.empty(shape, dtype=float)

        for i, k in enumerate(group_keys):
            u[i] = np.array(fp[k]["u"][:], dtype=float)
            v[i] = np.array(fp[k]["v"][:], dtype=float)
            w[i] = np.array(fp[k]["w"][:], dtype=float)

        lon1d, lat1d, alt1d = reduce_coords(lon, lat, alt)
        return t, lon1d, lat1d, alt1d, u, v, w


def read_winds_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".nc", ".nc4", ".netcdf"]:
        return read_winds_netcdf(path)
    if ext in [".h5", ".hdf5"]:
        return read_winds_hdf5(path)
    raise ValueError(f"Unsupported file extension: {ext}")


# -----------------------------
# Reading: multiple daily files
# -----------------------------
def concat_time_series(files):
    if not files:
        raise ValueError("No input files provided.")

    t_all, u_all, v_all, w_all = [], [], [], []
    lon_ref = lat_ref = alt_ref = None

    for f in files:
        t, lon, lat, alt, u, v, w = read_winds_file(f)

        if lon_ref is None:
            lon_ref, lat_ref, alt_ref = lon, lat, alt
        else:
            if lon.shape != lon_ref.shape or lat.shape != lat_ref.shape or alt.shape != alt_ref.shape:
                raise ValueError(f"Grid shape mismatch in {f}")
            if not (np.allclose(lon, lon_ref, equal_nan=True) and
                    np.allclose(lat, lat_ref, equal_nan=True) and
                    np.allclose(alt, alt_ref, equal_nan=True)):
                raise ValueError(f"Grid values mismatch in {f} (lon/lat/alt differ)")

        t_all.append(t); u_all.append(u); v_all.append(v); w_all.append(w)

    t_cat = np.concatenate(t_all, axis=0)
    u_cat = np.concatenate(u_all, axis=0)
    v_cat = np.concatenate(v_all, axis=0)
    w_cat = np.concatenate(w_all, axis=0)

    idx = np.argsort(t_cat)
    return t_cat[idx], lon_ref, lat_ref, alt_ref, u_cat[idx], v_cat[idx], w_cat[idx]


# -----------------------------
# Plotting: time planes (existing)
# -----------------------------
def to_days(t_sec: np.ndarray):
    t = np.asarray(t_sec, dtype=float)
    t0 = np.nanmin(t)
    return (t - t0) / 86400.0


def pick_index(coord_1d: np.ndarray, c0):
    if c0 is None:
        return len(coord_1d) // 2
    return int(np.abs(coord_1d - c0).argmin())


def window_slice(n: int, i_center: int, half_width: int) -> slice:
    i0 = max(0, i_center - half_width)
    i1 = min(n, i_center + half_width + 1)
    return slice(i0, i1)


def extract_time_planes(product, lon, lat, alt,
                        lon0=None, lat0=None, alt0=None,
                        win_lon=0, win_lat=0, win_alt=0):
    t = product["t"]
    u = product["u"]
    v = product["v"]
    w = product["w"]

    nt, nlon, nlat, nalt = u.shape

    ix = pick_index(lon, lon0)
    iy = pick_index(lat, lat0)
    iz = pick_index(alt, alt0)

    slon = window_slice(nlon, ix, win_lon)
    slat = window_slice(nlat, iy, win_lat)
    salt = window_slice(nalt, iz, win_alt)

    u_txlon = np.nanmean(u[:, :, slat, salt], axis=(2, 3))
    v_txlon = np.nanmean(v[:, :, slat, salt], axis=(2, 3))
    w_txlon = np.nanmean(w[:, :, slat, salt], axis=(2, 3))

    u_txlat = np.nanmean(u[:, slon, :, salt], axis=(1, 3))
    v_txlat = np.nanmean(v[:, slon, :, salt], axis=(1, 3))
    w_txlat = np.nanmean(w[:, slon, :, salt], axis=(1, 3))

    u_txalt = np.nanmean(u[:, slon, slat, :], axis=(1, 2))
    v_txalt = np.nanmean(v[:, slon, slat, :], axis=(1, 2))
    w_txalt = np.nanmean(w[:, slon, slat, :], axis=(1, 2))

    chosen = {"lon0": float(lon[ix]), "lat0": float(lat[iy]), "alt0": float(alt[iz])}

    planes = {
        "txlon": (lon, u_txlon, v_txlon, w_txlon, "Longitude [deg]"),
        "txlat": (lat, u_txlat, v_txlat, w_txlat, "Latitude [deg]"),
        "txalt": (alt, u_txalt, v_txalt, w_txalt, "Altitude [km]"),
    }
    return t, planes, chosen


def plot_three_panel_time_plane(t_sec, x_coord, U, V, W, ylabel, title, outpath,
                                vmins=(-100, -100, -5), vmaxs=(100, 100, 5), cmap="seismic"):
    tday = to_days(t_sec)
    T, X = np.meshgrid(tday, x_coord, indexing="ij")

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

    for ax, data, name, vmin, vmax in zip(
        axs, [U, V, W], ["u [m/s]", "v [m/s]", "w [m/s]"], vmins, vmaxs
    ):
        im = ax.pcolormesh(T, X, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, orientation="vertical")

    axs[-1].set_xlabel("Time [days since start]")
    fig.suptitle(title, fontsize=12)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# -----------------------------
# NEW: Lat–Lon maps at selected altitudes
# -----------------------------
def parse_float_list(csv: str):
    if csv is None or csv.strip() == "":
        return []
    return [float(x.strip()) for x in csv.split(",") if x.strip() != ""]


def time_to_yyyymmdd(t_sec: float) -> str:
    return datetime.datetime.utcfromtimestamp(float(t_sec)).strftime("%Y%m%d")


def select_day_indices(t_day: np.ndarray, day: str, all_days: bool):
    """
    t_day are representative day times (seconds).
    If all_days: return all indices.
    Else if day is None: return [0] (first day).
    Else day is 'YYYYMMDD': return indices matching that date.
    """
    if all_days:
        return list(range(len(t_day)))

    if day is None:
        return [0]

    target = day.strip()
    idx = []
    for i, tt in enumerate(t_day):
        if time_to_yyyymmdd(tt) == target:
            idx.append(i)
    if not idx:
        raise ValueError(f"No day matches {target}. Available range: {time_to_yyyymmdd(t_day[0])}..{time_to_yyyymmdd(t_day[-1])}")
    return idx


def plot_latlon_maps_for_day(product, lon, lat, alt, day_index: int, alt_list_km,
                             outdir, prefix, ext="png",
                             vmins=(-100, -100, -5), vmaxs=(100, 100, 5), cmap="seismic",
                             quiver: bool = False, quiver_step: int = 3):
    """
    For a given day (index in product time axis), plot lat-lon maps for u,v,w
    at each requested altitude.

    product u,v,w shape: (nday, nlon, nlat, nalt)
    """
    t_day = product["t"]
    u = product["u"][day_index]
    v = product["v"][day_index]
    w = product["w"][day_index]

    day_str = time_to_yyyymmdd(t_day[day_index])

    # Mesh for pcolormesh: lon x lat
    LON, LAT = np.meshgrid(lon, lat, indexing="ij")

    for akm in alt_list_km:
        iz = int(np.abs(alt - akm).argmin())
        akm_used = float(alt[iz])

        U = u[:, :, iz]
        V = v[:, :, iz]
        W = w[:, :, iz]

        fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

        for ax, data, name, vmin, vmax in zip(
            axs, [U, V, W], ["u [m/s]", "v [m/s]", "w [m/s]"], vmins, vmaxs
        ):
            im = ax.pcolormesh(LON, LAT, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel("Longitude [deg]")
            ax.set_ylabel("Latitude [deg]")
            ax.set_title(name)
            fig.colorbar(im, ax=ax, orientation="vertical")

        if quiver:
            # Overlay arrows on the u-panel
            ax0 = axs[0]
            ss = int(max(1, quiver_step))
            ax0.quiver(
                LON[::ss, ::ss], LAT[::ss, ::ss],
                U[::ss, ::ss], V[::ss, ::ss],
                scale=None, angles="xy"
            )

        fig.suptitle(f"{prefix} | {day_str} | alt={akm_used:.2f} km", fontsize=12)

        fname = f"map_latlon_{prefix}_h{akm_used:.2f}km_{day_str}.{ext}"
        fig.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close(fig)

def plot_latlon_maps_for_day_cartopy(
    product, lon, lat, alt, day_index: int, alt_list_km,
    outdir, prefix, ext="png",
    vmins=(-100, -100, -5), vmaxs=(100, 100, 5), cmap="seismic",
    projection="aeqd",  # "aeqd" | "lcc" | "stereo"
    add_coastlines=True,
    gridlines=True,
    quiver=False, quiver_step=3,
):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    t_day = product["t"]
    u = product["u"][day_index]
    v = product["v"][day_index]
    w = product["w"][day_index]

    day_str = datetime.datetime.utcfromtimestamp(float(t_day[day_index])).strftime("%Y%m%d")

    # Data are lon/lat in degrees => PlateCarree
    data_crs = ccrs.PlateCarree()

    # Compute center for projection
    lon0 = float(np.nanmean(lon))
    lat0 = float(np.nanmean(lat))

    if projection == "aeqd":
        proj = ccrs.AzimuthalEquidistant(central_longitude=lon0, central_latitude=lat0)
    elif projection == "lcc":
        proj = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0)
    elif projection == "stereo":
        proj = ccrs.Stereographic(central_longitude=lon0, central_latitude=lat0)
    else:
        raise ValueError("projection must be one of: aeqd, lcc, stereo")

    # Mesh (nlon,nlat) for pcolormesh
    LON, LAT = np.meshgrid(lon, lat, indexing="ij")

    # Optional: set plot extent from your data bounds (in data CRS)
    extent = [float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))]

    for akm in alt_list_km:
        iz = int(np.abs(alt - akm).argmin())
        akm_used = float(alt[iz])

        U = u[:, :, iz]
        V = v[:, :, iz]
        W = w[:, :, iz]

        fig = plt.figure(figsize=(15, 5), constrained_layout=True)
        axs = [
            fig.add_subplot(1, 3, 1, projection=proj),
            fig.add_subplot(1, 3, 2, projection=proj),
            fig.add_subplot(1, 3, 3, projection=proj),
        ]

        for ax in axs:
            ax.set_extent(extent, crs=data_crs)
            if add_coastlines:
                ax.coastlines(resolution="10m", linewidth=0.8)
                ax.add_feature(cfeature.BORDERS, linewidth=0.4)
            if gridlines:
                gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0.4, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False

        for ax, data, name, vmin, vmax in zip(
            axs, [U, V, W], ["u [m/s]", "v [m/s]", "w [m/s]"], vmins, vmaxs
        ):
            im = ax.pcolormesh(LON, LAT, data, transform=data_crs,
                               shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(name)
            fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.9)

        if quiver:
            ax0 = axs[0]
            ss = int(max(1, quiver_step))
            ax0.quiver(
                LON[::ss, ::ss], LAT[::ss, ::ss],
                U[::ss, ::ss], V[::ss, ::ss],
                transform=data_crs,
                angles="xy"
            )

        fig.suptitle(f"{prefix} | {day_str} | alt={akm_used:.2f} km | proj={projection}", fontsize=12)

        fname = f"map_latlon_{prefix}_h{akm_used:.2f}km_{day_str}.{ext}"
        fig.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close(fig)
        
# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Wind-only products + time planes + lat-lon maps from DAILY FILES")
    
    # Input: directory of daily files
    p.add_argument("--input-dir", default="/Users/radar/Data/IAP/SIMONe/Germany/IAPData/hyper/winds", help="Directory containing daily wind files")
    # Output
    p.add_argument("--outdir", default="/Users/radar/Data/IAP/SIMONe/Germany/IAPData/hyper/plots", help="Output directory for plots")
    
    p.add_argument("--pattern", default="*.nc", help="Glob pattern inside input-dir (e.g., '*.nc' or '*.h5')")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--skip-files", type=int, default=0)

    p.add_argument("--product", default="mean",
                   choices=["raw", "mean", "perturbation", "residual_kinematic"],
                   help="Which product to plot")

    p.add_argument("--running-days", type=int, default=21)
    p.add_argument("--centered", type=int, default=1)
    p.add_argument("--min-valid-frac", type=float, default=0.5)
    p.add_argument("--extra-lowpass-days", type=int, default=None)

    # time-plane extraction point
    p.add_argument("--lon0", type=float, default=None)
    p.add_argument("--lat0", type=float, default=None)
    p.add_argument("--alt0", type=float, default=None)
    p.add_argument("--win-lon", type=int, default=0)
    p.add_argument("--win-lat", type=int, default=0)
    p.add_argument("--win-alt", type=int, default=0)

    # plotting style
    p.add_argument("--ext", default="png")
    p.add_argument("--cmap", default="seismic")
    p.add_argument("--vmins", default="-100,-100,-1")
    p.add_argument("--vmaxs", default="100,100,1")

    # NEW: lat-lon map options
    p.add_argument("--make-maps", type=int, default=1, help="1 to produce lat-lon maps, 0 to skip")
    p.add_argument("--map-alts", type=str, default="", help="Comma-separated altitude list in km (e.g. '80,85,90')")
    p.add_argument("--map-day", type=str, default=None, help="YYYYMMDD for single day selection (default first day)")
    p.add_argument("--map-all-days", type=int, default=0, help="1 to generate maps for all days")
    p.add_argument("--map-quiver", type=int, default=1, help="1 to overlay quiver on u-panel")
    p.add_argument("--map-quiver-step", type=int, default=2, help="Downsampling step for quiver arrows")

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files and args.pattern == "*.nc":
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))
    if not files:
        raise ValueError(f"No files found in {args.input_dir} with pattern {args.pattern}")

    if args.skip_files:
        files = files[args.skip_files:]
    if args.max_files is not None:
        files = files[:args.max_files]
    if not files:
        raise ValueError("No files left after applying skip/max limits.")

    print(f"Found {len(files)} files. Reading and concatenating time ...")
    t, lon, lat, alt, u, v, w = concat_time_series(files)

    products = compute_wind_products(
        t, u, v, w,
        running_days=args.running_days,
        centered=bool(args.centered),
        min_valid_frac=args.min_valid_frac,
        extra_lowpass_days=args.extra_lowpass_days,
    )

    prod = products[args.product]

    vmins = tuple(float(x) for x in args.vmins.split(","))
    vmaxs = tuple(float(x) for x in args.vmaxs.split(","))

    # --- A) time planes ---
    t_day, planes, chosen = extract_time_planes(
        prod, lon, lat, alt,
        lon0=args.lon0, lat0=args.lat0, alt0=args.alt0,
        win_lon=args.win_lon, win_lat=args.win_lat, win_alt=args.win_alt
    )
    center_txt = f"(lon={chosen['lon0']:.2f}°, lat={chosen['lat0']:.2f}°, alt={chosen['alt0']:.1f} km)"

    for key in ["txlon", "txlat", "txalt"]:
        xcoord, U, V, W, ylabel = planes[key]
        title = f"{args.product} {center_txt} | {key}"
        outpath = os.path.join(args.outdir, f"{key}_{args.product}_run{args.running_days:02d}d.{args.ext}")
        plot_three_panel_time_plane(
            t_day, xcoord, U, V, W, ylabel, title, outpath,
            vmins=vmins, vmaxs=vmaxs, cmap=args.cmap
        )

    # --- B) NEW: lat-lon maps at selected altitudes ---
    if int(args.make_maps) == 1:
        alt_list = parse_float_list(args.map_alts)
        if len(alt_list) == 0:
            # reasonable default: three levels near the middle if not provided
            mid = len(alt) // 2
            alt_list = alt[::2]

        day_indices = select_day_indices(prod["t"], args.map_day, bool(args.map_all_days))
        prefix = f"{args.product}_run{args.running_days:02d}d"

        for di in day_indices:
            # plot_latlon_maps_for_day(
            #     prod, lon, lat, alt, di, alt_list,
            #     outdir=args.outdir,
            #     prefix=prefix,
            #     ext=args.ext,
            #     vmins=vmins, vmaxs=vmaxs, cmap=args.cmap,
            #     quiver=bool(args.map_quiver),
            #     quiver_step=args.map_quiver_step,
            # )
            
            plot_latlon_maps_for_day_cartopy(
                prod, lon, lat, alt, di, alt_list,
                outdir=args.outdir,
                prefix=prefix,
                ext=args.ext,
                vmins=vmins, vmaxs=vmaxs, cmap=args.cmap,
                projection="aeqd",      # recommended for 300 km domain
                quiver=bool(args.map_quiver),
                quiver_step=args.map_quiver_step,
            )

    dt0 = datetime.datetime.utcfromtimestamp(float(np.nanmin(prod["t"])))
    dt1 = datetime.datetime.utcfromtimestamp(float(np.nanmax(prod["t"])))
    print("Done.")
    print(f"Product: {args.product}")
    print(f"Daily range: {dt0.strftime('%Y-%m-%d')} to {dt1.strftime('%Y-%m-%d')} ({len(prod['t'])} days)")
    print(f"Output: {args.outdir}")


if __name__ == "__main__":
    main()
    