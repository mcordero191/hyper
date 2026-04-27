#!/usr/bin/env python3
"""
wind_products_plot_dailyfiles_diag.py

Wind workflow (u,v,w) across daily files:
  - raw: daily means
  - mean: running-window mean of daily winds
  - perturbation: raw - mean
  - residual_kinematic: mean (or additional low-pass)

PLUS DIAGNOSTICS (NO momentum residuals):
  - div_h = dudx + dvdy
  - divV  = dudx + dvdy + dwdz
  - continuity diagnostics: RMS profiles, time-altitude planes, maps
  - vertical mass-balance closure:
        <div_h>_A  vs  - d/dz <w>_A
        misfit = <div_h>_A + d/dz <w>_A
  - w climatology diagnostics: w(z) profile + LT–z composite
  - coverage diagnostics: fraction finite

Assumptions:
  - time is epoch seconds
  - winds arrays are (t, lon, lat, alt)
  - lon/lat in degrees
  - alt in km

Cartopy is optional: if not installed, diagnostics maps fall back to plain lon/lat pcolormesh.
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
# Existing plotting helpers
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
# Map plotting (existing + optional Cartopy)
# -----------------------------
def parse_float_list(csv: str):
    if csv is None or csv.strip() == "":
        return []
    return [float(x.strip()) for x in csv.split(",") if x.strip() != ""]


def time_to_yyyymmdd(t_sec: float) -> str:
    return datetime.datetime.utcfromtimestamp(float(t_sec)).strftime("%Y%m%d")


def select_day_indices(t_day: np.ndarray, day: str, all_days: bool):
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
        raise ValueError(
            f"No day matches {target}. Available range: {time_to_yyyymmdd(t_day[0])}..{time_to_yyyymmdd(t_day[-1])}"
        )
    return idx


def _cartopy_available():
    try:
        import cartopy  # noqa
        return True
    except Exception:
        return False


def plot_map_cartopy(lon, lat, field2d, outpath, title, cmap="seismic", vmin=None, vmax=None, projection="aeqd"):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    data_crs = ccrs.PlateCarree()
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

    LON, LAT = np.meshgrid(lon, lat, indexing="ij")
    extent = [float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))]

    fig = plt.figure(figsize=(6.5, 5.2), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(extent, crs=data_crs)
    ax.coastlines(resolution="10m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0.4, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    im = ax.pcolormesh(LON, LAT, field2d, transform=data_crs, shading="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.9)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_map_plain(lon, lat, field2d, outpath, title, cmap="seismic", vmin=None, vmax=None):
    LON, LAT = np.meshgrid(lon, lat, indexing="ij")
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.2), constrained_layout=True)
    im = ax.pcolormesh(LON, LAT, field2d, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation="vertical")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_map(lon, lat, field2d, outpath, title, cmap="seismic", vmin=None, vmax=None, projection="aeqd"):
    if _cartopy_available():
        plot_map_cartopy(lon, lat, field2d, outpath, title, cmap=cmap, vmin=vmin, vmax=vmax, projection=projection)
    else:
        plot_map_plain(lon, lat, field2d, outpath, title, cmap=cmap, vmin=vmin, vmax=vmax)


# -----------------------------
# NEW: Diagnostics core
# -----------------------------
def lonlat_to_xy_m(lon_deg: np.ndarray, lat_deg: np.ndarray, R: float = 6371000.0):
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    lon0 = float(np.nanmean(lon))
    lat0 = float(np.nanmean(lat))
    lam = np.deg2rad(lon); lam0 = np.deg2rad(lon0)
    phi = np.deg2rad(lat); phi0 = np.deg2rad(lat0)
    x = R * np.cos(phi0) * (lam - lam0)
    y = R * (phi - phi0)
    return x, y, lon0, lat0


def interior_mask(nlon: int, nlat: int, margin: int):
    m = int(max(0, margin))
    mask = np.ones((nlon, nlat), dtype=bool)
    if m == 0:
        return mask
    mask[:m, :] = False
    mask[-m:, :] = False
    mask[:, :m] = False
    mask[:, -m:] = False
    return mask


def masked_nanmean_xy(field: np.ndarray, mask2d: np.ndarray):
    # field shape: (t,nlon,nlat,nalt)
    m = mask2d[None, :, :, None]
    return np.nanmean(np.where(m, field, np.nan), axis=(1, 2))  # (t,nalt)


def masked_rms_profile(field: np.ndarray, mask2d: np.ndarray):
    m = mask2d[None, :, :, None]
    return np.sqrt(np.nanmean(np.where(m, field, np.nan) ** 2, axis=(0, 1, 2)))  # (nalt,)


def compute_continuity_diagnostics(prod, lon, lat, alt_km, edge_margin=2):
    """
    prod: dict with keys t,u,v,w. u shape (t,nlon,nlat,nalt)
    Returns dict with divh, divV, dwdz, tz_means, rms_profiles, massbalance.
    """
    u = prod["u"]
    v = prod["v"]
    w = prod["w"]
    t = prod["t"]

    x_m, y_m, lon0, lat0 = lonlat_to_xy_m(lon, lat)
    z_m = np.asarray(alt_km, dtype=float) * 1000.0  # km -> m

    # Derivatives (t,lon,lat,alt)
    dudx = np.gradient(u, x_m, axis=1, edge_order=2)
    dvdy = np.gradient(v, y_m, axis=2, edge_order=2)
    dwdz = np.gradient(w, z_m, axis=3, edge_order=2)

    divh = dudx + dvdy
    divV = divh + dwdz

    mask2d = interior_mask(u.shape[1], u.shape[2], edge_margin)

    # Time-height means (over x,y)
    divV_tz = masked_nanmean_xy(divV, mask2d)
    divh_tz = masked_nanmean_xy(divh, mask2d)
    w_tz    = masked_nanmean_xy(w,    mask2d)

    # RMS profiles (over t,x,y)
    rms_divV = masked_rms_profile(divV, mask2d)
    rms_divh = masked_rms_profile(divh, mask2d)
    rms_w    = masked_rms_profile(w,    mask2d)

    # Vertical mass-balance closure on box-mean quantities
    # misfit(z,t) = <divh> + d/dz <w>
    dwdz_mean = np.gradient(w_tz, z_m, axis=1, edge_order=2)  # (t,nalt) units 1/s
    mb_misfit_tz = divh_tz + dwdz_mean

    return {
        "coords": {"x_m": x_m, "y_m": y_m, "z_m": z_m, "lon0": lon0, "lat0": lat0},
        "mask2d": mask2d,
        "dwdz": dwdz,
        "divh": divh,
        "divV": divV,
        "tz": {"divV": divV_tz, "divh": divh_tz, "w": w_tz, "mb_misfit": mb_misfit_tz},
        "rms_z": {"divV": rms_divV, "divh": rms_divh, "w": rms_w},
    }


def compute_coverage(prod):
    """Coverage fraction over time for u,v,w (per lon,lat,alt)."""
    cov_u = np.mean(np.isfinite(prod["u"]), axis=0)
    cov_v = np.mean(np.isfinite(prod["v"]), axis=0)
    cov_w = np.mean(np.isfinite(prod["w"]), axis=0)
    return {"u": cov_u, "v": cov_v, "w": cov_w}


# -----------------------------
# NEW: Diagnostics plotting
# -----------------------------
def plot_profile_multi(alt_km, series, outpath, title, xlabel):
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 6.6), constrained_layout=True)
    for label, arr in series:
        ax.plot(arr, alt_km, label=label)
    ax.set_ylabel("Altitude [km]")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_profile_mean_spread(alt_km, mean_prof, spread_prof, outpath, title, xlabel, spread_label="±1σ"):
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 6.6), constrained_layout=True)
    ax.plot(mean_prof, alt_km, label="mean")
    ax.plot(mean_prof + spread_prof, alt_km, linestyle="--", label=f"mean {spread_label}")
    ax.plot(mean_prof - spread_prof, alt_km, linestyle="--")
    ax.set_ylabel("Altitude [km]")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_tz_plane(t_sec, alt_km, field_tz, outpath, title, cmap="seismic", vmin=None, vmax=None):
    tday = to_days(t_sec)
    T, Z = np.meshgrid(tday, alt_km, indexing="ij")
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.6), constrained_layout=True)
    im = ax.pcolormesh(T, Z, field_tz, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Time [days since start]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation="vertical")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def compute_local_time_hours(t_sec, lon0_deg):
    """Local solar time approximation: LT = UTC + lon/15."""
    dt = np.array([datetime.datetime.utcfromtimestamp(float(tt)) for tt in t_sec])
    utc_hours = np.array([d.hour + d.minute/60 + d.second/3600 for d in dt], dtype=float)
    lt = (utc_hours + lon0_deg / 15.0) % 24.0
    return lt


def lt_z_composite(field_tz, lt_hours, nbins=24):
    """
    field_tz: (t,nalt)
    lt_hours: (t,) in [0,24)
    Returns: comp (nbins,nalt), counts (nbins,)
    """
    nb = int(nbins)
    edges = np.linspace(0.0, 24.0, nb + 1)
    comp = np.full((nb, field_tz.shape[1]), np.nan)
    counts = np.zeros(nb, dtype=int)

    for b in range(nb):
        m = (lt_hours >= edges[b]) & (lt_hours < edges[b+1])
        counts[b] = int(np.sum(m))
        if counts[b] > 0:
            comp[b, :] = np.nanmean(field_tz[m, :], axis=0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, comp, counts


def plot_lt_z(lt_centers, alt_km, comp, outpath, title, cmap="seismic", vmin=None, vmax=None):
    LT, Z = np.meshgrid(lt_centers, alt_km, indexing="ij")  # (nbin,nalt)
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.6), constrained_layout=True)
    im = ax.pcolormesh(LT, Z, comp, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Local time [hours]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, orientation="vertical")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Wind products + continuity diagnostics from DAILY FILES (u,v,w).")

    p.add_argument("--input-dir", required=True, help="Directory containing daily wind files")
    p.add_argument("--outdir", required=True, help="Output directory for plots")
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

    # maps (existing)
    p.add_argument("--make-maps", type=int, default=1)
    p.add_argument("--map-alts", type=str, default="85,90,95", help="Comma-separated altitude list in km")
    p.add_argument("--map-day", type=str, default=None, help="YYYYMMDD for single day selection (default first day)")
    p.add_argument("--map-all-days", type=int, default=0)
    p.add_argument("--map-quiver", type=int, default=0)
    p.add_argument("--map-quiver-step", type=int, default=2)
    p.add_argument("--map-projection", type=str, default="aeqd", choices=["aeqd", "lcc", "stereo"])

    # diagnostics (new)
    p.add_argument("--make-diagnostics", type=int, default=1, help="1 to compute/plot continuity diagnostics")
    p.add_argument("--diag-edge-margin", type=int, default=2, help="Exclude N grid cells at each lateral boundary")
    p.add_argument("--diag-alts", type=str, default="85,90,95", help="Comma-separated altitudes (km) for diagnostic maps")
    p.add_argument("--diag-day", type=str, default=None, help="YYYYMMDD for diagnostic maps (default first day)")
    p.add_argument("--lt-bins", type=int, default=24, help="Number of local time bins for LT–z composites")
    p.add_argument("--diag-dirname", type=str, default="diagnostics", help="Subdirectory under outdir for diagnostics")

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    diag_dir = os.path.join(args.outdir, args.diag_dirname)
    os.makedirs(diag_dir, exist_ok=True)

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

    # --- Existing A) time planes ---
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

    # --- Existing B) maps of u,v,w ---
    if int(args.make_maps) == 1:
        alt_list = parse_float_list(args.map_alts)
        day_indices = select_day_indices(prod["t"], args.map_day, bool(args.map_all_days))
        prefix = f"{args.product}_run{args.running_days:02d}d"
        proj = args.map_projection

        for di in day_indices:
            day_str = time_to_yyyymmdd(prod["t"][di])
            u_di = prod["u"][di]
            v_di = prod["v"][di]
            w_di = prod["w"][di]
            for akm in alt_list:
                iz = int(np.abs(alt - akm).argmin())
                akm_used = float(alt[iz])

                # 3 panels u,v,w
                fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
                LON, LAT = np.meshgrid(lon, lat, indexing="ij")
                for ax, data, name, vmin, vmax in zip(
                    axs,
                    [u_di[:, :, iz], v_di[:, :, iz], w_di[:, :, iz]],
                    ["u [m/s]", "v [m/s]", "w [m/s]"],
                    vmins, vmaxs
                ):
                    im = ax.pcolormesh(LON, LAT, data, shading="auto", cmap=args.cmap, vmin=vmin, vmax=vmax)
                    ax.set_xlabel("Longitude [deg]")
                    ax.set_ylabel("Latitude [deg]")
                    ax.set_title(name)
                    fig.colorbar(im, ax=ax, orientation="vertical")
                fig.suptitle(f"{prefix} | {day_str} | alt={akm_used:.2f} km", fontsize=12)
                fname = f"map_latlon_{prefix}_h{akm_used:.2f}km_{day_str}.{args.ext}"
                fig.savefig(os.path.join(args.outdir, fname), dpi=150)
                plt.close(fig)

    # --- Diagnostics (Figures 1–6, excluding momentum residuals) ---
    if int(args.make_diagnostics) == 1:
        d = compute_continuity_diagnostics(
            prod, lon, lat, alt, edge_margin=args.diag_edge_margin
        )
        cov = compute_coverage(prod)

        prefix = f"{args.product}_run{args.running_days:02d}d"
        margin = int(args.diag_edge_margin)

        # Figure 1: RMS wind profiles
        plot_profile_multi(
            alt,
            [("RMS(u) [m/s]", np.sqrt(np.nanmean(prod["u"]**2, axis=(0,1,2)))),
             ("RMS(v) [m/s]", np.sqrt(np.nanmean(prod["v"]**2, axis=(0,1,2)))),
             ("RMS(w) [m/s]", d["rms_z"]["w"])],
            os.path.join(diag_dir, f"diag_rms_winds_{prefix}_margin{margin}.{args.ext}"),
            title=f"Wind magnitude profiles ({prefix}, margin={margin})",
            xlabel="RMS [m/s]"
        )

        # Figure 2a: w profile mean ± std (domain mean over x,y,t; spread over x,y,t)
        w_flat = prod["w"].reshape(prod["w"].shape[0], -1, prod["w"].shape[-1])  # (t, nxy, z)
        w_mean_z = np.nanmean(w_flat, axis=(0,1))
        w_std_z  = np.nanstd(w_flat, axis=(0,1))
        plot_profile_mean_spread(
            alt, w_mean_z, w_std_z,
            os.path.join(diag_dir, f"diag_w_profile_{prefix}.{args.ext}"),
            title=f"Vertical velocity profile ({prefix})",
            xlabel="w [m/s]",
            spread_label="±1σ"
        )

        # Figure 2b: LT–z composite of box-mean w
        lt = compute_local_time_hours(prod["t"], d["coords"]["lon0"])
        lt_centers, w_lt, counts = lt_z_composite(d["tz"]["w"], lt, nbins=args.lt_bins)
        plot_lt_z(
            lt_centers, alt, w_lt,
            os.path.join(diag_dir, f"diag_w_LTz_{prefix}_nb{args.lt_bins}.{args.ext}"),
            title=f"Local-time composite of <w> (box-mean), {prefix}",
            cmap=args.cmap
        )

        # Figure 3: continuity residual diagnostics
        plot_profile_multi(
            alt,
            [("RMS(divV) [1/s]", d["rms_z"]["divV"]),
             ("RMS(div_h) [1/s]", d["rms_z"]["divh"])],
            os.path.join(diag_dir, f"diag_rms_divergence_{prefix}_margin{margin}.{args.ext}"),
            title=f"Divergence diagnostics ({prefix}, margin={margin})",
            xlabel="RMS [1/s]"
        )
        plot_tz_plane(
            prod["t"], alt, d["tz"]["divV"],
            os.path.join(diag_dir, f"diag_tz_divV_{prefix}_margin{margin}.{args.ext}"),
            title=f"Time–altitude: box-mean divV ({prefix}, margin={margin})",
            cmap=args.cmap
        )

        # Figure 4: mass-balance closure
        # profile of time-mean terms: <divh> and -d<w>/dz
        z_m = d["coords"]["z_m"]
        dwdz_mean = np.gradient(d["tz"]["w"], z_m, axis=1, edge_order=2)  # (t,z)
        prof_divh = np.nanmean(d["tz"]["divh"], axis=0)
        prof_neg_dwdz = np.nanmean(-dwdz_mean, axis=0)
        prof_misfit = np.nanmean(d["tz"]["mb_misfit"], axis=0)

        plot_profile_multi(
            alt,
            [("<div_h> [1/s]", prof_divh),
             ("-d<w>/dz [1/s]", prof_neg_dwdz),
             ("misfit [1/s]", prof_misfit)],
            os.path.join(diag_dir, f"diag_massbalance_profile_{prefix}_margin{margin}.{args.ext}"),
            title=f"Vertical mass-balance closure ({prefix}, margin={margin})",
            xlabel="[1/s]"
        )
        plot_tz_plane(
            prod["t"], alt, d["tz"]["mb_misfit"],
            os.path.join(diag_dir, f"diag_tz_massbalance_misfit_{prefix}_margin{margin}.{args.ext}"),
            title=f"Time–altitude: mass-balance misfit ({prefix}, margin={margin})",
            cmap=args.cmap
        )

        # Figure 1 coverage maps + Figure 3/5 maps for one selected day
        alt_list = parse_float_list(args.map_alts)
        day_indices = select_day_indices(prod["t"], args.map_day, all_days=False)
        di = day_indices[0]
        day_str = time_to_yyyymmdd(prod["t"][di])
        proj = args.map_projection

        for akm in alt_list:
            iz = int(np.abs(alt - akm).argmin())
            akm_used = float(alt[iz])

            # Coverage (w)
            covw2d = cov["w"][:, :, iz]
            plot_map(
                lon, lat, covw2d,
                os.path.join(diag_dir, f"diag_coverage_w_{prefix}_h{akm_used:.2f}km_{day_str}.{args.ext}"),
                title=f"Coverage fraction (w) | {prefix} | {day_str} | {akm_used:.2f} km",
                cmap="viridis", vmin=0.0, vmax=1.0, projection=proj
            )

            # divV map
            divV2d = d["divV"][di, :, :, iz]
            plot_map(
                lon, lat, divV2d,
                os.path.join(diag_dir, f"diag_map_divV_{prefix}_h{akm_used:.2f}km_{day_str}.{args.ext}"),
                title=f"divV | {prefix} | {day_str} | {akm_used:.2f} km",
                cmap=args.cmap, projection=proj
            )

            # divh map
            divh2d = d["divh"][di, :, :, iz]
            plot_map(
                lon, lat, divh2d,
                os.path.join(diag_dir, f"diag_map_divh_{prefix}_h{akm_used:.2f}km_{day_str}.{args.ext}"),
                title=f"div_h | {prefix} | {day_str} | {akm_used:.2f} km",
                cmap=args.cmap, projection=proj
            )

        # Sensitivity figure hooks (Figure 6) are generated by re-running with different args:
        # - product raw vs mean
        # - diag-edge-margin 0 vs 2
        # No extra code needed here.

    dt0 = datetime.datetime.utcfromtimestamp(float(np.nanmin(prod["t"])))
    dt1 = datetime.datetime.utcfromtimestamp(float(np.nanmax(prod["t"])))
    print("Done.")
    print(f"Product: {args.product}")
    print(f"Daily range: {dt0.strftime('%Y-%m-%d')} to {dt1.strftime('%Y-%m-%d')} ({len(prod['t'])} days)")
    print(f"Output: {args.outdir}")
    if int(args.make_diagnostics) == 1:
        print(f"Diagnostics: {diag_dir}")


if __name__ == "__main__":
    main()