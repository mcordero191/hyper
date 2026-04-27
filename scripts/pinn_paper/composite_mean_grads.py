from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from georeference.geo_coordinates import lla2enu
# ============================================================
# Configuration containers
# ============================================================

@dataclass
class WindowConfig:
    time_hours: float = 2.0
    alt_km: float = 2.0
    kind: str = "rect"  # "rect" or "gaussian"
    sigma_time_hours: Optional[float] = None
    sigma_alt_km: Optional[float] = None


@dataclass
class PlotConfig:
    cmap: str = "RdBu_r"
    figsize: Tuple[float, float] = (12.5, 13.5)
    dpi: int = 180
    fontsize: int = 11
    title_fontsize: int = 13
    label_fontsize: int = 11
    tick_fontsize: int = 10
    output: Optional[str] = None
    center_time: Optional[np.datetime64] = None
    time_range_hours: float = 8.0


# ============================================================
# I/O
# ============================================================

def load_wind_dataset(ncfile: str) -> xr.Dataset:
    ds = xr.open_dataset(ncfile)

    required = ["u", "v", "w", "u_std", "v_std", "w_std",
                "time", "longitude", "latitude", "altitude"]
    missing = [v for v in required if v not in ds.variables and v not in ds.coords]
    if missing:
        raise KeyError(f"Missing required variables/coords: {missing}")

    return ds


# ============================================================
# Coordinates and selection
# ============================================================

def domain_center_latlon(ds: xr.Dataset) -> Tuple[float, float]:
    lon_vals = ds["longitude"].values
    lat_vals = ds["latitude"].values
    center_lon = float(lon_vals[len(lon_vals) // 2])
    center_lat = float(lat_vals[len(lat_vals) // 2])
    return center_lat, center_lon


def find_nearest_grid_point(
    ds: xr.Dataset,
    target_lat: Optional[float] = None,
    target_lon: Optional[float] = None,
) -> Dict[str, float]:
    if target_lat is None or target_lon is None:
        target_lat, target_lon = domain_center_latlon(ds)

    lat_vals = ds["latitude"].values
    lon_vals = ds["longitude"].values

    ilat = int(np.argmin(np.abs(lat_vals - target_lat)))
    ilon = int(np.argmin(np.abs(lon_vals - target_lon)))

    return {
        "ilat": ilat,
        "ilon": ilon,
        "latitude": float(lat_vals[ilat]),
        "longitude": float(lon_vals[ilon]),
    }


def geodetic_to_ecef(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    alt_km: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    WGS84 geodetic to ECEF. Output in meters.
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    h = alt_km * 1000.0

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = a / np.sqrt(1.0 - e2 * sin_lat**2)

    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1.0 - e2) + h) * sin_lat
    return x, y, z


def build_ecef_grid(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    alt = ds["altitude"].values
    
    lon_ref = np.nanmean(lon)
    lat_ref = np.nanmean(lat)
    alt_ref = np.nanmean(alt)

    lon3, lat3, alt3 = np.meshgrid(lon, lat, alt, indexing="ij")
    # x, y, z = geodetic_to_ecef(lon3, lat3, alt3)
    x, y, _ = lla2enu(lat3, lon3, alt3, lat_ref, lon_ref, alt_ref, units="m")
    z = alt3*1e3 #meters
    
    return x, y, z


# ============================================================
# Mask handling
# ============================================================

def apply_mask_to_data(ds: xr.Dataset) -> xr.Dataset:
    if "mask" not in ds:
        return ds

    out = ds.copy()
    valid = xr.where(ds["mask"] == 1, 1.0, np.nan)

    for v in ["u", "v", "w", "u_std", "v_std", "w_std"]:
        out[v] = out[v] * valid

    return out


# ============================================================
# Local background window
# ============================================================

def build_1d_weights(
    offsets: np.ndarray,
    half_width: float,
    kind: str = "rect",
    sigma: Optional[float] = None,
) -> np.ndarray:
    if kind not in ("rect", "gaussian"):
        raise ValueError("kind must be 'rect' or 'gaussian'")

    if kind == "rect":
        w = np.where(np.abs(offsets) <= half_width, 1.0, 0.0)
    else:
        if sigma is None:
            sigma = max(half_width / 2.0, 1e-6)
        w = np.exp(-0.5 * (offsets / sigma) ** 2)
        w[np.abs(offsets) > half_width] = 0.0

    return w


def weighted_local_background_2d(
    field_tz: np.ndarray,
    sigma_tz: Optional[np.ndarray],
    time_vals: np.ndarray,
    alt_vals: np.ndarray,
    config: WindowConfig,
    propagate_uncertainty: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    
    nt, nz = field_tz.shape
    mean = np.full((nt, nz), np.nan)
    mean_sigma = np.full((nt, nz), np.nan) if propagate_uncertainty else None

    t_hours = (time_vals - time_vals[0]) / np.timedelta64(1, "h")
    z_km = alt_vals.astype(float)

    for it in range(nt):
        wt = build_1d_weights(
            t_hours - t_hours[it],
            half_width=config.time_hours / 2.0,
            kind=config.kind,
            sigma=config.sigma_time_hours,
        )
        if np.all(wt == 0):
            continue

        for iz in range(nz):
            wz = build_1d_weights(
                z_km - z_km[iz],
                half_width=config.alt_km / 2.0,
                kind=config.kind,
                sigma=config.sigma_alt_km,
            )
            if np.all(wz == 0):
                continue

            W = np.outer(wt, wz)
            valid = np.isfinite(field_tz)
            Wv = np.where(valid, W, 0.0)
            denom = np.sum(Wv)

            if denom <= 0:
                continue

            mean[it, iz] = np.nansum(field_tz * Wv) / denom

            if propagate_uncertainty and sigma_tz is not None:
                wn = Wv / denom
                wn = np.where(np.isfinite(sigma_tz), wn, 0.0)
                mean_sigma[it, iz] = np.sqrt(np.nansum((wn ** 2) * (sigma_tz ** 2)))

    return mean, mean_sigma


# ============================================================
# Residuals
# ============================================================

def compute_residual(
    field: np.ndarray,
    background: np.ndarray,
    sigma_field: Optional[np.ndarray] = None,
    sigma_background: Optional[np.ndarray] = None,
    propagate_uncertainty: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    
    residual = field - background

    if not propagate_uncertainty or sigma_field is None or sigma_background is None:
        return residual, None

    sigma_res = np.sqrt(sigma_field**2 + sigma_background**2)
    return residual, sigma_res


# ============================================================
# Gradients
# ============================================================

def finite_difference(
    f_prev: np.ndarray,
    f_next: np.ndarray,
    x_prev: np.ndarray,
    x_next: np.ndarray,
    sigma_prev: Optional[np.ndarray] = None,
    sigma_next: Optional[np.ndarray] = None,
    propagate_uncertainty: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    dx = x_next - x_prev
    grad = np.where(np.abs(dx) > 0, (f_next - f_prev) / dx, np.nan)

    if not propagate_uncertainty or sigma_prev is None or sigma_next is None:
        return grad, None

    sigma_grad = np.where(
        np.abs(dx) > 0,
        np.sqrt(sigma_prev**2 + sigma_next**2) / np.abs(dx),
        np.nan,
    )
    return grad, sigma_grad

def compute_gradients_at_location(
    fields: Dict[str, np.ndarray],
    sigmas: Dict[str, Optional[np.ndarray]],
    x_ecef: np.ndarray,
    y_ecef: np.ndarray,
    z_ecef: np.ndarray,
    ilon: int,
    ilat: int,
    include_w_gradients: bool = False,
    propagate_uncertainty: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute gradients at the selected location from arbitrary 4D fields.
    Expected field shapes: (time, longitude, latitude, altitude)
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}

    sample = fields["u"]
    nt, nlon, nlat, nz = sample.shape

    def one_component(var: str) -> Dict[str, np.ndarray]:
        f = fields[var]
        sf = sigmas[var] if propagate_uncertainty else None

        # x-gradient
        if ilon == 0:
            f_prev = f[:, ilon, ilat, :]
            f_next = f[:, ilon + 1, ilat, :]
            x_prev = x_ecef[ilon, ilat, :]
            x_next = x_ecef[ilon + 1, ilat, :]
            s_prev = sf[:, ilon, ilat, :] if sf is not None else None
            s_next = sf[:, ilon + 1, ilat, :] if sf is not None else None
        elif ilon == nlon - 1:
            f_prev = f[:, ilon - 1, ilat, :]
            f_next = f[:, ilon, ilat, :]
            x_prev = x_ecef[ilon - 1, ilat, :]
            x_next = x_ecef[ilon, ilat, :]
            s_prev = sf[:, ilon - 1, ilat, :] if sf is not None else None
            s_next = sf[:, ilon, ilat, :] if sf is not None else None
        else:
            f_prev = f[:, ilon - 1, ilat, :]
            f_next = f[:, ilon + 1, ilat, :]
            x_prev = x_ecef[ilon - 1, ilat, :]
            x_next = x_ecef[ilon + 1, ilat, :]
            s_prev = sf[:, ilon - 1, ilat, :] if sf is not None else None
            s_next = sf[:, ilon + 1, ilat, :] if sf is not None else None

        gx, sgx = finite_difference(
            f_prev, f_next,
            x_prev[np.newaxis, :], x_next[np.newaxis, :],
            s_prev, s_next, propagate_uncertainty
        )

        # y-gradient
        if ilat == 0:
            f_prev = f[:, ilon, ilat, :]
            f_next = f[:, ilon, ilat + 1, :]
            y_prev = y_ecef[ilon, ilat, :]
            y_next = y_ecef[ilon, ilat + 1, :]
            s_prev = sf[:, ilon, ilat, :] if sf is not None else None
            s_next = sf[:, ilon, ilat + 1, :] if sf is not None else None
        elif ilat == nlat - 1:
            f_prev = f[:, ilon, ilat - 1, :]
            f_next = f[:, ilon, ilat, :]
            y_prev = y_ecef[ilon, ilat - 1, :]
            y_next = y_ecef[ilon, ilat, :]
            s_prev = sf[:, ilon, ilat - 1, :] if sf is not None else None
            s_next = sf[:, ilon, ilat, :] if sf is not None else None
        else:
            f_prev = f[:, ilon, ilat - 1, :]
            f_next = f[:, ilon, ilat + 1, :]
            y_prev = y_ecef[ilon, ilat - 1, :]
            y_next = y_ecef[ilon, ilat + 1, :]
            s_prev = sf[:, ilon, ilat - 1, :] if sf is not None else None
            s_next = sf[:, ilon, ilat + 1, :] if sf is not None else None

        gy, sgy = finite_difference(
            f_prev, f_next,
            y_prev[np.newaxis, :], y_next[np.newaxis, :],
            s_prev, s_next, propagate_uncertainty
        )

        # z-gradient
        gz = np.full((nt, nz), np.nan)
        sgz = np.full((nt, nz), np.nan) if propagate_uncertainty else None

        col = f[:, ilon, ilat, :]
        scol = sf[:, ilon, ilat, :] if sf is not None else None
        zcol = z_ecef[ilon, ilat, :]

        for iz in range(nz):
            if iz == 0:
                f_prev = col[:, iz]
                f_next = col[:, iz + 1]
                z_prev = zcol[iz]
                z_next = zcol[iz + 1]
                s_prev = scol[:, iz] if scol is not None else None
                s_next = scol[:, iz + 1] if scol is not None else None
            elif iz == nz - 1:
                f_prev = col[:, iz - 1]
                f_next = col[:, iz]
                z_prev = zcol[iz - 1]
                z_next = zcol[iz]
                s_prev = scol[:, iz - 1] if scol is not None else None
                s_next = scol[:, iz] if scol is not None else None
            else:
                f_prev = col[:, iz - 1]
                f_next = col[:, iz + 1]
                z_prev = zcol[iz - 1]
                z_next = zcol[iz + 1]
                s_prev = scol[:, iz - 1] if scol is not None else None
                s_next = scol[:, iz + 1] if scol is not None else None

            gtmp, stmp = finite_difference(
                f_prev, f_next, z_prev, z_next,
                s_prev, s_next, propagate_uncertainty
            )
            gz[:, iz] = gtmp
            if sgz is not None:
                sgz[:, iz] = stmp

        # convert from per meter to per km
        gx *= 1000.0
        gy *= 1000.0
        gz *= 1000.0
        if sgx is not None:
            sgx *= 1000.0
            sgy *= 1000.0
            sgz *= 1000.0

        result = {"gx": gx, "gy": gy, "gz": gz}
        if propagate_uncertainty:
            result["sgx"] = sgx
            result["sgy"] = sgy
            result["sgz"] = sgz
        return result

    out["u"] = one_component("u")
    out["v"] = one_component("v")
    if include_w_gradients:
        out["w"] = one_component("w")

    return out

def analyze_local_column(
    ncfile: str,
    target_lat: Optional[float] = None,
    target_lon: Optional[float] = None,
    window: WindowConfig = WindowConfig(),
    include_w: bool = False,
    propagate_uncertainty: bool = False,
) -> Dict[str, object]:
    ds = load_wind_dataset(ncfile)
    ds = apply_mask_to_data(ds)

    loc = find_nearest_grid_point(ds, target_lat=target_lat, target_lon=target_lon)
    ilon = loc["ilon"]
    ilat = loc["ilat"]

    x_ecef, y_ecef, z_ecef = build_ecef_grid(ds)

    time_vals = ds["time"].values
    alt_vals = ds["altitude"].values

    nlon = ds.sizes["longitude"]
    nlat = ds.sizes["latitude"]

    def needed_indices(i: int, n: int) -> List[int]:
        idx = {i}
        if i > 0:
            idx.add(i - 1)
        if i < n - 1:
            idx.add(i + 1)
        return sorted(idx)

    lon_needed = needed_indices(ilon, nlon)
    lat_needed = needed_indices(ilat, nlat)

    # --------------------------------------------------------
    # Compute local background + residual only for needed columns
    # --------------------------------------------------------
    def compute_background_and_residual(var: str, svar: str):
        full = ds[var].values
        sfull = ds[svar].values if propagate_uncertainty else None

        residual_sub = {}
        sigma_res_sub = {}
        background_center = None
        sigma_background_center = None
        residual_center = None
        sigma_residual_center = None

        for jlon in lon_needed:
            for jlat in lat_needed:
                col = full[:, jlon, jlat, :]
                scol = sfull[:, jlon, jlat, :] if sfull is not None else None

                bg, sbg = weighted_local_background_2d(
                    col, scol, time_vals, alt_vals, window, propagate_uncertainty
                )
                res, sres = compute_residual(
                    col, bg, scol, sbg, propagate_uncertainty
                )

                residual_sub[(jlon, jlat)] = res
                sigma_res_sub[(jlon, jlat)] = sres

                if jlon == ilon and jlat == ilat:
                    background_center = bg
                    sigma_background_center = sbg
                    residual_center = res
                    sigma_residual_center = sres

        return (
            residual_sub,
            sigma_res_sub,
            background_center,
            sigma_background_center,
            residual_center,
            sigma_residual_center,
        )

    u_res_sub, su_res_sub, u0, su0, ur, sur = compute_background_and_residual("u", "u_std")
    v_res_sub, sv_res_sub, v0, sv0, vr, svr = compute_background_and_residual("v", "v_std")

    if include_w:
        w_res_sub, sw_res_sub, w0, sw0, wr, swr = compute_background_and_residual("w", "w_std")
    else:
        w_res_sub, sw_res_sub, w0, sw0, wr, swr = None, None, None, None, None, None

    # --------------------------------------------------------
    # Build small local residual arrays only for needed stencil
    # --------------------------------------------------------
    lon_map = {j: i for i, j in enumerate(lon_needed)}
    lat_map = {j: i for i, j in enumerate(lat_needed)}

    nt = ds.sizes["time"]
    nz = ds.sizes["altitude"]

    def make_local_field(res_sub):
        arr = np.full((nt, len(lon_needed), len(lat_needed), nz), np.nan, dtype=float)
        for (jlon, jlat), val in res_sub.items():
            arr[:, lon_map[jlon], lat_map[jlat], :] = val
        return arr

    def make_local_sigma(sig_sub):
        if not propagate_uncertainty:
            return None
        arr = np.full((nt, len(lon_needed), len(lat_needed), nz), np.nan, dtype=float)
        for (jlon, jlat), val in sig_sub.items():
            if val is not None:
                arr[:, lon_map[jlon], lat_map[jlat], :] = val
        return arr

    fields_for_grad = {
        "u": make_local_field(u_res_sub),
        "v": make_local_field(v_res_sub),
    }
    sigmas_for_grad = {
        "u": make_local_sigma(su_res_sub),
        "v": make_local_sigma(sv_res_sub),
    }

    if include_w:
        fields_for_grad["w"] = make_local_field(w_res_sub)
        sigmas_for_grad["w"] = make_local_sigma(sw_res_sub)

    # local coordinate stencil
    x_local = x_ecef[np.ix_(lon_needed, lat_needed, np.arange(x_ecef.shape[2]))]
    y_local = y_ecef[np.ix_(lon_needed, lat_needed, np.arange(y_ecef.shape[2]))]
    z_local = z_ecef[np.ix_(lon_needed, lat_needed, np.arange(z_ecef.shape[2]))]

    ilon_local = lon_map[ilon]
    ilat_local = lat_map[ilat]

    grads = compute_gradients_at_location(
        fields_for_grad,
        sigmas_for_grad,
        x_local, y_local, z_local,
        ilon=ilon_local,
        ilat=ilat_local,
        include_w_gradients=include_w,
        propagate_uncertainty=propagate_uncertainty,
    )

    return {
        "location": loc,
        "time": time_vals,
        "altitude": alt_vals,
        "include_w": include_w,
        "mean": {"u": u0, "v": v0, "w": w0, "su": su0, "sv": sv0, "sw": sw0},
        "residual": {"u": ur, "v": vr, "w": wr, "su": sur, "sv": svr, "sw": swr},
        "gradient": grads,
    }
    
def analyze_local_column_slow(
    ncfile: str,
    target_lat: Optional[float] = None,
    target_lon: Optional[float] = None,
    window: WindowConfig = WindowConfig(),
    include_w: bool = False,
    propagate_uncertainty: bool = False,
) -> Dict[str, object]:
    ds = load_wind_dataset(ncfile)
    ds = apply_mask_to_data(ds)

    loc = find_nearest_grid_point(ds, target_lat=target_lat, target_lon=target_lon)
    ilon = loc["ilon"]
    ilat = loc["ilat"]

    x_ecef, y_ecef, z_ecef = build_ecef_grid(ds)

    time_vals = ds["time"].values
    alt_vals = ds["altitude"].values

    # local columns for plotting
    u_col = ds["u"].values[:, ilon, ilat, :]
    v_col = ds["v"].values[:, ilon, ilat, :]
    w_col = ds["w"].values[:, ilon, ilat, :] if include_w else None

    su_col = ds["u_std"].values[:, ilon, ilat, :] if propagate_uncertainty else None
    sv_col = ds["v_std"].values[:, ilon, ilat, :] if propagate_uncertainty else None
    sw_col = ds["w_std"].values[:, ilon, ilat, :] if (propagate_uncertainty and include_w) else None

    # local background for plotting
    u0, su0 = weighted_local_background_2d(
        u_col, su_col, time_vals, alt_vals, window, propagate_uncertainty
    )
    v0, sv0 = weighted_local_background_2d(
        v_col, sv_col, time_vals, alt_vals, window, propagate_uncertainty
    )

    if include_w and w_col is not None:
        w0, sw0 = weighted_local_background_2d(
            w_col, sw_col, time_vals, alt_vals, window, propagate_uncertainty
        )
    else:
        w0, sw0 = None, None

    ur, sur = compute_residual(u_col, u0, su_col, su0, propagate_uncertainty)
    vr, svr = compute_residual(v_col, v0, sv_col, sv0, propagate_uncertainty)

    if include_w and w_col is not None and w0 is not None:
        wr, swr = compute_residual(w_col, w0, sw_col, sw0, propagate_uncertainty)
    else:
        wr, swr = None, None

    # --------------------------------------------------------
    # Build full residual fields so gradients are taken from
    # residuals, not from the original winds
    # --------------------------------------------------------
    u_full = ds["u"].values
    v_full = ds["v"].values
    w_full = ds["w"].values if include_w else None

    su_full = ds["u_std"].values if propagate_uncertainty else None
    sv_full = ds["v_std"].values if propagate_uncertainty else None
    sw_full = ds["w_std"].values if (propagate_uncertainty and include_w) else None

    ntime = ds.sizes["time"]
    nlon = ds.sizes["longitude"]
    nlat = ds.sizes["latitude"]
    nalt = ds.sizes["altitude"]

    u0_full = np.full((ntime, nlon, nlat, nalt), np.nan, dtype=float)
    v0_full = np.full((ntime, nlon, nlat, nalt), np.nan, dtype=float)
    w0_full = np.full((ntime, nlon, nlat, nalt), np.nan, dtype=float) if include_w else None

    su0_full = np.full((ntime, nlon, nlat, nalt), np.nan, dtype=float) if propagate_uncertainty else None
    sv0_full = np.full((ntime, nlon, nlat, nalt), np.nan, dtype=float) if propagate_uncertainty else None
    sw0_full = np.full((ntime, nlon, nlat, nalt), np.nan, dtype=float) if (propagate_uncertainty and include_w) else None

    for jlon in range(nlon):
        for jlat in range(nlat):
            u_bg, su_bg = weighted_local_background_2d(
                u_full[:, jlon, jlat, :],
                su_full[:, jlon, jlat, :] if propagate_uncertainty else None,
                time_vals, alt_vals, window, propagate_uncertainty
            )
            v_bg, sv_bg = weighted_local_background_2d(
                v_full[:, jlon, jlat, :],
                sv_full[:, jlon, jlat, :] if propagate_uncertainty else None,
                time_vals, alt_vals, window, propagate_uncertainty
            )

            u0_full[:, jlon, jlat, :] = u_bg
            v0_full[:, jlon, jlat, :] = v_bg

            if propagate_uncertainty:
                su0_full[:, jlon, jlat, :] = su_bg
                sv0_full[:, jlon, jlat, :] = sv_bg

            if include_w:
                w_bg, sw_bg = weighted_local_background_2d(
                    w_full[:, jlon, jlat, :],
                    sw_full[:, jlon, jlat, :] if propagate_uncertainty else None,
                    time_vals, alt_vals, window, propagate_uncertainty
                )
                w0_full[:, jlon, jlat, :] = w_bg
                if propagate_uncertainty:
                    sw0_full[:, jlon, jlat, :] = sw_bg

    ur_full = u_full - u0_full
    vr_full = v_full - v0_full
    wr_full = w_full - w0_full if include_w else None

    sur_full = np.sqrt(su_full**2 + su0_full**2) if propagate_uncertainty else None
    svr_full = np.sqrt(sv_full**2 + sv0_full**2) if propagate_uncertainty else None
    swr_full = np.sqrt(sw_full**2 + sw0_full**2) if (propagate_uncertainty and include_w) else None

    fields_for_grad = {
        "u": ur_full,
        "v": vr_full,
    }
    sigmas_for_grad = {
        "u": sur_full,
        "v": svr_full,
    }

    if include_w:
        fields_for_grad["w"] = wr_full
        sigmas_for_grad["w"] = swr_full

    grads = compute_gradients_at_location(
        fields_for_grad,
        sigmas_for_grad,
        x_ecef, y_ecef, z_ecef,
        ilon=ilon, ilat=ilat,
        include_w_gradients=include_w,
        propagate_uncertainty=propagate_uncertainty,
    )

    return {
        "location": loc,
        "time": time_vals,
        "altitude": alt_vals,
        "include_w": include_w,
        "mean": {"u": u0, "v": v0, "w": w0, "su": su0, "sv": sv0, "sw": sw0},
        "residual": {"u": ur, "v": vr, "w": wr, "su": sur, "sv": svr, "sw": swr},
        "gradient": grads,
    }
    
def crop_time_indices(
    time_vals: np.ndarray,
    center_time: Optional[np.datetime64],
    time_range_hours: float,
) -> np.ndarray:
    if center_time is None:
        return np.ones(time_vals.shape, dtype=bool)

    half = np.timedelta64(int(round(time_range_hours * 3600)), "s") / 2
    t0 = center_time - half
    t1 = center_time + half
    return (time_vals >= t0) & (time_vals <= t1)


def robust_symmetric_limits(data_list: List[np.ndarray], percentile: float = 100) -> Tuple[float, float]:
    vals = []
    for d in data_list:
        if d is None:
            continue
        finite = d[np.isfinite(d)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return -1.0, 1.0
    vals = np.concatenate(vals)
    vmax = 1.5*np.nanpercentile(np.abs(vals), percentile)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    return -vmax, vmax


def setup_mpl_style(plot_cfg: PlotConfig) -> None:
    plt.rcParams.update({
        "figure.dpi": plot_cfg.dpi,
        "savefig.dpi": plot_cfg.dpi,
        "savefig.bbox": "tight",

        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "mathtext.default": "regular",

        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 8.8,

        "axes.linewidth": 0.8,
        "axes.titlepad": 5.0,
        "axes.labelpad": 4.0,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,

        "xtick.top": True,
        "ytick.right": True,

        "axes.grid": False,
    })

# ============================================================
# Plotting
# ============================================================

def plot_composite(
    result: Dict[str, object],
    plot_cfg: PlotConfig,
) -> Tuple[plt.Figure, np.ndarray]:
    setup_mpl_style(plot_cfg)

    time_vals = result["time"]
    alt_vals = result["altitude"]
    include_w = result["include_w"]

    keep = crop_time_indices(time_vals, plot_cfg.center_time, plot_cfg.time_range_hours)
    tplot = time_vals[keep]

    mean = result["mean"]
    residual = result["residual"]
    grad = result["gradient"]

    columns = ["u", "v"] + (["w"] if include_w else [])
    col_titles = {
        "u": "Zonal wind",
        "v": "Meridional wind",
        "w": "Vertical wind",
    }

    row_titles = [
        # "Background wind",
        "Residual Winds",
        "Gradient in Lon",
        "Gradient in Lat",
        "Gradient in Alt",
    ]

    units = [
        # "m/s",
        "m/s",
        "m/s/km",
        "m/s/km",
        "m/s/km",
    ]

    row_data = [
        # [mean[c][keep, :] if mean[c] is not None else None for c in columns],
        [residual[c][keep, :] if residual[c] is not None else None for c in columns],
        [grad[c]["gx"][keep, :] if c in grad else None for c in columns],
        [grad[c]["gy"][keep, :] if c in grad else None for c in columns],
        [grad[c]["gz"][keep, :] if c in grad else None for c in columns],
    ]

    nrows = 4
    ncols = len(columns)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=plot_cfg.figsize,
        sharex=True,
        sharey=True,
        constrained_layout=False
    )
    if ncols == 1:
        axes = np.array(axes).reshape(nrows, 1)

    plt.subplots_adjust(
        left=0.08, right=0.90, top=0.93, bottom=0.08,
        hspace=0.16, wspace=0.08
    )

    tnum = mdates.date2num(tplot.astype("datetime64[ms]").astype(object))

    row_limits = [
        # robust_symmetric_limits(row_data[0]),
        # robust_symmetric_limits(row_data[1]),
        # robust_symmetric_limits(row_data[2]),
        # robust_symmetric_limits(row_data[3]),
        # robust_symmetric_limits(row_data[4]),
        # [-100, 100],
        [-30, 30],
        [-0.3, 0.3],
        [-0.3, 0.3],
        [-30, 30],
    ]

    row_pcms = []

    for irow in range(nrows):
        pcm_ref = None
        for icol, comp in enumerate(columns):
            ax = axes[irow, icol]
            data = row_data[irow][icol]
            vmin, vmax = row_limits[irow]

            pcm = ax.pcolormesh(
                tnum, alt_vals, data.T,
                shading="auto",
                cmap=plot_cfg.cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if pcm_ref is None:
                pcm_ref = pcm

            if irow == 0:
                ax.set_title(col_titles[comp])

            if icol == 0:
                ax.set_ylabel(f"{row_titles[irow]}\nAltitude (km)")
            else:
                ax.set_ylabel("")

            ax.minorticks_on()
            ax.grid(False)

        row_pcms.append(pcm_ref)

    # Shared colorbar per row
    for irow in range(nrows):
        cbar = fig.colorbar(
            row_pcms[irow],
            ax=axes[irow, :],
            orientation="vertical",
            fraction=0.025,
            pad=0.015
        )
        cbar.set_label(units[irow])

    # Time formatting UTC
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = ['%Y', '%b', '%d', '%H:%M', '%H:%M', '%S']
    formatter.zero_formats = [''] * 6
    formatter.offset_formats = ['', '%Y', '%Y-%b', '%Y-%m-%d', '%Y-%m-%d UTC', '%Y-%m-%d %H:%M UTC']

    for ax in axes[-1, :]:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel("Time (UTC)")

    loc = result["location"]
    fig.suptitle(
        f"Wind diagnostics at lat = {loc['latitude']:.2f}°, lon = {loc['longitude']:.2f}°",
        y=0.98
    )

    return fig, axes


# ============================================================
# CLI
# ============================================================

def parse_datetime_utc(s: Optional[str]) -> Optional[np.datetime64]:
    if s is None:
        return None
    # expected format: YYYY-MM-DD HH:MM:SS
    return np.datetime64(s.replace("T", " "))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot local wind diagnostics from 4D NetCDF winds."
    )
    path = "/Users/radar/Data/IAP/SIMONe/Norway/VorTex/"
    # path = "/Users/radar/Data/IAP/SIMONe/Norway/VorTex2/"
    subdir = "hVV_noNul05.256_lr5.0e-04lf0do0ur1.0e-08r1.0e-02Validated"
    subdir = "hVVl05.256_lr5.0e-04lf0do0ur1.0e-08r1.0e-02Validated"
    file ="winds/winds3D_20230323_000000_v1.4.0.nc"
    
    filename = os.path.join(path, subdir, file)
    
    p.add_argument("--filename", type=str, default=filename,
                   help="Full path to NetCDF wind file.")
    p.add_argument("--lat", type=float, default=68.5, help="Target latitude in degrees.")
    p.add_argument("--lon", type=float, default=14, help="Target longitude in degrees.")

    p.add_argument("--time-window-hours", type=float, default=4.0,
                   help="Centered smoothing window in time [hours].")
    p.add_argument("--alt-window-km", type=float, default=2.0,
                   help="Centered smoothing window in altitude [km].")

    p.add_argument("--window-kind", type=str, default="gaussian", choices=["rect", "gaussian"],
                   help="Smoothing window type.")
    p.add_argument("--sigma-time-hours", type=float, default=None,
                   help="Gaussian sigma in time [hours].")
    p.add_argument("--sigma-alt-km", type=float, default=None,
                   help="Gaussian sigma in altitude [km].")

    p.add_argument("--include-w", action="store_true",
                   help="Include vertical wind and its gradients.")
    p.add_argument("--propagate-uncertainty", action="store_true",
                   help="Propagate uncertainties in mean, residuals, and gradients.")

    p.add_argument("--center-time", type=str, default=None, #"2024-11-10 21:00:00", #"2023-03-23 21:00:00",#
                   help='Center time for plotting in UTC, e.g. "2023-03-23 12:00:00".')
    p.add_argument("--time-range-hours", type=float, default=None,
                   help="Displayed time span around center time [hours].")

    p.add_argument("--cmap", type=str, default="seismic",
                   help="Matplotlib colormap name.")
    p.add_argument("--figsize-x", type=float, default=8,
                   help="Figure width in inches.")
    p.add_argument("--figsize-y", type=float, default=8,
                   help="Figure height in inches.")
    p.add_argument("--dpi", type=int, default=100,
                   help="Figure DPI for screen/export.")
    p.add_argument("--output", type=str, default="composite.pdf",
                   help="Output figure filename, e.g. diagnostics.png")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dpath = os.path.split(args.filename)[0]
    
    window = WindowConfig(
        time_hours=args.time_window_hours,
        alt_km=args.alt_window_km,
        kind=args.window_kind,
        sigma_time_hours=args.sigma_time_hours,
        sigma_alt_km=args.sigma_alt_km,
    )

    plot_cfg = PlotConfig(
        cmap=args.cmap,
        figsize=(args.figsize_x, args.figsize_y),
        dpi=args.dpi,
        output=args.output,
        center_time=parse_datetime_utc(args.center_time),
        time_range_hours=args.time_range_hours,
    )

    result = analyze_local_column(
        ncfile=args.filename,
        target_lat=args.lat,
        target_lon=args.lon,
        window=window,
        include_w=args.include_w,
        propagate_uncertainty=args.propagate_uncertainty,
    )

    fig, _ = plot_composite(result, plot_cfg)

    if plot_cfg.output:
        output_filename = os.path.join(dpath, plot_cfg.output)
        fig.savefig(output_filename, bbox_inches="tight")
        print(f"Saved figure to: {plot_cfg.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()