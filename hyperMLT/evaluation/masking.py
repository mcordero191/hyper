from __future__ import annotations

import numpy as np

from hyperMLT.utils.coordinates import lla2xyh


def build_central_radius_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    lat_ref: float,
    lon_ref: float,
    alt_ref_km: float,
    radius_fraction: float = 0.7,
) -> tuple[np.ndarray, dict[str, float]]:
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon_mesh, lat_mesh = np.meshgrid(lon, lat, indexing="ij")
    alt_mesh = np.full_like(lon_mesh, float(alt_ref_km), dtype=np.float64)

    x, y, _ = lla2xyh(
        lat_mesh.reshape(-1),
        lon_mesh.reshape(-1),
        alt_mesh.reshape(-1),
        float(lat_ref),
        float(lon_ref),
        float(alt_ref_km),
    )
    x = x.reshape(lon_mesh.shape)
    y = y.reshape(lon_mesh.shape)

    x_center = 0.5 * (float(np.nanmin(x)) + float(np.nanmax(x)))
    y_center = 0.5 * (float(np.nanmin(y)) + float(np.nanmax(y)))
    x_halfwidth = 0.5 * (float(np.nanmax(x)) - float(np.nanmin(x)))
    y_halfwidth = 0.5 * (float(np.nanmax(y)) - float(np.nanmin(y)))
    max_radius = min(x_halfwidth, y_halfwidth)
    radius = float(radius_fraction) * max_radius

    r = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    mask = r <= radius

    meta = {
        "x_center_m": x_center,
        "y_center_m": y_center,
        "x_halfwidth_m": x_halfwidth,
        "y_halfwidth_m": y_halfwidth,
        "max_radius_m": max_radius,
        "radius_fraction": float(radius_fraction),
        "radius_m": radius,
    }
    return mask, meta
