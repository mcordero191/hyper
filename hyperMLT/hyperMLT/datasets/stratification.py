from __future__ import annotations

import numpy as np
import pandas as pd


def _resolve_altitude_values(df: pd.DataFrame) -> np.ndarray:

    if "heights" in df.columns:
        return df["heights"].to_numpy(dtype=np.float64)

    if "z" in df.columns:
        values = df["z"].to_numpy(dtype=np.float64)

        if np.nanmax(np.abs(values)) < 1e3:
            values = values * 1e3

        return values

    raise ValueError("Unable to resolve altitude values from dataframe.")


def _resolve_radius_values(df: pd.DataFrame) -> np.ndarray:

    if "x" not in df.columns or "y" not in df.columns:
        return np.zeros(len(df), dtype=np.float64)

    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)

    return np.sqrt(x ** 2 + y ** 2)


def _digitize_equal_area_radius(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, int]:

    values = np.asarray(values, dtype=np.float64)

    if n_bins <= 1 or values.size == 0:
        return np.zeros(values.shape[0], dtype=np.int32), 1

    finite_mask = np.isfinite(values)

    if not np.any(finite_mask):
        return np.zeros(values.shape[0], dtype=np.int32), 1

    finite_values = values[finite_mask]
    radius_sq = finite_values**2
    value_min = float(np.nanmin(radius_sq))
    value_max = float(np.nanmax(radius_sq))

    if not np.isfinite(value_min) or not np.isfinite(value_max) or np.isclose(value_min, value_max):
        return np.zeros(values.shape[0], dtype=np.int32), 1

    edges_sq = np.linspace(value_min, value_max, int(n_bins) + 1, dtype=np.float64)
    ids = np.digitize(values**2, edges_sq[1:-1], right=False).astype(np.int32)

    ids[~finite_mask] = 0
    ids = np.clip(ids, 0, int(n_bins) - 1)

    return ids, int(n_bins)


def _digitize(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, int]:

    values = np.asarray(values, dtype=np.float64)

    if n_bins <= 1 or values.size == 0:
        return np.zeros(values.shape[0], dtype=np.int32), 1

    finite_mask = np.isfinite(values)

    if not np.any(finite_mask):
        return np.zeros(values.shape[0], dtype=np.int32), 1

    finite_values = values[finite_mask]
    value_min = float(np.nanmin(finite_values))
    value_max = float(np.nanmax(finite_values))

    if not np.isfinite(value_min) or not np.isfinite(value_max) or np.isclose(value_min, value_max):
        return np.zeros(values.shape[0], dtype=np.int32), 1

    edges = np.linspace(value_min, value_max, int(n_bins) + 1, dtype=np.float64)
    ids = np.digitize(values, edges[1:-1], right=False).astype(np.int32)

    ids[~finite_mask] = 0
    ids = np.clip(ids, 0, int(n_bins) - 1)

    return ids, int(n_bins)


def build_strata_ids(
    df: pd.DataFrame,
    time_bins: int = 8,
    altitude_bins: int = 10,
    radius_bins: int = 5,
) -> tuple[np.ndarray, dict[str, int]]:

    if len(df) == 0:
        return np.zeros(0, dtype=np.int32), {
            "time_bins": 1,
            "altitude_bins": 1,
            "radius_bins": 1,
        }

    time_ids, time_count = _digitize(df["times"].to_numpy(dtype=np.float64), int(time_bins))
    altitude_ids, altitude_count = _digitize(_resolve_altitude_values(df), int(altitude_bins))
    radius_ids, radius_count = _digitize_equal_area_radius(_resolve_radius_values(df), int(radius_bins))

    strata_ids = time_ids + time_count * (altitude_ids + altitude_count * radius_ids)

    return strata_ids.astype(np.int32), {
        "time_bins": time_count,
        "altitude_bins": altitude_count,
        "radius_bins": radius_count,
    }
