from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import chi2, norm
from sklearn.preprocessing import LabelEncoder

from hyperMLT.datasets.clustering import hierarchical_cluster
from hyperMLT.datasets.mean_winds import mean_wind_grad
from hyperMLT.datasets.selection import attach_selection_key
from hyperMLT.plotting.mean_winds import plot_mean_winds
from hyperMLT.plotting.parameters import plot_parameter_histograms, plot_spacetime_sampling


FILTER_KEY_ALIASES = {
    "duplicate_space_threshold_m": "duplicate_distance_m",
    "duplicate_time_threshold_s": "duplicate_time_s",
    "enable_doppler_clustering_filter": "enable_cluster_outlier_filter",
    "max_abs_doppler_velocity": "doppler_velocity_abs_max",
    "max_arrival_angle_deg": "angle_of_arrival_max_deg",
    "clustering_passes": "clustering_iterations",
    "clustering_radius": "clustering_dbscan_eps",
    "clustering_min_samples": "clustering_dbscan_min_samples",
    "enable_mean_wind_quality_filter": "enable_mean_wind_filter",
    "mean_wind_residual_sigma": "mean_wind_outlier_sigma",
    "require_smr_like": "use_smr_like_only",
    "horizontal_support_sigma": "xyz_horizontal_sigma",
    "vertical_support_sigma": "xyz_vertical_sigma",
    "support_core_fraction": "xyz_support_fraction",
}


def normalize_filter_config(filter_config: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(filter_config or {})

    for new_key, old_key in FILTER_KEY_ALIASES.items():
        if new_key in normalized and old_key not in normalized:
            normalized[old_key] = normalized.pop(new_key)

    return normalized


def get_xyz_bounds(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    z: pd.Series | np.ndarray,
    sigma: float = 1.5,
    sigma_z: float = 3.0,
    support_fraction: float = 0.75,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)

    r2 = x_arr**2 + y_arr**2
    support_fraction = float(np.clip(support_fraction, 1.0e-3, 1.0))
    support_threshold = np.quantile(r2, support_fraction)
    support_mask = r2 <= support_threshold

    x_support = x_arr[support_mask]
    y_support = y_arr[support_mask]

    def _robust_centered_scale(values: np.ndarray) -> float:
        if values.size == 0:
            return 1.0

        mad = np.median(np.abs(values))
        scale = 1.4826 * mad

        if not np.isfinite(scale) or scale <= 0.0:
            scale = np.sqrt(np.mean(values**2))

        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0

        return float(scale)

    scale_x = _robust_centered_scale(x_support)
    scale_y = _robust_centered_scale(y_support)

    d2 = (x_arr / scale_x) ** 2 + (y_arr / scale_y) ** 2
    threshold_xy = chi2.ppf(norm.cdf(sigma), df=2)
    mask_xy = d2 < threshold_xy

    z0 = np.median(z_arr)
    mad_z = np.median(np.abs(z_arr - z0))
    mask_z = np.abs(z_arr - z0) < sigma_z * mad_z

    return mask_xy & mask_z


def remove_duplicates(
    df: pd.DataFrame,
    spatial_threshold: float = 0.5e3,
    temporal_threshold: float = 1.0,
) -> pd.DataFrame:
    df_sorted = df.sort_values(by="times").reset_index(drop=True)
    coordinates = df_sorted[["x", "y", "z"]].to_numpy()
    times = df_sorted["times"].to_numpy()
    links = df_sorted["link"].to_numpy()

    spatial_tree = KDTree(coordinates)
    keep_indices: list[int] = []
    discard = np.zeros(len(df_sorted), dtype=bool)

    for i in range(len(df_sorted)):
        if discard[i]:
            continue

        spatial_neighbors = spatial_tree.query_ball_point(coordinates[i], spatial_threshold)
        for neighbor in spatial_neighbors:
            if neighbor == i:
                continue
            if links[neighbor] != links[i]:
                continue
            if abs(times[i] - times[neighbor]) < temporal_threshold:
                discard[neighbor] = True
                break

        if not discard[i]:
            keep_indices.append(i)

    return df_sorted.iloc[keep_indices]


def filter_by_angle(df: pd.DataFrame, angle_limit: float = 60.0) -> pd.DataFrame:
    dxy = np.sqrt(df["dcosx"] ** 2 + df["dcosy"] ** 2)
    angle = np.degrees(np.arcsin(dxy))
    return df[angle < angle_limit]


def filter_data(
    df: pd.DataFrame,
    tini: float = 0.0,
    dt: float = 24.0,
    overlapping_time: float = 2 * 60 * 60,
    central_date: Any = None,
    ena_clustering: int = 1,
    use_smr_like_only: bool = True,
    doppler_velocity_abs_max: float = 70.0,
    angle_of_arrival_max_deg: float = 60.0,
    clustering_iterations: int = 2,
    clustering_dbscan_eps: float = 0.1,
    clustering_dbscan_min_samples: int = 100,
    xyz_horizontal_sigma: float = 1.5,
    xyz_vertical_sigma: float = 3.0,
    xyz_support_fraction: float = 0.75,
    verbose: bool = False,
) -> pd.DataFrame:
    if use_smr_like_only and "SMR_like" in df.keys():
        df = df[df["SMR_like"] == 1]

    if central_date is not None:
        tbase = pd.to_datetime(central_date)
    elif tini >= 0:
        tbase = pd.to_datetime(df["t"].min())
    else:
        tbase = pd.to_datetime(df["t"].max())

    t = tbase + pd.to_timedelta(tini, unit="h")
    tmin = t - pd.to_timedelta(overlapping_time, unit="s")
    tmax = t + pd.to_timedelta(dt * 60 * 60 + overlapping_time, unit="s")

    df = df[(df["t"] >= tmin) & (df["t"] <= tmax)]
    if df.empty:
        return df

    df = df[np.abs(df["dops"]) < float(doppler_velocity_abs_max)]
    if df.empty:
        return df

    df = filter_by_angle(df, angle_limit=float(angle_of_arrival_max_deg))
    if df.empty:
        return df

    if ena_clustering:
        le = LabelEncoder()
        le.fit(df["link"].values)
        nlinks = int(clustering_iterations)

        for _ in range(nlinks):
            dxy = np.sqrt(df["dcosx"].values ** 2 + df["dcosy"].values ** 2)
            zenith = np.arcsin(dxy) * 180 / np.pi
            if np.max(zenith) <= 0:
                break

            X = np.stack(
                [df["dops"].values, zenith, le.transform(df["link"].values)],
                axis=1,
            )
            valid = hierarchical_cluster(
                X,
                verbose=verbose,
                eps=float(clustering_dbscan_eps),
                min_samples=int(clustering_dbscan_min_samples),
            )
            if np.count_nonzero(~valid) == 0:
                break
            df = df[valid]

        if df.empty:
            return df

    valid = get_xyz_bounds(
        df["x"],
        df["y"],
        df["z"],
        sigma=float(xyz_horizontal_sigma),
        sigma_z=float(xyz_vertical_sigma),
        support_fraction=float(xyz_support_fraction),
    )
    return df[valid]


def save_mean_winds(df: dict[str, np.ndarray], filename: str) -> None:
    with h5py.File(filename, "w") as fp:
        for key, value in df.items():
            fp[key] = value


def _split_selected_and_discarded(raw_df: pd.DataFrame, selected_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    raw_keyed = attach_selection_key(raw_df)
    selected_keyed = attach_selection_key(selected_df)

    selected_pairs = set(
        selected_keyed[["_selection_key", "_selection_rank"]].itertuples(index=False, name=None)
    )

    keep_mask = np.array(
        [
            pair in selected_pairs
            for pair in raw_keyed[["_selection_key", "_selection_rank"]].itertuples(index=False, name=None)
        ],
        dtype=bool,
    )

    selected = raw_keyed.loc[keep_mask].copy()
    discarded = raw_keyed.loc[~keep_mask].copy()

    drop_cols = ["_selection_key", "_selection_rank"]
    selected.drop(columns=[column for column in drop_cols if column in selected.columns], inplace=True)
    discarded.drop(columns=[column for column in drop_cols if column in discarded.columns], inplace=True)

    return selected, discarded


@dataclass
class FilterResult:
    filtered_df: pd.DataFrame
    mean_winds: dict[str, np.ndarray] | None
    stats: dict[str, Any]


def apply_filter_pipeline(
    full_df: pd.DataFrame,
    *,
    central_date: Any = None,
    output_dir: str | None = None,
    enable_plots: bool = True,
    filter_config: dict[str, Any] | None = None,
) -> FilterResult:
    filter_config = normalize_filter_config(filter_config)

    def _decode_unique(values: Any) -> list[str]:
        decoded: list[str] = []
        for value in values:
            if isinstance(value, (bytes, bytearray)):
                decoded.append(value.decode("ascii", "replace"))
            else:
                decoded.append(str(value))
        return sorted(decoded)

    stats: dict[str, Any] = {
        "raw_samples": int(len(full_df)),
        "deduplicated_samples": 0,
        "used_samples": 0,
        "raw_links": _decode_unique(full_df["link"].unique()) if "link" in full_df else [],
        "used_links": [],
        "removed_links": [],
        "raw_tx_stations": _decode_unique(full_df["tx_station"].unique()) if "tx_station" in full_df else [],
        "raw_rx_stations": _decode_unique(full_df["rx_station"].unique()) if "rx_station" in full_df else [],
        "used_tx_stations": [],
        "used_rx_stations": [],
    }

    df = full_df.copy()
    raw_df = df.copy()
    duplicate_distance_m = float(filter_config.pop("duplicate_distance_m", 0.5e3))
    duplicate_time_s = float(filter_config.pop("duplicate_time_s", 1.0))
    mean_wind_outlier_sigma = float(filter_config.pop("mean_wind_outlier_sigma", 3.0))
    enable_mean_wind_filter = bool(filter_config.pop("enable_mean_wind_filter", True))

    df = remove_duplicates(
        df,
        spatial_threshold=duplicate_distance_m,
        temporal_threshold=duplicate_time_s,
    )
    stats["deduplicated_samples"] = int(len(df))
    if df.empty:
        return FilterResult(filtered_df=df, mean_winds=None, stats=stats)

    start_hour = float(filter_config.pop("start_hour", 0.0))
    duration_hours = float(filter_config.pop("duration_hours", 24.0))
    padding_fraction = float(filter_config.pop("padding_fraction", 0.0))
    overlapping_time = duration_hours * 60.0 * 60.0 * padding_fraction
    ena_clustering = 1 if bool(filter_config.pop("enable_cluster_outlier_filter", True)) else 0

    df = filter_data(
        df,
        tini=start_hour,
        dt=duration_hours,
        overlapping_time=overlapping_time,
        central_date=central_date,
        ena_clustering=ena_clustering,
        **filter_config,
    )
    if df.empty:
        return FilterResult(filtered_df=df, mean_winds=None, stats=stats)

    mean_winds = None
    if enable_mean_wind_filter:
        mean_winds, df = mean_wind_grad(df, outlier_sigma=mean_wind_outlier_sigma)
    stats["used_samples"] = int(len(df))
    stats["used_links"] = _decode_unique(df["link"].unique()) if "link" in df else []
    stats["removed_links"] = sorted(set(stats["raw_links"]) - set(stats["used_links"]))
    stats["used_tx_stations"] = _decode_unique(df["tx_station"].unique()) if "tx_station" in df else []
    stats["used_rx_stations"] = _decode_unique(df["rx_station"].unique()) if "rx_station" in df else []

    if output_dir is not None and enable_plots and not raw_df.empty and not df.empty:
        ini_date = datetime.utcfromtimestamp(float(df["times"].min()))
        selected_df, discarded_df = _split_selected_and_discarded(raw_df, df)
        parameter_plot = os.path.join(
            output_dir,
            f"sampling_parameters_selected_vs_discarded_{ini_date.strftime('%Y%m%d')}.png",
        )
        spacetime_plot = os.path.join(
            output_dir,
            f"sampling_spacetime_selected_vs_discarded_{ini_date.strftime('%Y%m%d')}.png",
        )
        plot_parameter_histograms(selected_df, discarded_df, parameter_plot)
        plot_spacetime_sampling(selected_df, discarded_df, spacetime_plot)

    if output_dir is not None and mean_winds is not None:
        ini_date = datetime.utcfromtimestamp(df["times"].min())
        if enable_plots:
            figfile = os.path.join(output_dir, f"mean_wind_{ini_date.strftime('%Y%m%d')}.png")
            plot_mean_winds(
                mean_winds["times"],
                mean_winds["alts"],
                mean_winds["u0"],
                mean_winds["v0"],
                mean_winds["w0"],
                vmins=[-100, -100, -10],
                vmaxs=[100, 100, 10],
                figfile=figfile,
                histogram=True,
            )
        filename = os.path.join(output_dir, f"mean_wind_{ini_date.strftime('%Y%m%d')}.hdf5")
        save_mean_winds(mean_winds, filename)

    return FilterResult(filtered_df=df, mean_winds=mean_winds, stats=stats)
