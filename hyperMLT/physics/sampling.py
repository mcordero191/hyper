from __future__ import annotations

import numpy as np
import tensorflow as tf

from hyperMLT.datasets.stratification import build_strata_ids


def _domain_center_and_halfwidth(lower: tf.Tensor, upper: tf.Tensor) -> tuple[np.ndarray, np.ndarray]:

    lower_np = lower.numpy().astype(np.float32)
    upper_np = upper.numpy().astype(np.float32)

    center = (lower_np + upper_np) / 2.0
    halfwidth = (upper_np - lower_np) / 2.0

    return center, halfwidth


def _quantile_bounds(values: np.ndarray, coverage_fraction: float) -> tuple[float, float]:

    if values.size == 0:
        return 0.0, 1.0

    coverage_fraction = float(np.clip(coverage_fraction, 1.0e-3, 1.0))
    tail = 0.5 * (1.0 - coverage_fraction)

    if tail <= 0.0:
        return float(np.min(values)), float(np.max(values))

    lower = float(np.quantile(values, tail))
    upper = float(np.quantile(values, 1.0 - tail))

    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(np.min(values))
        upper = float(np.max(values))

    return lower, upper


def _fit_selected_support(
    training_df,
    *,
    xy_coverage_fraction: float,
    z_coverage_fraction: float,
    use_full_time_span: bool,
    fallback_center: np.ndarray,
    fallback_halfwidth: np.ndarray,
) -> dict[str, np.ndarray | float]:

    if training_df is None or len(training_df) == 0:
        transform = np.diag(fallback_halfwidth[2:4].astype(np.float64))
        return {
            "xy_center": fallback_center[2:4].astype(np.float64),
            "xy_transform": transform,
            "xy_radius_limit": 1.0,
            "z_bounds": np.asarray(
                [fallback_center[1] - fallback_halfwidth[1], fallback_center[1] + fallback_halfwidth[1]],
                dtype=np.float64,
            ),
            "t_bounds": np.asarray(
                [fallback_center[0] - fallback_halfwidth[0], fallback_center[0] + fallback_halfwidth[0]],
                dtype=np.float64,
            ),
        }

    xy_coverage_fraction = float(np.clip(xy_coverage_fraction, 1.0e-3, 1.0))
    x_values = training_df["x"].to_numpy(dtype=np.float64)
    y_values = training_df["y"].to_numpy(dtype=np.float64)
    x_bounds = np.asarray(_quantile_bounds(x_values, xy_coverage_fraction), dtype=np.float64)
    y_bounds = np.asarray(_quantile_bounds(y_values, xy_coverage_fraction), dtype=np.float64)

    xy_center = np.asarray(
        [
            0.5 * (x_bounds[0] + x_bounds[1]),
            0.5 * (y_bounds[0] + y_bounds[1]),
        ],
        dtype=np.float64,
    )
    xy_halfwidth = np.asarray(
        [
            max(0.5 * (x_bounds[1] - x_bounds[0]), 1.0),
            max(0.5 * (y_bounds[1] - y_bounds[0]), 1.0),
        ],
        dtype=np.float64,
    )
    transform = np.diag(xy_halfwidth)
    xy_radius_limit = 1.0

    z_values = training_df["z"].to_numpy(dtype=np.float64)
    z_bounds = np.asarray(_quantile_bounds(z_values, z_coverage_fraction), dtype=np.float64)

    t_values = training_df["times"].to_numpy(dtype=np.float64)
    if use_full_time_span:
        t_bounds = np.asarray([np.min(t_values), np.max(t_values)], dtype=np.float64)
    else:
        t_bounds = np.asarray(_quantile_bounds(t_values, xy_coverage_fraction), dtype=np.float64)

    return {
        "xy_center": xy_center.astype(np.float64),
        "xy_transform": transform.astype(np.float64),
        "xy_radius_limit": xy_radius_limit,
        "z_bounds": z_bounds,
        "t_bounds": t_bounds,
    }


def _sample_random(
    sample_count: int,
    support: dict[str, np.ndarray | float],
    *,
    time_base: float,
    rng: np.random.Generator,
) -> np.ndarray:

    theta = rng.uniform(0.0, 2.0 * np.pi, size=(sample_count, 1))
    radius = np.sqrt(float(support["xy_radius_limit"]) * rng.uniform(0.0, 1.0, size=(sample_count, 1)))
    unit_disk = np.concatenate([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

    xy_center = np.asarray(support["xy_center"], dtype=np.float64)
    xy_transform = np.asarray(support["xy_transform"], dtype=np.float64)
    xy = xy_center.reshape(1, 2) + unit_disk @ xy_transform.T

    z_bounds = np.asarray(support["z_bounds"], dtype=np.float64)
    t_bounds = np.asarray(support["t_bounds"], dtype=np.float64)
    z = rng.uniform(z_bounds[0], z_bounds[1], size=(sample_count, 1))
    t = rng.uniform(t_bounds[0], t_bounds[1], size=(sample_count, 1)) - float(time_base)

    return np.concatenate([t, z, xy[:, 0:1], xy[:, 1:2]], axis=1).astype(np.float32)


def _sample_inverse_density(
    sample_count: int,
    training_df,
    support: dict[str, np.ndarray | float],
    *,
    time_base: float,
    rng: np.random.Generator,
) -> np.ndarray:

    strata_ids, info = build_strata_ids(training_df, time_bins=8, altitude_bins=10, radius_bins=5)

    _, inverse, counts = np.unique(strata_ids, return_inverse=True, return_counts=True)

    row_weights = 1.0 / np.maximum(counts[inverse].astype(np.float64), 1.0)
    row_weights = row_weights / np.sum(row_weights)

    chosen_rows = rng.choice(np.arange(len(training_df)), size=sample_count, replace=True, p=row_weights)
    selected = training_df.iloc[chosen_rows]

    time_bins = int(info["time_bins"])
    altitude_bins = int(info["altitude_bins"])
    radius_bins = int(info["radius_bins"])

    t_bounds = np.asarray(support["t_bounds"], dtype=np.float64)
    z_bounds = np.asarray(support["z_bounds"], dtype=np.float64)
    time_edges = np.linspace(t_bounds[0], t_bounds[1], time_bins + 1)
    z_edges = np.linspace(z_bounds[0], z_bounds[1], altitude_bins + 1)

    xy_center = np.asarray(support["xy_center"], dtype=np.float64)
    xy_transform = np.asarray(support["xy_transform"], dtype=np.float64)
    xy_radius_limit = float(support["xy_radius_limit"])

    inv_transform = np.linalg.pinv(xy_transform)
    xy_values = training_df[["x", "y"]].to_numpy(dtype=np.float64) - xy_center.reshape(1, 2)
    uv_values = xy_values @ inv_transform.T
    radial_norm_values = np.sqrt(np.sum(uv_values**2, axis=1))
    radial_norm_values = np.clip(radial_norm_values, 0.0, np.sqrt(xy_radius_limit))
    radial_edges_sq = np.linspace(radial_norm_values.min() ** 2, radial_norm_values.max() ** 2, radius_bins + 1)
    radial_edges = np.sqrt(radial_edges_sq)

    selected_xy = selected[["x", "y"]].to_numpy(dtype=np.float64) - xy_center.reshape(1, 2)
    selected_uv = selected_xy @ inv_transform.T
    selected_radial_norm = np.sqrt(np.sum(selected_uv**2, axis=1))
    selected_radial_norm = np.clip(selected_radial_norm, 0.0, np.sqrt(xy_radius_limit))

    time_ids = np.clip(np.digitize(selected["times"].to_numpy(), time_edges[1:-1]), 0, time_bins - 1)
    z_ids = np.clip(np.digitize(selected["z"].to_numpy(), z_edges[1:-1]), 0, altitude_bins - 1)
    radius_ids = np.clip(np.digitize(selected_radial_norm, radial_edges[1:-1]), 0, radius_bins - 1)

    t = rng.uniform(time_edges[time_ids], time_edges[time_ids + 1]).reshape(-1, 1) - float(time_base)
    z = rng.uniform(z_edges[z_ids], z_edges[z_ids + 1]).reshape(-1, 1)

    rho0 = radial_edges[radius_ids]
    rho1 = radial_edges[radius_ids + 1]
    rho = np.sqrt(rng.uniform(rho0**2, rho1**2)).reshape(-1, 1)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(sample_count, 1))
    unit_disk = np.concatenate([rho * np.cos(theta), rho * np.sin(theta)], axis=1)
    xy = xy_center.reshape(1, 2) + unit_disk @ xy_transform.T

    return np.concatenate([t.astype(np.float32), z.astype(np.float32), xy[:, 0:1].astype(np.float32), xy[:, 1:2].astype(np.float32)], axis=1).astype(np.float32)


def sample_pde_points(
    training_df,
    lower: tf.Tensor,
    upper: tf.Tensor,
    *,
    sample_count: int,
    method: str,
    time_base: float,
    random_seed: int,
    xy_coverage_fraction: float = 0.97,
    z_coverage_fraction: float = 0.97,
    use_full_time_span: bool = True,
) -> np.ndarray:

    rng = np.random.default_rng(int(random_seed))

    center, halfwidth = _domain_center_and_halfwidth(lower, upper)
    support = _fit_selected_support(
        training_df,
        xy_coverage_fraction=xy_coverage_fraction,
        z_coverage_fraction=z_coverage_fraction,
        use_full_time_span=use_full_time_span,
        fallback_center=center,
        fallback_halfwidth=halfwidth,
    )

    normalized_method = str(method).lower()

    if normalized_method == "inverse_density":
        coords_raw = _sample_inverse_density(
            sample_count,
            training_df,
            support,
            time_base=time_base,
            rng=rng,
        )
    else:
        coords_raw = _sample_random(
            sample_count,
            support,
            time_base=time_base,
            rng=rng,
        )

    return coords_raw
