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


def _sample_random(
    sample_count: int,
    center: np.ndarray,
    halfwidth: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:

    theta = rng.uniform(0.0, 2.0 * np.pi, size=(sample_count, 1))
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=(sample_count, 1)))

    x = center[2] + halfwidth[2] * radius * np.cos(theta)
    y = center[3] + halfwidth[3] * radius * np.sin(theta)

    z = center[1] + halfwidth[1] * rng.uniform(-1.0, 1.0, size=(sample_count, 1))
    t = center[0] + halfwidth[0] * rng.uniform(-1.0, 1.0, size=(sample_count, 1))

    return np.concatenate([t, z, x, y], axis=1).astype(np.float32)


def _sample_inverse_density(
    sample_count: int,
    training_df,
    halfwidth: np.ndarray,
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

    time_edges = np.linspace(training_df["times"].min(), training_df["times"].max(), time_bins + 1)
    z_edges = np.linspace(training_df["z"].min(), training_df["z"].max(), altitude_bins + 1)

    radial_norm_values = np.sqrt(
        (training_df["x"].to_numpy() / halfwidth[2]) ** 2
        + (training_df["y"].to_numpy() / halfwidth[3]) ** 2
    )
    radial_norm_values = np.clip(radial_norm_values, 0.0, 1.0)
    radial_edges_sq = np.linspace(radial_norm_values.min() ** 2, radial_norm_values.max() ** 2, radius_bins + 1)
    radial_edges = np.sqrt(radial_edges_sq)

    selected_radial_norm = np.sqrt(
        (selected["x"].to_numpy() / halfwidth[2]) ** 2
        + (selected["y"].to_numpy() / halfwidth[3]) ** 2
    )
    selected_radial_norm = np.clip(selected_radial_norm, 0.0, 1.0)

    time_ids = np.clip(np.digitize(selected["times"].to_numpy(), time_edges[1:-1]), 0, time_bins - 1)
    z_ids = np.clip(np.digitize(selected["z"].to_numpy(), z_edges[1:-1]), 0, altitude_bins - 1)
    radius_ids = np.clip(np.digitize(selected_radial_norm, radial_edges[1:-1]), 0, radius_bins - 1)

    t = rng.uniform(time_edges[time_ids], time_edges[time_ids + 1]).reshape(-1, 1)
    z = rng.uniform(z_edges[z_ids], z_edges[z_ids + 1]).reshape(-1, 1)

    rho0 = radial_edges[radius_ids]
    rho1 = radial_edges[radius_ids + 1]
    rho = np.sqrt(rng.uniform(rho0**2, rho1**2)).reshape(-1, 1)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(sample_count, 1))

    x = (halfwidth[2] * rho * np.cos(theta)).astype(np.float32)
    y = (halfwidth[3] * rho * np.sin(theta)).astype(np.float32)

    return np.concatenate([t.astype(np.float32), z.astype(np.float32), x, y], axis=1).astype(np.float32)


def sample_pde_points(
    training_df,
    lower: tf.Tensor,
    upper: tf.Tensor,
    *,
    sample_count: int,
    method: str,
    time_base: float,
    random_seed: int,
) -> np.ndarray:

    rng = np.random.default_rng(int(random_seed))

    center, halfwidth = _domain_center_and_halfwidth(lower, upper)

    normalized_method = str(method).lower()

    if normalized_method == "inverse_density":
        coords_raw = _sample_inverse_density(sample_count, training_df, halfwidth, rng)
        coords_raw[:, 0] = coords_raw[:, 0] - float(time_base)
    else:
        coords_raw = _sample_random(sample_count, center, halfwidth, rng)

    return coords_raw
