from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from hyperMLT.artifacts.io import load_trained_model
from hyperMLT.utils.coordinates import lla2xyh, xyh2lla


@dataclass
class InferenceResult:
    artifact_dir: Path
    manifest: dict[str, Any]
    t: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    alt_km: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray


def _normalize(coords: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return 2.0 * (coords - lower) / (upper - lower) - 1.0


def _build_grid(manifest: dict[str, Any], infer_config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = infer_config.domain.grid
    lower = np.asarray(manifest["normalization"]["lower_bounds"], dtype=np.float64)
    upper = np.asarray(manifest["normalization"]["upper_bounds"], dtype=np.float64)

    time_step = float(grid.get("time_step", 600))
    alt_step_km = float(grid.get("alt_step", 1.0))

    t_range = grid.get("time_range") or [float(lower[0]), float(upper[0])]
    alt_range = grid.get("alt_range") or [float(lower[1] * 1e-3), float(upper[1] * 1e-3)]

    x_min, x_max = float(lower[2]), float(upper[2])
    y_min, y_max = float(lower[3]), float(upper[3])
    lat_ref = float(manifest["lat_ref"])
    lon_ref = float(manifest["lon_ref"])
    alt_ref = float(manifest["alt_ref_km"])

    lon_corners, lat_corners, _ = xyh2lla(
        np.asarray([x_min, x_max, x_min, x_max]),
        np.asarray([y_min, y_min, y_max, y_max]),
        np.asarray([alt_ref, alt_ref, alt_ref, alt_ref]),
        lat_ref,
        lon_ref,
        alt_ref,
    )
    lon_range = grid.get("lon_range") or [float(np.min(lon_corners)), float(np.max(lon_corners))]
    lat_range = grid.get("lat_range") or [float(np.min(lat_corners)), float(np.max(lat_corners))]

    lon_step = float(grid.get("lon_step", 0.25))
    lat_step = float(grid.get("lat_step", 0.25))

    t = np.arange(float(t_range[0]), float(t_range[1]) + 0.5 * time_step, time_step, dtype=np.float64)
    lon = np.arange(float(lon_range[0]), float(lon_range[1]) + 0.5 * lon_step, lon_step, dtype=np.float64)
    lat = np.arange(float(lat_range[0]), float(lat_range[1]) + 0.5 * lat_step, lat_step, dtype=np.float64)
    alt_km = np.arange(float(alt_range[0]), float(alt_range[1]) + 0.5 * alt_step_km, alt_step_km, dtype=np.float64)
    return t, lon, lat, alt_km


def run_inference(config) -> InferenceResult:
    artifact_dir, _train_config, manifest, model = load_trained_model(config.model.artifact_path)
    t, lon, lat, alt_km = _build_grid(manifest, config)

    t_mesh, lon_mesh, lat_mesh, alt_mesh = np.meshgrid(t, lon, lat, alt_km, indexing="ij")
    x, y, z = lla2xyh(
        lat_mesh.reshape(-1),
        lon_mesh.reshape(-1),
        alt_mesh.reshape(-1),
        manifest["lat_ref"],
        manifest["lon_ref"],
        manifest["alt_ref_km"],
    )
    coords = np.stack([t_mesh.reshape(-1), z.reshape(-1), x.reshape(-1), y.reshape(-1)], axis=1).astype(np.float32)
    lower = np.asarray(manifest["normalization"]["lower_bounds"], dtype=np.float32)
    upper = np.asarray(manifest["normalization"]["upper_bounds"], dtype=np.float32)
    coords_norm = _normalize(coords, lower, upper)
    outputs = model(tf.convert_to_tensor(coords_norm, dtype=tf.float32), training=False).numpy()

    shape = (len(t), len(lon), len(lat), len(alt_km))
    return InferenceResult(
        artifact_dir=artifact_dir,
        manifest=manifest,
        t=t,
        lon=lon,
        lat=lat,
        alt_km=alt_km,
        u=outputs[:, 0].reshape(shape),
        v=outputs[:, 1].reshape(shape),
        w=outputs[:, 2].reshape(shape),
    )
