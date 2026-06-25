from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hyperMLT.artifacts.io import read_json, resolve_artifact_dir


@dataclass
class FieldData:
    path: Path
    t: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    alt_km: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray


def _read_dataset(fp: h5py.File, name: str) -> np.ndarray:
    if name not in fp:
        raise KeyError(f"Dataset '{name}' not found in file '{fp.filename}'.")
    return np.asarray(fp[name][...])


def load_field_data(
    path: str | Path,
    *,
    t_name: str = "t",
    lon_name: str = "lon",
    lat_name: str = "lat",
    alt_name: str = "alt_km",
    u_name: str = "u",
    v_name: str = "v",
    w_name: str = "w",
) -> FieldData:
    field_path = Path(path).expanduser().resolve()
    with h5py.File(field_path, "r") as fp:
        return FieldData(
            path=field_path,
            t=_read_dataset(fp, t_name).astype(np.float64),
            lon=_read_dataset(fp, lon_name).astype(np.float64),
            lat=_read_dataset(fp, lat_name).astype(np.float64),
            alt_km=_read_dataset(fp, alt_name).astype(np.float64),
            u=_read_dataset(fp, u_name).astype(np.float64),
            v=_read_dataset(fp, v_name).astype(np.float64),
            w=_read_dataset(fp, w_name).astype(np.float64),
        )


def validate_aligned_grids(reference: FieldData, other: FieldData, *, atol: float = 1.0e-6) -> None:
    for name in ("t", "lon", "lat", "alt_km"):
        a = np.asarray(getattr(reference, name), dtype=np.float64)
        b = np.asarray(getattr(other, name), dtype=np.float64)
        if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=0.0, equal_nan=True):
            raise ValueError(
                f"Grid mismatch for '{name}' between '{reference.path.name}' and '{other.path.name}'."
            )

    if reference.u.shape != other.u.shape:
        raise ValueError(
            f"Field shape mismatch between '{reference.path.name}' and '{other.path.name}': "
            f"{reference.u.shape} != {other.u.shape}."
        )


def load_manifest_refs(artifact_path: str | Path) -> dict[str, Any]:
    artifact_dir = resolve_artifact_dir(artifact_path)
    manifest = read_json(artifact_dir / "manifest.json")
    return {
        "artifact_dir": artifact_dir,
        "lat_ref": float(manifest["lat_ref"]),
        "lon_ref": float(manifest["lon_ref"]),
        "alt_ref_km": float(manifest["alt_ref_km"]),
    }
