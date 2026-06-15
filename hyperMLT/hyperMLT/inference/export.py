from __future__ import annotations

from pathlib import Path

import h5py

from .runner import InferenceResult


def export_hdf5(result: InferenceResult, output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as fp:
        fp["t"] = result.t
        fp["lon"] = result.lon
        fp["lat"] = result.lat
        fp["alt_km"] = result.alt_km
        fp["u"] = result.u
        fp["v"] = result.v
        fp["w"] = result.w
    return output_path
