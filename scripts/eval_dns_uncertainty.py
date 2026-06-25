#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = PROJECT_ROOT / ".mplcache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperMLT.evaluation.io import load_field_data, load_manifest_refs, validate_aligned_grids
from hyperMLT.evaluation.masking import build_central_radius_mask
from hyperMLT.evaluation.plots import plot_uncertainty_snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DNS uncertainty across multiple realizations.")
    parser.add_argument("--inference-file", action="append", required=True, help="Repeat for each realization.")
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radius-fraction", type=float, default=0.7)
    return parser


def _write_csv(path: Path, summary: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "central_mean_std", "central_median_std", "central_max_std"])
        for component in ("u", "v", "w"):
            writer.writerow(
                [
                    component,
                    summary[f"{component}_central_mean_std"],
                    summary[f"{component}_central_median_std"],
                    summary[f"{component}_central_max_std"],
                ]
            )


def main() -> None:
    args = _parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [load_field_data(path) for path in args.inference_file]
    reference = files[0]
    for other in files[1:]:
        validate_aligned_grids(reference, other)

    refs = load_manifest_refs(args.artifact_path)
    horizontal_mask, mask_meta = build_central_radius_mask(
        reference.lon,
        reference.lat,
        lat_ref=refs["lat_ref"],
        lon_ref=refs["lon_ref"],
        alt_ref_km=refs["alt_ref_km"],
        radius_fraction=float(args.radius_fraction),
    )
    mask4 = horizontal_mask[None, :, :, None]

    std_fields: dict[str, np.ndarray] = {}
    summary: dict[str, float] = {
        "n_realizations": float(len(files)),
        "radius_fraction": float(args.radius_fraction),
    }

    for component in ("u", "v", "w"):
        stack = np.stack([getattr(field, component) for field in files], axis=0)
        std_field = np.nanstd(stack, axis=0)
        std_fields[component] = std_field
        central_values = std_field[mask4]
        summary[f"{component}_central_mean_std"] = float(np.nanmean(central_values))
        summary[f"{component}_central_median_std"] = float(np.nanmedian(central_values))
        summary[f"{component}_central_max_std"] = float(np.nanmax(central_values))

    with h5py.File(output_dir / "dns_uncertainty_std_fields.h5", "w") as fp:
        fp["t"] = reference.t
        fp["lon"] = reference.lon
        fp["lat"] = reference.lat
        fp["alt_km"] = reference.alt_km
        fp["u_std"] = std_fields["u"]
        fp["v_std"] = std_fields["v"]
        fp["w_std"] = std_fields["w"]
        fp["central_mask_lon_lat"] = horizontal_mask.astype(np.uint8)

    payload = {
        "artifact_dir": str(refs["artifact_dir"]),
        "inference_files": [str(field.path) for field in files],
        "mask": mask_meta,
        "summary": summary,
    }
    (output_dir / "dns_uncertainty_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output_dir / "dns_uncertainty_summary.csv", summary)
    plot_uncertainty_snapshot(
        reference.lon,
        reference.lat,
        reference.alt_km,
        reference.t,
        std_fields,
        horizontal_mask,
        output_dir / "dns_uncertainty_snapshot.png",
    )

    print("DNS uncertainty evaluation completed")
    print(f"Realizations: {len(files)}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
