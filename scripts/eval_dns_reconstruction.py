#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = PROJECT_ROOT / ".mplcache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperMLT.evaluation.io import load_field_data, load_manifest_refs, validate_aligned_grids
from hyperMLT.evaluation.masking import build_central_radius_mask
from hyperMLT.evaluation.metrics import summarize_metrics
from hyperMLT.evaluation.plots import plot_metric_profiles, plot_xy_snapshot_comparison


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DNS reconstruction against truth.")
    parser.add_argument("--recon-file", required=True)
    parser.add_argument("--truth-file", required=True)
    parser.add_argument("--artifact-path", required=True, help="Training artifact path for lat/lon reference metadata.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radius-fraction", type=float, default=0.7)
    parser.add_argument("--truth-t-name", default="t")
    parser.add_argument("--truth-lon-name", default="lon")
    parser.add_argument("--truth-lat-name", default="lat")
    parser.add_argument("--truth-alt-name", default="alt_km")
    parser.add_argument("--truth-u-name", default="u")
    parser.add_argument("--truth-v-name", default="v")
    parser.add_argument("--truth-w-name", default="w")
    parser.add_argument("--recon-t-name", default="t")
    parser.add_argument("--recon-lon-name", default="lon")
    parser.add_argument("--recon-lat-name", default="lat")
    parser.add_argument("--recon-alt-name", default="alt_km")
    parser.add_argument("--recon-u-name", default="u")
    parser.add_argument("--recon-v-name", default="v")
    parser.add_argument("--recon-w-name", default="w")
    return parser


def _write_csv(path: Path, summary: dict[str, dict[str, dict[str, float]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["domain", "component", "count", "cc", "rae", "nmae", "rmse", "bias"])
        for domain_name, metrics_by_component in summary.items():
            for component, metrics in metrics_by_component.items():
                writer.writerow(
                    [
                        domain_name,
                        component,
                        metrics["count"],
                        metrics["cc"],
                        metrics["rae"],
                        metrics["nmae"],
                        metrics["rmse"],
                        metrics["bias"],
                    ]
                )


def main() -> None:
    args = _parser().parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    recon = load_field_data(
        args.recon_file,
        t_name=args.recon_t_name,
        lon_name=args.recon_lon_name,
        lat_name=args.recon_lat_name,
        alt_name=args.recon_alt_name,
        u_name=args.recon_u_name,
        v_name=args.recon_v_name,
        w_name=args.recon_w_name,
    )
    truth = load_field_data(
        args.truth_file,
        t_name=args.truth_t_name,
        lon_name=args.truth_lon_name,
        lat_name=args.truth_lat_name,
        alt_name=args.truth_alt_name,
        u_name=args.truth_u_name,
        v_name=args.truth_v_name,
        w_name=args.truth_w_name,
    )
    validate_aligned_grids(recon, truth)

    refs = load_manifest_refs(args.artifact_path)
    horizontal_mask, mask_meta = build_central_radius_mask(
        recon.lon,
        recon.lat,
        lat_ref=refs["lat_ref"],
        lon_ref=refs["lon_ref"],
        alt_ref_km=refs["alt_ref_km"],
        radius_fraction=float(args.radius_fraction),
    )

    truth_fields = {"u": truth.u, "v": truth.v, "w": truth.w}
    recon_fields = {"u": recon.u, "v": recon.v, "w": recon.w}
    full_mask = np.ones((len(recon.lon), len(recon.lat)), dtype=bool)[None, :, :, None]
    central_mask = horizontal_mask[None, :, :, None]

    summary = {
        "full_domain": summarize_metrics(truth_fields, recon_fields, mask=full_mask),
        "central_domain": summarize_metrics(truth_fields, recon_fields, mask=central_mask),
    }

    payload = {
        "recon_file": str(recon.path),
        "truth_file": str(truth.path),
        "artifact_dir": str(refs["artifact_dir"]),
        "radius_fraction": float(args.radius_fraction),
        "mask": mask_meta,
        "metrics": summary,
    }
    (output_dir / "dns_reconstruction_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output_dir / "dns_reconstruction_metrics.csv", summary)

    plot_metric_profiles(truth, recon, horizontal_mask, output_dir / "dns_central_mean_profiles.png")
    plot_xy_snapshot_comparison(truth, recon, horizontal_mask, output_dir / "dns_xy_snapshot_comparison.png")

    print("DNS reconstruction evaluation completed")
    print(f"Recon file: {recon.path}")
    print(f"Truth file: {truth.path}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
