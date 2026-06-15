#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = PROJECT_ROOT / ".mplcache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LEGACY_SRC_ROOT = PROJECT_ROOT.parent / "src"
if str(LEGACY_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC_ROOT))

from hyperMLT.config import load_infer_config
from hyperMLT.inference.export import export_hdf5
from hyperMLT.inference.plots import plot_inference_diagnostics
from hyperMLT.inference.runner import run_inference
from hyperMLT.utils.paths import resolve_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="hyperMLT inference entrypoint")
    parser.add_argument("--config-file", required=True, help="Path to the hyperMLT inference config")
    parser.add_argument("--print-json", action="store_true", help="Print resolved config as JSON")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_infer_config(args.config_file)
    if args.print_json:
        print(json.dumps(config.to_dict(), indent=2))
        return

    config.model.artifact_path = str(resolve_from_config(config._config_path, config.model.artifact_path))
    result = run_inference(config)
    output_root = resolve_from_config(config._config_path, config.output.root_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{config.experiment.name}.h5"
    export_hdf5(result, output_path)

    print("hyperMLT inference run completed")
    print(f"Experiment: {config.experiment.name}")
    print(f"Artifact path: {result.artifact_dir}")
    print(f"Output file: {output_path}")
    print(f"Grid shape: t={len(result.t)}, lon={len(result.lon)}, lat={len(result.lat)}, alt={len(result.alt_km)}")

    if bool(config.inference.plotting.get("enabled", True)):
        plot_paths = plot_inference_diagnostics(
            result,
            output_root,
            plotting_cfg=dict(config.inference.plotting or {}),
        )
        print(f"Plots: {', '.join(str(path) for path in plot_paths)}")


if __name__ == "__main__":
    main()
