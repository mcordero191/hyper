#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = PROJECT_ROOT / ".mplcache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LEGACY_SRC_ROOT = PROJECT_ROOT.parent / "src"
if str(LEGACY_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC_ROOT))

from hyperMLT.artifacts.io import create_run_directory, write_json, write_resolved_config
from hyperMLT.config import load_train_config
from hyperMLT.datasets import build_window_summary, load_first_window
from hyperMLT.models import describe_model_family
from hyperMLT.physics.formulations import describe_formulation, normalize_formulation_name
from hyperMLT.training.trainer import train_model
from hyperMLT.utils.paths import resolve_from_config
from hyperMLT.utils.console import format_summary_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="hyperMLT training entrypoint")
    parser.add_argument("--config-file", required=True, help="Path to the hyperMLT training config")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for smoke tests")
    parser.add_argument("--print-json", action="store_true", help="Print resolved config as JSON")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_train_config(args.config_file)
    if args.print_json:
        print(json.dumps(config.to_dict(), indent=2))
        return

    model_family = describe_model_family(config.model.architecture)
    formulation_name = normalize_formulation_name(config.physics.formulation)
    formulation_label = describe_formulation(formulation_name)
    output_root = resolve_from_config(config._config_path, config.output.root_dir)
    config.datasets.primary["path"] = str(resolve_from_config(config._config_path, config.datasets.primary["path"]))
    run_dir = create_run_directory(output_root, config.experiment.name, "latest")
    write_resolved_config(run_dir, config)

    window = load_first_window(
        config.datasets.primary,
        config.to_dict()["domain"],
        output_dir=run_dir,
        enable_plots=bool(config.output.write_plots),
    )

    print("\n" + build_window_summary(window))
    print(f"Training meteors after selection: {len(window.training_df)}")
    print(f"Validation meteors: {0 if window.validation_df is None else len(window.validation_df)}")
    if getattr(window, "validation_inner_df", None) is not None:
        print(f"Validation inner meteors: {len(window.validation_inner_df)}")
    if getattr(window, "validation_outer_df", None) is not None:
        print(f"Validation outer meteors: {len(window.validation_outer_df)}")
    dataset_timing = dict(window.metadata.get("timings", {}))
    cache_info = dict(window.metadata.get("cache", {}))

    if dataset_timing:
        print("")
        print(
            format_summary_table(
                "Dataset Preparation Timing",
                [
                    ("Cache used", bool(cache_info.get("used", False))),
                    *[
                        (label.replace("_s", "").replace("_", " "), f"{float(value):.2f}s")
                        for label, value in dataset_timing.items()
                    ],
                ],
            )
        )
    artifacts = train_model(
        config,
        window,
        run_dir,
        model_family=model_family,
        formulation_label=formulation_label,
        epochs_override=args.epochs,
    )

    print("")
    print(
        format_summary_table(
            "Training Run Completed",
            [
                ("Experiment", config.experiment.name),
                ("Architecture", model_family),
                ("Dataset source", config.datasets.primary.get("source", "unknown")),
                ("PDE formulation", f"{formulation_label} ({formulation_name})"),
                ("Artifacts", run_dir),
            ],
        )
    )

    write_json(
        run_dir,
        "dataset_summary.json",
        {
            "experiment": config.experiment.name,
            "architecture": model_family,
            "formulation": formulation_name,
            "training_samples": int(len(window.training_df)),
            "validation_samples": int(0 if window.validation_df is None else len(window.validation_df)),
            "validation_inner_samples": int(0 if getattr(window, "validation_inner_df", None) is None else len(window.validation_inner_df)),
            "validation_outer_samples": int(0 if getattr(window, "validation_outer_df", None) is None else len(window.validation_outer_df)),
            "metadata": window.metadata,
            "history_file": artifacts.history_path.name,
            "weights_file": artifacts.weights_path.name,
        },
    )


if __name__ == "__main__":
    main()
