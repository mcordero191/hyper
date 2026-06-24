#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = PROJECT_ROOT / ".mplcache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LEGACY_SRC_ROOT = PROJECT_ROOT.parent / "src"
if str(LEGACY_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC_ROOT))

from hyperMLT.artifacts.io import create_run_directory, write_json, write_resolved_config
from hyperMLT.config import load_train_config
from hyperMLT.datasets import load_window_for_file
from hyperMLT.models import build_model, describe_model_family
from hyperMLT.physics.formulations import describe_formulation, normalize_formulation_name
from hyperMLT.training.trainer import train_model
from hyperMLT.utils.console import format_dataset_summary, format_summary_table
from hyperMLT.utils.paths import resolve_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one day for hyperMLT multiday workflow")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--file-path", required=True)
    parser.add_argument("--day-index", type=int, required=True)
    parser.add_argument("--day-label", required=True)
    parser.add_argument("--day-runs-root", required=True)
    parser.add_argument("--shared-trunk-dir", required=True)
    parser.add_argument("--trunk-signature", required=True)
    parser.add_argument("--trunk-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser


def _load_trunk_checkpoint(path: str | None) -> list[np.ndarray] | None:

    if not path:
        return None

    checkpoint_path = Path(path).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Shared trunk checkpoint '{checkpoint_path}' does not exist.")

    with np.load(checkpoint_path) as payload:
        arrays = [np.array(payload[key], copy=True) for key in sorted(payload.files, key=lambda item: int(item.split('_')[-1]))]

    return arrays


def _initialize_model_for_day(config, trunk_weights):

    model = build_model(config.model, shape_out=3)
    dummy = np.zeros((1, 4), dtype=np.float32)
    _ = model(dummy, training=False)

    if trunk_weights is not None:
        model.backbone.trunk.set_weights(trunk_weights)

    return model


def _write_trunk_checkpoint(shared_trunk_dir: Path, *, day_label: str, trunk_signature: str, trunk_weights) -> Path:

    shared_trunk_dir.mkdir(parents=True, exist_ok=True)
    target = shared_trunk_dir / f"trunk_{trunk_signature}_{day_label}.npz"
    latest = shared_trunk_dir / f"latest_trunk_{trunk_signature}.npz"
    payload = {f"arr_{index}": weight for index, weight in enumerate(trunk_weights)}
    np.savez_compressed(target, **payload)
    np.savez_compressed(latest, **payload)

    return target


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_train_config(args.config_file)

    if str(config.model.architecture).lower() != "deepgreen":
        raise ValueError("multiday_train_day.py currently supports only DeepGreen.")

    model_family = describe_model_family(config.model.architecture)
    formulation_name = normalize_formulation_name(config.physics.formulation)
    formulation_label = describe_formulation(formulation_name)

    config.datasets.primary["path"] = str(resolve_from_config(config._config_path, config.datasets.primary["path"]))
    config_dict = config.to_dict()
    dataset_config = dict(config_dict["datasets"]["primary"])
    domain_config = dict(config_dict["domain"])

    day_runs_root = Path(args.day_runs_root).expanduser().resolve()
    day_run_dir = create_run_directory(day_runs_root, config.experiment.name, args.day_label)
    write_resolved_config(day_run_dir, config)

    file_path = Path(args.file_path).expanduser().resolve()
    window = load_window_for_file(
        file_path,
        dataset_config,
        domain_config,
        output_dir=day_run_dir,
        enable_plots=bool(config.output.write_plots),
    )
    config.domain.region["lon_center"] = float(window.metadata["active_center"][0])
    config.domain.region["lat_center"] = float(window.metadata["active_center"][1])
    config.domain.region["alt_center_km"] = float(window.metadata["active_center"][2])
    write_resolved_config(day_run_dir, config)

    print("")
    print(format_summary_table("Multiday Step", [("Day index", args.day_index), ("Day label", args.day_label), ("File", file_path.name)]))
    print("")
    print(
        format_dataset_summary(
            window=window,
            dataset_source=str(config.datasets.primary.get("source", "unknown")),
            validation_config=dict(config.datasets.primary.get("validation", {})),
            noise_config=dict(config.datasets.primary.get("noise", {})),
        )
    )

    carried_trunk_weights = _load_trunk_checkpoint(args.trunk_checkpoint)
    model = _initialize_model_for_day(config, carried_trunk_weights)
    trunk_reference = None if carried_trunk_weights is None else [np.array(weight, copy=True) for weight in carried_trunk_weights]
    trunk_anchor_weight = float(config.multiday.shared_trunk.get("anchor_weight", 0.0)) if trunk_reference is not None else 0.0
    freeze_trunk_until_epoch = int(config.multiday.shared_trunk.get("freeze_until_epoch", 0)) if trunk_reference is not None else 0

    artifacts = train_model(
        config,
        window,
        day_run_dir,
        model_family=model_family,
        formulation_label=formulation_label,
        epochs_override=args.epochs,
        model=model,
        trunk_anchor_reference_weights=trunk_reference,
        trunk_anchor_weight=trunk_anchor_weight,
        freeze_trunk_until_epoch=freeze_trunk_until_epoch,
    )

    current_trunk_weights = [np.array(weight, copy=True) for weight in model.backbone.trunk.get_weights()]
    shared_trunk_dir = Path(args.shared_trunk_dir).expanduser().resolve()
    trunk_checkpoint = _write_trunk_checkpoint(
        shared_trunk_dir,
        day_label=args.day_label,
        trunk_signature=args.trunk_signature,
        trunk_weights=current_trunk_weights,
    )

    write_json(
        day_run_dir,
        "multiday_summary.json",
        {
            "day_index": int(args.day_index),
            "day_label": args.day_label,
            "file_name": file_path.name,
            "trunk_signature": args.trunk_signature,
            "shared_trunk_checkpoint": str(trunk_checkpoint),
            "carried_trunk_from_previous_day": carried_trunk_weights is not None,
            "trunk_anchor_weight": trunk_anchor_weight,
            "freeze_trunk_until_epoch": freeze_trunk_until_epoch,
            "history_file": artifacts.history_path.name,
            "weights_file": artifacts.weights_path.name,
            "run_dir": str(day_run_dir),
        },
    )

    print("")
    print(
        format_summary_table(
            "Multiday Day Completed",
            [
                ("Day index", int(args.day_index)),
                ("Day label", args.day_label),
                ("Trunk signature", args.trunk_signature),
                ("Shared trunk checkpoint", trunk_checkpoint),
                ("Run dir", day_run_dir),
            ],
        )
    )


if __name__ == "__main__":
    main()
