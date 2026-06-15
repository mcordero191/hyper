#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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

from hyperMLT.artifacts.io import create_run_directory, write_json
from hyperMLT.config import load_train_config
from hyperMLT.models import describe_model_family
from hyperMLT.physics.formulations import describe_formulation, normalize_formulation_name
from hyperMLT.utils.console import format_summary_table
from hyperMLT.utils.paths import resolve_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="hyperMLT multi-day DeepGreen training entrypoint")
    parser.add_argument("--config-file", required=True, help="Path to the hyperMLT training config")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override")
    parser.add_argument("--print-json", action="store_true", help="Print resolved config as JSON")
    return parser


def _discover_files(dataset_config: dict, *, config_path: str) -> list[Path]:

    dataset_path = resolve_from_config(config_path, dataset_config["path"])
    pattern = str(dataset_config.get("pattern", "*"))
    candidates = sorted(dataset_path.glob(f"{pattern}.h5"))

    if not candidates:
        raise RuntimeError(f"No SMR files matching '{pattern}.h5' in '{dataset_path}'.")

    return candidates


def _day_label(file_path: Path) -> str:

    stem = file_path.stem

    for token in stem.split("_"):
        if len(token) == 8 and token.isdigit():
            return token

    return stem


def _shared_trunk_signature(config) -> str:

    network = config.to_dict()["model"]["network"]
    payload = json.dumps(network, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    kernel = network.get("KernelFunction", {})
    sources = network.get("SourceGenerator", {})
    encoder = network.get("Encoder", {})
    decoder = network.get("Decoder", {})

    return (
        f"ld{int(kernel.get('latent_dim', encoder.get('width', 64)))}"
        f"_ng{int(kernel.get('n_green_functions', 16))}"
        f"_ns{int(sources.get('n_sources', 32))}"
        f"_sw{int(sources.get('width', 16))}"
        f"_ew{int(encoder.get('width', 64))}d{int(encoder.get('depth', 4))}"
        f"_dw{int(decoder.get('width', encoder.get('width', 64)))}d{int(decoder.get('depth', encoder.get('depth', 4)))}"
        f"_{digest}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_train_config(args.config_file)

    if args.print_json:
        print(json.dumps(config.to_dict(), indent=2))
        return

    if str(config.model.architecture).lower() != "deepgreen":
        raise ValueError("multiday_train.py currently supports only DeepGreen.")

    if not bool(config.multiday.enabled):
        raise ValueError("multiday.enabled must be true for multiday_train.py.")

    model_family = describe_model_family(config.model.architecture)
    formulation_name = normalize_formulation_name(config.physics.formulation)
    formulation_label = describe_formulation(formulation_name)

    config.datasets.primary["path"] = str(resolve_from_config(config._config_path, config.datasets.primary["path"]))
    output_root = resolve_from_config(config._config_path, config.output.root_dir)
    dataset_config = dict(config.to_dict()["datasets"]["primary"])
    files = _discover_files(dataset_config, config_path=config._config_path)

    if str(config.multiday.file_order).lower() != "chronological":
        raise ValueError("Only multiday.file_order = 'chronological' is currently supported.")

    day_runs_root_value = config.multiday.day_runs.get("root_dir", config.output.root_dir)
    day_runs_root = resolve_from_config(config._config_path, day_runs_root_value)
    shared_trunk_dir_value = config.multiday.shared_trunk.get("checkpoint_dir", "../../runs/hyperMLT/shared_trunks")
    shared_trunk_dir = resolve_from_config(config._config_path, shared_trunk_dir_value)
    trunk_anchor_weight = float(config.multiday.shared_trunk.get("anchor_weight", 0.0))
    freeze_trunk_until_epoch = int(config.multiday.shared_trunk.get("freeze_until_epoch", 0))
    trunk_signature = _shared_trunk_signature(config)
    latest_trunk_checkpoint = shared_trunk_dir / f"latest_trunk_{trunk_signature}.npz"

    print(
        format_summary_table(
            "Multiday Training Plan",
            [
                ("Experiment", config.experiment.name),
                ("Architecture", model_family),
                ("PDE formulation", f"{formulation_label} ({formulation_name})"),
                ("Days", len(files)),
                ("Shared trunk dir", shared_trunk_dir),
                ("Trunk signature", trunk_signature),
                ("Day runs dir", day_runs_root),
                ("Trunk anchor", f"{trunk_anchor_weight:.1e}"),
                ("Freeze trunk until", freeze_trunk_until_epoch),
                ("Resume trunk", latest_trunk_checkpoint if latest_trunk_checkpoint.exists() else "fresh"),
                ("Execution model", "subprocess per day"),
            ],
        )
    )

    completed_days: list[dict[str, object]] = []
    trunk_checkpoint: Path | None = latest_trunk_checkpoint if latest_trunk_checkpoint.exists() else None
    worker_script = PROJECT_ROOT / "scripts" / "multiday_train_day.py"

    for day_index, file_path in enumerate(files):
        day_label = _day_label(file_path)
        command = [
            sys.executable,
            str(worker_script),
            "--config-file",
            str(Path(args.config_file).expanduser().resolve()),
            "--file-path",
            str(file_path),
            "--day-index",
            str(day_index),
            "--day-label",
            day_label,
            "--day-runs-root",
            str(day_runs_root),
            "--shared-trunk-dir",
            str(shared_trunk_dir),
            "--trunk-signature",
            trunk_signature,
        ]

        if trunk_checkpoint is not None:
            command.extend(["--trunk-checkpoint", str(trunk_checkpoint)])
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])

        print("")
        print(format_summary_table("Launching Day Process", [("Day index", day_index), ("Day label", day_label), ("File", file_path.name)]))
        subprocess.run(command, check=True)

        day_run_dir = create_run_directory(day_runs_root, config.experiment.name, day_label)
        summary_path = day_run_dir / "multiday_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trunk_checkpoint = Path(summary["shared_trunk_checkpoint"]).expanduser().resolve()
        completed_days.append(summary)

    summary_dir = create_run_directory(output_root, config.experiment.name, "multiday_summary")
    write_json(
        summary_dir,
        "completed_days.json",
        {
            "experiment": config.experiment.name,
            "architecture": model_family,
            "formulation": formulation_name,
            "shared_trunk_dir": str(shared_trunk_dir),
            "completed_days": completed_days,
        },
    )

    print("")
    print(
        format_summary_table(
            "Multiday Training Completed",
            [
                ("Experiment", config.experiment.name),
                ("Architecture", model_family),
                ("Days completed", len(completed_days)),
                ("Shared trunk dir", shared_trunk_dir),
                ("Summary", summary_dir / "completed_days.json"),
            ],
        )
    )


if __name__ == "__main__":
    main()
