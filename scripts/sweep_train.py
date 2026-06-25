#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = PROJECT_ROOT / ".mplcache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperMLT.artifacts.io import read_json
from hyperMLT.config import load_train_config
from hyperMLT.utils.console import format_summary_table
from hyperMLT.utils.paths import resolve_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and compare multiple hyperMLT training configurations")
    parser.add_argument("--config-file", action="append", required=True, help="Training config to execute; repeat for multiple configs")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for every run")
    parser.add_argument(
        "--reuse-stamp",
        type=str,
        default=None,
        help="Reuse existing run directories matching <stamp>*<config-stem> and build only the sweep summary",
    )
    parser.add_argument(
        "--metric",
        default="best_val_doppler_rmse",
        choices=["best_val_doppler_rmse", "last_val_doppler_rmse", "last_val_inner_doppler_rmse", "last_val_outer_doppler_rmse"],
        help="Metric used to rank configurations",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(PROJECT_ROOT / "runs" / "hyperMLT" / "sweeps"),
        help="Directory where sweep summaries and comparison plots are written",
    )
    return parser


def _artifact_dir_for_run(config_path: Path, run_name: str) -> Path:
    config = load_train_config(config_path)
    output_root = resolve_from_config(config._config_path, config.output.root_dir)
    return output_root / config.experiment.name / run_name


def _artifact_root_for_config(config_path: Path) -> Path:
    config = load_train_config(config_path)
    output_root = resolve_from_config(config._config_path, config.output.root_dir)
    return output_root / config.experiment.name


def _resolve_reuse_run_name(config_path: Path, reuse_stamp: str) -> str:
    artifact_root = _artifact_root_for_config(config_path)
    stem = config_path.stem
    exact_name = f"{reuse_stamp}_{stem}"
    exact_dir = artifact_root / exact_name
    if exact_dir.is_dir():
        return exact_name

    matches = sorted(
        path.name
        for path in artifact_root.iterdir()
        if path.is_dir() and path.name.startswith(reuse_stamp) and path.name.endswith(f"_{stem}")
    )
    if not matches:
        raise FileNotFoundError(
            f"No existing run found for config '{config_path.name}' with reuse stamp '{reuse_stamp}' in '{artifact_root}'."
        )
    return matches[-1]


def _last_finite(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite[-1])


def _build_record(config_path: Path, run_name: str, run_dir: Path) -> dict[str, object]:
    manifest = read_json(run_dir / "manifest.json")
    history = read_json(run_dir / "history.json")
    resolved = load_train_config(run_dir / "config.resolved.yaml")
    parameterization = dict(resolved.model.network.get("parameterization", {}))
    backbone = dict(resolved.model.network.get("backbone", {}))
    pde_sampling = dict(resolved.physics.pde_sampling or {})
    scheduling = resolved.training.scheduling or {}
    if isinstance(scheduling, dict):
        stages = list(scheduling.get("stages", []) or [])
    else:
        stages = list(getattr(scheduling, "stages", []) or [])
    baseline_stage = stages[1] if len(stages) > 1 else None
    if isinstance(baseline_stage, dict):
        baseline_weights = dict(baseline_stage.get("weights", {}) or {})
    else:
        baseline_weights = dict(getattr(baseline_stage, "weights", {}) or {}) if baseline_stage is not None else {}
    return {
        "config_file": str(config_path),
        "experiment": manifest["experiment"],
        "run_name": run_name,
        "run_dir": str(run_dir),
        "architecture": manifest["architecture"],
        "formulation": manifest["formulation"],
        "best_val_doppler_rmse": float(manifest.get("best_val_doppler_rmse", float("nan"))),
        "last_val_doppler_rmse": _last_finite(history.get("val_doppler_rmse", [])),
        "last_val_inner_doppler_rmse": _last_finite(history.get("val_inner_doppler_rmse", [])),
        "last_val_outer_doppler_rmse": _last_finite(history.get("val_outer_doppler_rmse", [])),
        "output_scale": parameterization.get("output_scale"),
        "width": float(backbone.get("width", float("nan"))),
        "depth": float(backbone.get("depth", float("nan"))),
        "nblocks": float(backbone.get("nblocks", float("nan"))),
        "pde_samples": float(pde_sampling.get("sample_count", float("nan"))),
        "div_weight": float(baseline_weights.get("div", float("nan"))),
        "learning_rate": float(resolved.training.learning_rate),
        "epochs": int(resolved.training.epochs),
        "doppler_comparison_plot": str(run_dir / "doppler_comparison.png") if (run_dir / "doppler_comparison.png").exists() else None,
        "history": history,
    }


def _shared_positive_limits(series_list: list[np.ndarray]) -> tuple[float, float]:
    positive = []
    for series in series_list:
        values = np.asarray(series, dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size > 0:
            positive.append(values)
    if not positive:
        return 1e-4, 1.0
    merged = np.concatenate(positive)
    ymin = float(np.nanmin(merged))
    ymax = float(np.nanmax(merged))
    if ymin == ymax:
        return max(ymin / 10.0, 1e-12), ymax * 10.0
    ymin = max(ymin, ymax / 1e4)
    return ymin, ymax


def _plot_comparison(records: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    train_series = []
    val_series = []

    for record in records:
        history = record["history"]
        epochs = np.asarray(history["epoch"], dtype=np.int32)
        train_data = np.asarray(history["train_data_loss"], dtype=np.float64)
        val_rmse = np.asarray(history["val_doppler_rmse"], dtype=np.float64)
        label = str(record["run_name"])

        axes[0].plot(epochs, train_data, label=label)
        valid = np.isfinite(val_rmse)
        axes[1].plot(epochs[valid], val_rmse[valid], marker="o", markersize=3, linewidth=1.2, label=label)

        train_series.append(train_data)
        val_series.append(val_rmse)

    train_ymin, train_ymax = _shared_positive_limits(train_series)
    val_ymin, val_ymax = _shared_positive_limits(val_series)

    axes[0].set_title("Train Data Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].set_ylim(train_ymin, train_ymax)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].set_yscale("log")
    axes[1].set_ylim(val_ymin, val_ymax)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_doppler_comparison_grid(records: list[dict[str, object]], path: Path) -> None:
    image_records = [record for record in records if record.get("doppler_comparison_plot")]

    if not image_records:
        return

    ncols = min(2, len(image_records))
    nrows = int(math.ceil(len(image_records) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    axes_array = np.atleast_1d(axes).ravel()

    for axis, record in zip(axes_array, image_records):
        image = plt.imread(str(record["doppler_comparison_plot"]))
        axis.imshow(image)
        axis.set_title(
            f"{record['run_name']}\nval={float(record['best_val_doppler_rmse']):.3e}",
            fontsize=10,
        )
        axis.axis("off")

    for axis in axes_array[len(image_records):]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_metric_by_config(records: list[dict[str, object]], path: Path) -> None:
    ordered = sorted(records, key=lambda item: float(item.get("best_val_doppler_rmse", float("inf"))))
    labels = [Path(str(record["config_file"])).stem.replace("respinn_dns_", "") for record in ordered]
    best = np.asarray([float(record.get("best_val_doppler_rmse", float("nan"))) for record in ordered], dtype=np.float64)
    last = np.asarray([float(record.get("last_val_doppler_rmse", float("nan"))) for record in ordered], dtype=np.float64)
    gap = np.maximum(last - best, 0.0)

    fig, axis = plt.subplots(figsize=(max(10, 1.15 * len(labels)), 5.8))
    x = np.arange(len(labels), dtype=np.int32)

    axis.errorbar(
        x,
        best,
        yerr=gap,
        fmt="o",
        color="tab:blue",
        ecolor="tab:orange",
        elinewidth=1.5,
        capsize=4,
    )
    axis.plot(x, best, color="tab:blue", linewidth=1.2, alpha=0.7)
    axis.set_xticks(x, labels=labels, rotation=30, ha="right")
    axis.set_ylabel("best_val_doppler_rmse")
    axis.set_title("Best Validation RMSE with Last-Best Gap")
    axis.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _record_hparam_family(record: dict[str, object]) -> str | None:
    stem = Path(str(record["config_file"])).stem

    for family in ("div", "lr", "scale", "pde", "skip", "width"):
        token = f"_sweep_{family}"
        if token in stem:
            return family

    return None


def _record_hparam_x(record: dict[str, object]) -> tuple[float, str]:
    stem = Path(str(record["config_file"])).stem

    if "_sweep_div" in stem:
        value = float(str(stem.split("_sweep_div", 1)[1]).replace("e", "e"))
        return value, f"{value:.0e}"

    if "_sweep_lr" in stem:
        suffix = stem.split("_sweep_lr", 1)[1]
        mapping = {"3e4": 3e-4, "3e3": 3e-3}
        value = mapping.get(suffix, float("nan"))
        return value, suffix

    if "_sweep_pde" in stem:
        suffix = stem.split("_sweep_pde", 1)[1]
        value = float(suffix)
        return value, suffix

    if "_sweep_scale" in stem:
        suffix = stem.split("_sweep_scale", 1)[1]
        mapping = {"0p1": 0.1, "10": 10.0}
        value = mapping.get(suffix, 1.0)
        return value, f"scale{suffix}"

    if "_sweep_skip" in stem:
        suffix = stem.split("_sweep_skip", 1)[1]
        mapping = {"_down": 0.0, "_up": 2.0}
        value = mapping.get(suffix, 1.0)
        return value, suffix.lstrip("_")

    if "_sweep_width" in stem:
        suffix = stem.split("_sweep_width", 1)[1]
        value = float(suffix)
        return value, suffix

    return 1.0, "ref"


def _baseline_x(record: dict[str, object], family: str) -> tuple[float, str]:
    if family == "div":
        return float(record["div_weight"]), "ref"
    if family == "lr":
        return float(record["learning_rate"]), "ref"
    if family == "pde":
        return float(record["pde_samples"]), "ref"
    if family == "width":
        return float(record["width"]), "ref"
    if family == "skip":
        return 1.0, "ref"
    return 1.0, "ref"


def _plot_metric_vs_hparam(records: list[dict[str, object]], metric_name: str, path: Path) -> None:
    families = ("div", "lr", "scale", "pde", "width", "skip")
    family_labels = {
        "div": "DIV Weight",
        "lr": "Learning Rate",
        "scale": "Output Scale",
        "pde": "PDE Samples",
        "width": "Network Width",
        "skip": "Residual Blocks",
    }

    baseline = next((record for record in records if Path(str(record["config_file"])).stem.endswith("_sweep_ref")), None)
    if baseline is None:
        return

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    axes_array = axes.ravel()

    for axis, family in zip(axes_array, families):
        variants = [record for record in records if _record_hparam_family(record) == family]
        if not variants:
            axis.axis("off")
            continue

        family_records = [baseline] + variants

        points = []
        for record in family_records:
            if record is baseline:
                x_value, x_label = _baseline_x(baseline, family)
            else:
                x_value, x_label = _record_hparam_x(record)

            points.append((x_value, float(record[metric_name]), x_label, record is baseline))

        points = [
            item
            for item in points
            if np.isfinite(item[0]) and np.isfinite(item[1]) and (family not in {"div", "lr"} or item[0] > 0.0)
        ]
        if not points:
            axis.set_title(family_labels[family])
            axis.set_ylabel(metric_name)
            axis.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=axis.transAxes)
            axis.grid(True, alpha=0.3)
            continue

        points.sort(key=lambda item: item[0])
        x = [item[0] for item in points]
        y = [item[1] for item in points]
        labels = [item[2] for item in points]
        baseline_index = next((index for index, item in enumerate(points) if item[3]), None)

        axis.plot(x, y, color="tab:blue", linewidth=1.5)
        axis.scatter(x, y, color="tab:blue", s=40)
        if baseline_index is not None:
            axis.scatter([x[baseline_index]], [y[baseline_index]], color="tab:red", s=70, zorder=3)

        if family in {"div", "lr"}:
            axis.set_xscale("log")

        axis.set_title(family_labels[family])
        axis.set_ylabel(metric_name)
        axis.grid(True, alpha=0.3)
        axis.set_xticks(x, labels=labels)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    summary_root = Path(args.summary_dir).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = summary_root / stamp
    sweep_dir.mkdir(parents=True, exist_ok=True)

    worker_script = PROJECT_ROOT / "scripts" / "train.py"
    records: list[dict[str, object]] = []

    for index, config_value in enumerate(args.config_file):
        config_path = Path(config_value).expanduser().resolve()
        if args.reuse_stamp is None:
            run_name = f"{stamp}_{config_path.stem}"
            command = [sys.executable, str(worker_script), "--config-file", str(config_path), "--run-name", run_name]

            if args.epochs is not None:
                command.extend(["--epochs", str(args.epochs)])

            print("")
            print(format_summary_table("Sweep Run", [("Index", index), ("Config", config_path.name), ("Run name", run_name)]))
            subprocess.run(command, check=True)
        else:
            run_name = _resolve_reuse_run_name(config_path, args.reuse_stamp)
            print("")
            print(format_summary_table("Sweep Summary Reuse", [("Index", index), ("Config", config_path.name), ("Run name", run_name)]))

        run_dir = _artifact_dir_for_run(config_path, run_name)
        records.append(_build_record(config_path, run_name, run_dir))

    ranked = sorted(records, key=lambda item: float(item.get(args.metric, float("inf"))))

    comparison_plot = sweep_dir / "comparison.png"
    _plot_comparison(ranked, comparison_plot)
    doppler_grid_plot = sweep_dir / "doppler_comparison_grid.png"
    _plot_doppler_comparison_grid(ranked, doppler_grid_plot)
    metric_by_config_plot = sweep_dir / "metric_by_config_best_gap.png"
    _plot_metric_by_config(ranked, metric_by_config_plot)
    metric_vs_hparam_plot = sweep_dir / "metric_vs_hparam.png"
    _plot_metric_vs_hparam(ranked, args.metric, metric_vs_hparam_plot)
    metric_vs_hparam_last_plot = sweep_dir / "metric_vs_hparam_last.png"
    _plot_metric_vs_hparam(ranked, "last_val_doppler_rmse", metric_vs_hparam_last_plot)
    metric_vs_hparam_best_plot = sweep_dir / "metric_vs_hparam_best.png"
    _plot_metric_vs_hparam(ranked, "best_val_doppler_rmse", metric_vs_hparam_best_plot)

    summary_json = sweep_dir / "leaderboard.json"
    summary_csv = sweep_dir / "leaderboard.csv"

    serializable = []
    for record in ranked:
        payload = dict(record)
        payload.pop("history", None)
        serializable.append(payload)

    summary_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config_file",
                "experiment",
                "run_name",
                "run_dir",
                "architecture",
                "formulation",
                "best_val_doppler_rmse",
                "last_val_doppler_rmse",
                "last_val_inner_doppler_rmse",
                "last_val_outer_doppler_rmse",
                "output_scale",
                "width",
                "depth",
                "nblocks",
                "pde_samples",
                "div_weight",
                "learning_rate",
                "epochs",
                "doppler_comparison_plot",
            ],
        )
        writer.writeheader()
        for record in serializable:
            writer.writerow(record)

    best = ranked[0]
    print("")
    print(
        format_summary_table(
            "Sweep Completed",
            [
                ("Metric", args.metric),
                ("Best run", best["run_name"]),
                ("Best value", f"{float(best[args.metric]):.4e}"),
                ("Leaderboard JSON", summary_json),
                ("Leaderboard CSV", summary_csv),
                ("Comparison plot", comparison_plot),
                ("Doppler grid", doppler_grid_plot if doppler_grid_plot.exists() else "-"),
                ("Metric by config", metric_by_config_plot if metric_by_config_plot.exists() else "-"),
                ("Metric vs hparam", metric_vs_hparam_plot if metric_vs_hparam_plot.exists() else "-"),
                ("Metric vs hparam last", metric_vs_hparam_last_plot if metric_vs_hparam_last_plot.exists() else "-"),
                ("Metric vs hparam best", metric_vs_hparam_best_plot if metric_vs_hparam_best_plot.exists() else "-"),
            ],
        )
    )


if __name__ == "__main__":
    main()
