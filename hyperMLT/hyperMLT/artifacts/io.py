from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from hyperMLT.config import load_train_config
from hyperMLT.models import build_model


def create_run_directory(root_dir: str | Path, experiment_name: str, run_name: str) -> Path:
    run_dir = Path(root_dir).expanduser().resolve() / experiment_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_resolved_config(run_dir: str | Path, config: Any) -> Path:
    path = Path(run_dir) / "config.resolved.yaml"
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    return path


def write_json(run_dir: str | Path, name: str, payload: dict[str, Any]) -> Path:
    path = Path(run_dir) / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_artifact_dir(path: str | Path) -> Path:
    artifact_dir = Path(path).expanduser().resolve()
    if (artifact_dir / "manifest.json").exists():
        return artifact_dir
    candidates = sorted(
        candidate
        for candidate in artifact_dir.rglob("manifest.json")
        if candidate.parent.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"No manifest.json found under artifact path '{artifact_dir}'.")
    return candidates[-1].parent


def load_trained_model(artifact_path: str | Path):
    artifact_dir = resolve_artifact_dir(artifact_path)
    config = load_train_config(artifact_dir / "config.resolved.yaml")
    model = build_model(config.model, shape_out=3)
    manifest = read_json(artifact_dir / "manifest.json")
    norm = manifest["normalization"]
    dummy = [[norm["lower_bounds"][0], norm["lower_bounds"][1], norm["lower_bounds"][2], norm["lower_bounds"][3]]]
    import tensorflow as tf

    _ = model(tf.convert_to_tensor(dummy, dtype=tf.float32), training=False)
    model.load_weights(artifact_dir / manifest["weights_file"])
    return artifact_dir, config, manifest, model
