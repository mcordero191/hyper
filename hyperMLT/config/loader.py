from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .schema_infer import InferConfig
from .schema_train import TrainConfig


def load_mapping_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        mapping = json.loads(text)
    else:
        mapping = yaml.safe_load(text)
    return mapping or {}


def load_train_config(path: str | Path) -> TrainConfig:
    config = TrainConfig.from_mapping(load_mapping_file(path))
    config._config_path = str(Path(path).expanduser().resolve())
    return config


def load_infer_config(path: str | Path) -> InferConfig:
    config = InferConfig.from_mapping(load_mapping_file(path))
    config._config_path = str(Path(path).expanduser().resolve())
    return config
