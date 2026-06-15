from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExperimentSection:
    name: str = "hyperMLT"
    version: str = "0.1.0"
    description: str = ""


@dataclass
class ModelSection:
    artifact_path: str | None = None
    log_index: str | int | None = None


@dataclass
class DomainSection:
    grid: dict[str, Any] = field(default_factory=dict)
    points: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceSection:
    gradients: bool = False
    plotting: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputSection:
    root_dir: str = "./runs/winds"
    format: str = "hdf5"


@dataclass
class InferConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    model: ModelSection = field(default_factory=ModelSection)
    domain: DomainSection = field(default_factory=DomainSection)
    inference: InferenceSection = field(default_factory=InferenceSection)
    output: OutputSection = field(default_factory=OutputSection)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None = None) -> "InferConfig":
        normalized = copy.deepcopy(mapping or {})
        _validate_top_level_sections(normalized)
        config = cls()
        config._raw_mapping = copy.deepcopy(normalized)
        for section_name in ("experiment", "model", "domain", "inference", "output"):
            _apply_section(getattr(config, section_name), normalized.get(section_name, {}))
        return config


def _validate_top_level_sections(mapping: dict[str, Any]) -> None:
    allowed = {"experiment", "model", "domain", "inference", "output"}
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown hyperMLT inference config sections: {unknown}")


def _apply_section(target: Any, values: dict[str, Any]) -> None:
    if not isinstance(values, dict):
        raise TypeError(f"Expected mapping for section {type(target).__name__}")
    for key, value in values.items():
        if not hasattr(target, key):
            raise ValueError(f"Unknown config field '{key}' in section '{type(target).__name__}'")
        current = getattr(target, key)
        if hasattr(current, "__dataclass_fields__"):
            if not isinstance(value, dict):
                raise TypeError(f"Expected mapping for nested section '{key}'")
            _apply_section(current, value)
        else:
            setattr(target, key, value)
