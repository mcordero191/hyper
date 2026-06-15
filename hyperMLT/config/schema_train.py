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
class DatasetsSection:
    primary: dict[str, Any] = field(default_factory=dict)
    auxiliary: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DomainSection:
    time: dict[str, Any] = field(default_factory=dict)
    region: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSection:
    architecture: str = "deepgreen"
    network: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysicsSection:
    formulation: str = "continuity_only"
    state: str = "uvw"
    residual_weights: dict[str, Any] = field(default_factory=dict)
    pde_sampling: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingSection:
    epochs: int = 1000
    learning_rate: float = 1e-4
    optimizer: str = "Adam"
    data_loss: str = "mae"
    gradient_diagnostics: bool = True
    random_seed: int = 1234
    scheduling: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputSection:
    root_dir: str = "./runs"
    write_plots: bool = True
    verbose: bool = True


@dataclass
class MultidaySection:
    enabled: bool = False
    file_order: str = "chronological"
    first_day: dict[str, Any] = field(default_factory=dict)
    subsequent_days: dict[str, Any] = field(default_factory=dict)
    shared_trunk: dict[str, Any] = field(default_factory=dict)
    day_runs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    datasets: DatasetsSection = field(default_factory=DatasetsSection)
    domain: DomainSection = field(default_factory=DomainSection)
    model: ModelSection = field(default_factory=ModelSection)
    physics: PhysicsSection = field(default_factory=PhysicsSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    output: OutputSection = field(default_factory=OutputSection)
    multiday: MultidaySection = field(default_factory=MultidaySection)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None = None) -> "TrainConfig":
        normalized = copy.deepcopy(mapping or {})
        _validate_top_level_sections(normalized)
        config = cls()
        config._raw_mapping = copy.deepcopy(normalized)
        for section_name in ("experiment", "datasets", "domain", "model", "physics", "training", "output", "multiday"):
            _apply_section(getattr(config, section_name), normalized.get(section_name, {}))
        return config


def _validate_top_level_sections(mapping: dict[str, Any]) -> None:
    allowed = {"experiment", "datasets", "domain", "model", "physics", "training", "output", "multiday"}
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown hyperMLT training config sections: {unknown}")


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
