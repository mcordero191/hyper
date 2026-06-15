from __future__ import annotations

from .deepgreen import build_deepgreen_model
from .respinn import build_respinn_model


def describe_model_family(architecture: str) -> str:
    name = str(architecture).lower()
    if name == "deepgreen":
        return "DeepGreen"
    if name == "respinn":
        return "RESPINN"
    raise ValueError(f"Unsupported hyperMLT model architecture '{architecture}'")


def build_model(model_config, shape_out: int = 3):
    name = str(model_config.architecture).lower()
    if name == "deepgreen":
        return build_deepgreen_model(model_config.network, shape_out=shape_out)
    if name == "respinn":
        return build_respinn_model(model_config.network, shape_out=shape_out)
    raise ValueError(f"Unsupported hyperMLT model architecture '{model_config.architecture}'")
