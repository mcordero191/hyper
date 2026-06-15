from __future__ import annotations

from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras


def _resolve_activation(name: str):

    normalized = str(name).lower()

    if normalized in {"sine", "sin"}:
        return tf.sin

    if normalized == "tanh":
        return tf.tanh

    if normalized == "swish":
        return keras.activations.swish

    if normalized == "relu":
        return keras.activations.relu

    return keras.activations.linear


def _scaling_values(shape_out: int, output_scale: float) -> List[float]:

    values = [float(output_scale)] * min(shape_out, 3)

    if shape_out > 3:
        values.extend([1.0] * (shape_out - 3))

    return values


class MLPStack(keras.layers.Layer):
    def __init__(self, width: int, depth: int, activation: str, name: str | None = None):
        super().__init__(name=name)

        self.activation = _resolve_activation(activation)
        self.hidden = [
            keras.layers.Dense(width, activation=self.activation, name=f"{name}_dense_{i}")
            for i in range(depth)
        ]

    def call(self, inputs):

        x = inputs

        for layer in self.hidden:
            x = layer(x)

        return x
