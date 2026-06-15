from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from .layers import SIRENLayerInitializer
from .multioperator import _resolve_activation


def _fixed_output_scale(shape_out: int) -> list[float]:

    values = [1.0, 1.0, 0.1]

    if shape_out <= 3:
        return values[:shape_out]

    return values + [1.0] * (shape_out - 3)


def _normalize_multiscale_factors(values) -> list[float]:

    if values is None:
        return [1.0]

    if isinstance(values, (int, float)):
        values = [float(values)]
    else:
        values = [float(value) for value in values]

    if not values:
        return [1.0]

    return values


class MultiScaleInputEncoder(keras.layers.Layer):
    def __init__(self, factors: list[float], name: str | None = None):
        super().__init__(name=name)

        self.factors = tuple(float(value) for value in factors)

    def call(self, inputs):

        encoded = [inputs * factor for factor in self.factors]

        if len(encoded) == 1:
            return encoded[0]

        return tf.concat(encoded, axis=1)


def _resolve_kernel_initializer(name: str, *, activation: str, w0: float) -> str | keras.initializers.Initializer:

    normalized = str(name).strip().lower()
    activation_name = str(activation).strip().lower()

    if normalized == "siren":
        if activation_name not in {"sine", "sin"}:
            raise ValueError("RESPINN initializer 'siren' is only valid with sine activation.")

        return SIRENLayerInitializer(w0=float(w0))

    if normalized in {"glorot_uniform", "xavier_uniform"}:
        return "glorot_uniform"

    if normalized in {"glorot_normal", "xavier_normal"}:
        return "glorot_normal"

    raise ValueError(
        f"Unsupported RESPINN initializer '{name}'. "
        "Expected 'glorot_uniform', 'glorot_normal', or 'siren'."
    )


class ResidualDenseBlock(keras.layers.Layer):
    def __init__(
        self,
        width: int,
        depth: int,
        activation: str = "sine",
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        name: str | None = None,
    ):
        super().__init__(name=name)

        self.width = int(width)
        self.depth = max(int(depth), 1)
        self.activation = _resolve_activation(activation)

        self.hidden_layers = [
            keras.layers.Dense(
                self.width,
                activation=None,
                kernel_initializer=kernel_initializer,
                bias_initializer="zeros",
                name=f"{name}_dense_{index}" if name else f"dense_{index}",
            )
            for index in range(self.depth)
        ]

    def call(self, inputs, training: bool = False):
        del training

        hidden = inputs
        shortcut = inputs

        for layer in self.hidden_layers:
            hidden = self.activation(layer(hidden))

        return shortcut + hidden


class ResPINNDirect(keras.Model):
    def __init__(
        self,
        shape_out: int,
        width: int,
        depth: int,
        nblocks: int = 2,
        activation: str = "sine",
        multiscale_factors: list[float] | None = None,
        initializer: str = "glorot_uniform",
        w0: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.shape_out = int(shape_out)
        self.scale_values = tf.constant(_fixed_output_scale(self.shape_out), dtype=tf.float32)
        self.multiscale_factors = _normalize_multiscale_factors(multiscale_factors)
        self.input_encoder = MultiScaleInputEncoder(self.multiscale_factors, name="multiscale_input")
        kernel_initializer = _resolve_kernel_initializer(initializer, activation=activation, w0=w0)

        self.input_layer = keras.layers.Dense(
            int(width),
            activation=None,
            kernel_initializer=kernel_initializer,
            bias_initializer="zeros",
            name="input_projection",
        )

        self.activation = _resolve_activation(activation)
        self.blocks = [
            ResidualDenseBlock(
                width=int(width),
                depth=int(depth),
                activation=activation,
                kernel_initializer=kernel_initializer,
                name=f"block_{index}",
            )
            for index in range(max(int(nblocks), 1))
        ]

        self.output_layer = keras.layers.Dense(
            self.shape_out,
            activation=None,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="output_projection",
        )

    def predict_fields(self, inputs, training: bool = False):

        encoded_inputs = self.input_encoder(inputs)
        hidden = self.activation(self.input_layer(encoded_inputs))

        for block in self.blocks:
            hidden = block(hidden, training=training)

        outputs = self.output_layer(hidden)

        return outputs * self.scale_values

    def call(self, inputs, training: bool = False, **_kwargs):

        return self.predict_fields(inputs, training=training)


def build_respinn_model(network: dict[str, object], shape_out: int = 3):

    parameterization = network.get("parameterization", {})
    backbone = network.get("backbone", {})

    normalized = str(parameterization.get("wind", "direct")).lower()

    if normalized != "direct":
        raise ValueError(f"Unsupported RESPINN parameterization '{normalized}' in hyperMLT.")

    return ResPINNDirect(
        shape_out=shape_out,
        width=int(backbone.get("width", network.get("width", 64))),
        depth=int(backbone.get("depth", network.get("depth", 2))),
        nblocks=int(backbone.get("nblocks", 2)),
        activation=str(backbone.get("activation", "sine")),
        multiscale_factors=backbone.get("multiscale_factors", [1.0]),
        initializer=str(backbone.get("initializer", "glorot_uniform")),
        w0=float(backbone.get("w0", 1.0)),
    )
