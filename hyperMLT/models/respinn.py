from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .layers import SIRENLayerInitializer
from .multioperator import _resolve_activation, _scaling_values


def _respinn_scaling_values(shape_out: int, output_scale) -> list[float]:

    if isinstance(output_scale, (list, tuple, np.ndarray)):
        values = [float(value) for value in output_scale]

        if len(values) != int(shape_out):
            raise ValueError(
                f"RESPINN output_scale list must have length {shape_out}, got {len(values)}."
            )

        return values

    if output_scale is None:
        values = [1.0, 1.0, 0.1]

        if shape_out <= 3:
            return values[:shape_out]

        return values + [1.0] * (shape_out - 3)

    return _scaling_values(int(shape_out), float(output_scale))


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
        trainable_alpha: bool = False,
        alpha_init: float = 0.05,
        alpha_activation: str = "softplus",
        name: str | None = None,
    ):
        super().__init__(name=name)

        self.width = int(width)
        self.depth = max(int(depth), 1)
        self.activation = _resolve_activation(activation)
        self.trainable_alpha = bool(trainable_alpha)
        alpha_init = float(max(alpha_init, 1.0e-6))
        self.alpha_init = alpha_init
        self.alpha_activation = str(alpha_activation).strip().lower()

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

        if self.trainable_alpha:
            if self.alpha_activation == "softplus":
                raw_init = np.log(np.expm1(self.alpha_init))
            elif self.alpha_activation == "exp":
                raw_init = np.log(self.alpha_init)
            else:
                raise ValueError(
                    f"Unsupported RESPINN residual alpha activation '{alpha_activation}'. "
                    "Expected 'softplus' or 'exp'."
                )
            self.raw_alpha = self.add_weight(
                name=f"{name}_raw_alpha" if name else "raw_alpha",
                shape=(),
                initializer=keras.initializers.Constant(raw_init),
                trainable=True,
            )
        else:
            self.raw_alpha = None

    def residual_scale(self) -> tf.Tensor:
        if self.raw_alpha is None:
            return tf.ones((), dtype=tf.float32)
        if self.alpha_activation == "softplus":
            return tf.nn.softplus(self.raw_alpha)
        if self.alpha_activation == "exp":
            return tf.exp(self.raw_alpha)
        raise ValueError(
            f"Unsupported RESPINN residual alpha activation '{self.alpha_activation}'. "
            "Expected 'softplus' or 'exp'."
        )

    def call(self, inputs, training: bool = False):
        del training

        hidden = inputs
        shortcut = inputs

        for layer in self.hidden_layers:
            hidden = self.activation(layer(hidden))

        return shortcut + self.residual_scale() * hidden


class ResPINNDirect(keras.Model):
    def __init__(
        self,
        shape_out: int,
        width: int,
        depth: int,
        nblocks: int = 2,
        use_skip_connections: bool = True,
        trainable_skip_scale: bool = False,
        skip_scale_init: float = 0.05,
        skip_scale_activation: str = "softplus",
        activation: str = "sine",
        multiscale_factors: list[float] | None = None,
        initializer: str = "glorot_uniform",
        w0: float = 1.0,
        output_scale = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.shape_out = int(shape_out)
        self.scale_values = tf.constant(_respinn_scaling_values(self.shape_out, output_scale), dtype=tf.float32)
        self.multiscale_factors = _normalize_multiscale_factors(multiscale_factors)
        self.input_encoder = MultiScaleInputEncoder(self.multiscale_factors, name="multiscale_input")
        kernel_initializer = _resolve_kernel_initializer(initializer, activation=activation, w0=w0)
        self.use_skip_connections = bool(use_skip_connections)

        self.input_layer = keras.layers.Dense(
            int(width),
            activation=None,
            kernel_initializer=kernel_initializer,
            bias_initializer="zeros",
            name="input_projection",
        )

        self.activation = _resolve_activation(activation)
        if self.use_skip_connections:
            self.blocks = [
                ResidualDenseBlock(
                    width=int(width),
                    depth=int(depth),
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    trainable_alpha=bool(trainable_skip_scale),
                    alpha_init=float(skip_scale_init),
                    alpha_activation=str(skip_scale_activation),
                    name=f"block_{index}",
                )
                for index in range(max(int(nblocks), 1))
            ]
        else:
            self.blocks = [
                keras.layers.Dense(
                    int(width),
                    activation=None,
                    kernel_initializer=kernel_initializer,
                    bias_initializer="zeros",
                    name=f"dense_{layer_index}",
                )
                for layer_index in range(max(int(depth), 1))
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

        if self.use_skip_connections:
            for block in self.blocks:
                hidden = block(hidden, training=training)
        else:
            for block in self.blocks:
                hidden = self.activation(block(hidden))

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
        use_skip_connections=bool(backbone.get("use_skip_connections", True)),
        trainable_skip_scale=bool(backbone.get("trainable_skip_scale", False)),
        skip_scale_init=float(backbone.get("skip_scale_init", 0.05)),
        skip_scale_activation=str(backbone.get("skip_scale_activation", "softplus")),
        activation=str(backbone.get("activation", "sine")),
        multiscale_factors=backbone.get("multiscale_factors", [1.0]),
        initializer=str(backbone.get("initializer", "glorot_uniform")),
        w0=float(backbone.get("w0", 1.0)),
        output_scale=parameterization.get("output_scale"),
    )
