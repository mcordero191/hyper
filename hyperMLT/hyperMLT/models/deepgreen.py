from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .layers import Scaler
from .multioperator import MLPStack, _resolve_activation, _scaling_values


def _make_stack(width: int, depth: int, activation: str, name: str):

    return MLPStack(width=int(width), depth=int(depth), activation=activation, name=name)


def _matern52(distance: tf.Tensor) -> tf.Tensor:

    sqrt5 = tf.sqrt(tf.constant(5.0, dtype=distance.dtype))
    r = sqrt5 * distance

    return (1.0 + r + tf.square(r) / 3.0) * tf.exp(-r)


def _yukawa(distance: tf.Tensor, epsilon: float = 1e-3) -> tf.Tensor:

    softened = tf.sqrt(tf.square(distance) + tf.cast(epsilon**2, distance.dtype))

    return tf.exp(-softened) / softened


def _deepgreen_scaling_values(shape_out: int, output_scale) -> list[float]:

    if isinstance(output_scale, (list, tuple, np.ndarray)):
        values = [float(value) for value in output_scale]

        if len(values) != int(shape_out):
            raise ValueError(
                f"DeepGreen output_scale list must have length {shape_out}, got {len(values)}."
            )

        return values

    return _scaling_values(int(shape_out), float(output_scale))


class ResidualMLPStack(keras.layers.Layer):
    def __init__(self, width: int, depth: int, activation: str, name: str | None = None):
        super().__init__(name=name)

        resolved_activation = _resolve_activation(activation)

        self.input_proj = keras.layers.Dense(
            int(width),
            activation=resolved_activation,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name=f"{name}_input_proj",
        )

        self.blocks = []
        self.block_scales = []

        for index in range(max(int(depth) - 1, 0)):
            self.blocks.append(
                keras.layers.Dense(
                    int(width),
                    activation=resolved_activation,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"{name}_dense_{index}",
                )
            )
            self.block_scales.append(
                self.add_weight(
                    name=f"{name}_residual_scale_param_{index}",
                    shape=(),
                    initializer=keras.initializers.Constant(0.05),
                    trainable=True,
                )
            )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        x = self.input_proj(inputs)

        for dense_layer, scale_param in zip(self.blocks, self.block_scales):
            residual = dense_layer(x)
            alpha = tf.nn.relu(scale_param)
            x = x + alpha * residual

        return x

    def gate_values(self) -> tf.Tensor:

        if not self.block_scales:
            return tf.zeros((0,), dtype=tf.float32)

        return tf.stack([tf.nn.relu(scale_param) for scale_param in self.block_scales], axis=0)


def _make_residual_stack(width: int, depth: int, activation: str, name: str):

    return ResidualMLPStack(width=int(width), depth=int(depth), activation=activation, name=name)


class DeepGreenEncoder(keras.layers.Layer):
    def __init__(
        self,
        width: int,
        depth: int,
        latent_dim: int,
        activation: str,
        use_skip_connections: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name)

        stack_builder = _make_residual_stack if use_skip_connections else _make_stack
        self.stack = stack_builder(width=width, depth=depth, activation=activation, name=f"{name}_stack")

        self.head = keras.layers.Dense(
            int(latent_dim),
            activation=None,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name=f"{name}_latent_head",
        )

    def call(self, coords: tf.Tensor) -> tf.Tensor:

        latent = self.head(self.stack(coords))

        return latent


class DeepGreenKernelFunction(keras.layers.Layer):
    def __init__(
        self,
        latent_dim: int,
        n_green_functions: int,
        kernel_type: str = "matern52",
        name: str | None = None,
    ):
        super().__init__(name=name)

        self.latent_dim = int(latent_dim)
        self.n_green_functions = int(n_green_functions)
        self.kernel_type = str(kernel_type).lower()

        self.centers_raw = self.add_weight(
            name=f"{name}_centers" if name else "centers",
            shape=(self.n_green_functions, self.latent_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.25),
            trainable=True,
        )

        self.log_length_scales = self.add_weight(
            name=f"{name}_log_lengths" if name else "log_lengths",
            shape=(self.n_green_functions, self.latent_dim),
            initializer=keras.initializers.Constant(np.log(0.5)),
            trainable=True,
        )

    def centers(self) -> tf.Tensor:

        return self.centers_raw

    def length_scales(self) -> tf.Tensor:

        return tf.nn.softplus(self.log_length_scales) + 1e-3

    def call(
        self,
        theta: tf.Tensor,
        coeffs: tf.Tensor,
        shifts: tf.Tensor,
        stretches: tf.Tensor,
        gains: tf.Tensor,
    ) -> tf.Tensor:

        theta = tf.expand_dims(theta, axis=1)
        transformed = stretches * theta + shifts

        centers = self.centers()
        length_scales = self.length_scales()

        diffs = (
            tf.expand_dims(transformed, axis=2) - tf.expand_dims(tf.expand_dims(centers, axis=0), axis=0)
        ) / tf.expand_dims(tf.expand_dims(length_scales, axis=0), axis=0)

        distances = tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-8)

        if self.kernel_type in {"yukawa", "screened_poisson", "screened-poisson"}:
            kernels = _yukawa(distances)
        else:
            kernels = _matern52(distances)

        weighted = gains * coeffs * kernels

        return tf.reduce_sum(weighted, axis=1)


class DeepGreenSourceGenerator(keras.layers.Layer):
    def __init__(
        self,
        width: int,
        depth: int,
        latent_dim: int,
        n_green_functions: int,
        n_sources: int,
        activation: str,
        name: str | None = None,
    ):
        super().__init__(name=name)

        self.latent_dim = int(latent_dim)
        self.n_green_functions = int(n_green_functions)
        self.n_sources = int(n_sources)

        self.stack = _make_stack(width=width, depth=depth, activation=activation, name=f"{name}_stack")

        self.coeff_head = keras.layers.Dense(
            self.n_sources * self.n_green_functions,
            activation=None,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name=f"{name}_coefficients",
        )
        self.shift_head = keras.layers.Dense(
            self.n_sources * self.latent_dim,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name=f"{name}_shifts",
        )
        self.stretch_head = keras.layers.Dense(
            self.n_sources * self.latent_dim,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name=f"{name}_stretches",
        )
        self.gain_head = keras.layers.Dense(
            self.n_sources,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(0.5),
            name=f"{name}_gains",
        )

    def call(self, coords: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        hidden = self.stack(coords)

        coeffs = tf.reshape(self.coeff_head(hidden), (-1, self.n_sources, self.n_green_functions))

        shifts = tf.reshape(self.shift_head(hidden), (-1, self.n_sources, self.latent_dim))
        shifts = 0.25 * tf.tanh(shifts)

        stretches = tf.reshape(self.stretch_head(hidden), (-1, self.n_sources, self.latent_dim))
        stretches = 1.0 + 0.25 * tf.tanh(stretches)

        gains = tf.expand_dims(self.gain_head(hidden), axis=-1)

        return coeffs, shifts, stretches, gains

    def regularization(self) -> tf.Tensor:

        weights = [tf.reduce_mean(tf.square(weight)) for weight in self.trainable_weights]

        if not weights:
            return tf.constant(0.0, dtype=tf.float32)

        return tf.add_n(weights) / tf.cast(len(weights), tf.float32)


class DeepGreenDecoder(keras.layers.Layer):
    def __init__(
        self,
        width: int,
        depth: int,
        input_dim: int,
        output_dim: int,
        activation: str,
        use_skip_connections: bool = False,
        output_scale: float | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)

        del input_dim

        stack_builder = _make_residual_stack if use_skip_connections else _make_stack
        self.stack = stack_builder(width=width, depth=depth, activation=activation, name=f"{name}_stack")

        self.head = keras.layers.Dense(
            int(output_dim),
            activation=None,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name=f"{name}_field_head",
        )

        self.output_scaler = None

        if output_scale is not None:
            self.output_scaler = Scaler(
                values=_deepgreen_scaling_values(int(output_dim), output_scale),
                add_bias=True,
                name=f"{name}_scaler",
            )

    def call(self, kernel_responses: tf.Tensor) -> tf.Tensor:

        outputs = self.head(self.stack(kernel_responses))

        if self.output_scaler is not None:
            outputs = self.output_scaler(outputs)

        return outputs


class DeepGreenSharedTrunk(keras.layers.Layer):
    def __init__(
        self,
        encoder_width: int,
        encoder_depth: int,
        latent_dim: int,
        n_green_functions: int,
        kernel_type: str,
        decoder_width: int,
        decoder_depth: int,
        output_dim: int,
        activation: str,
        encoder_skip_connections: bool = False,
        decoder_skip_connections: bool = False,
        output_scale: float | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)

        self.encoder = DeepGreenEncoder(
            width=encoder_width,
            depth=encoder_depth,
            latent_dim=latent_dim,
            activation=activation,
            use_skip_connections=encoder_skip_connections,
            name=f"{name}_encoder",
        )

        self.kernel_function = DeepGreenKernelFunction(
            latent_dim=latent_dim,
            n_green_functions=n_green_functions,
            kernel_type=kernel_type,
            name=f"{name}_kernel_function",
        )

        self.decoder = DeepGreenDecoder(
            width=decoder_width,
            depth=decoder_depth,
            input_dim=n_green_functions,
            output_dim=output_dim,
            activation=activation,
            use_skip_connections=decoder_skip_connections,
            output_scale=output_scale,
            name=f"{name}_decoder",
        )

    def call(
        self,
        coords: tf.Tensor,
        coeffs: tf.Tensor,
        shifts: tf.Tensor,
        stretches: tf.Tensor,
        gains: tf.Tensor,
    ) -> tf.Tensor:

        theta = self.encoder(coords)
        responses = self.kernel_function(theta, coeffs, shifts, stretches, gains)

        return self.decoder(responses)


class DeepGreenBackbone(keras.layers.Layer):
    def __init__(
        self,
        encoder_width: int,
        encoder_depth: int,
        source_width: int,
        source_depth: int,
        latent_dim: int,
        n_green_functions: int,
        n_sources: int,
        kernel_type: str,
        activation: str,
        decoder_width: int | None = None,
        decoder_depth: int | None = None,
        encoder_skip_connections: bool = False,
        decoder_skip_connections: bool = False,
        output_dim: int = 3,
        output_scale: float | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)

        self.trunk = DeepGreenSharedTrunk(
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            latent_dim=latent_dim,
            n_green_functions=n_green_functions,
            kernel_type=kernel_type,
            decoder_width=int(encoder_width if decoder_width is None else decoder_width),
            decoder_depth=int(encoder_depth if decoder_depth is None else decoder_depth),
            output_dim=output_dim,
            activation=activation,
            encoder_skip_connections=encoder_skip_connections,
            decoder_skip_connections=decoder_skip_connections,
            output_scale=output_scale,
            name=f"{name}_trunk",
        )

        self.source_generator = DeepGreenSourceGenerator(
            width=source_width,
            depth=source_depth,
            latent_dim=latent_dim,
            n_green_functions=n_green_functions,
            n_sources=n_sources,
            activation=activation,
            name=f"{name}_source_generator",
        )

    def advance_curriculum(self, total_epochs: int):

        del total_epochs

        return

    def call(self, coords: tf.Tensor) -> tf.Tensor:

        coeffs, shifts, stretches, gains = self.source_generator(coords[:, :4])

        return self.trunk(coords, coeffs, shifts, stretches, gains)

class DeepGreenBase(keras.Model):
    def __init__(
        self,
        learn_nu: bool = False,
        nu_init: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.learn_nu = bool(learn_nu)
        self.nu_init = float(nu_init)
        self.ena_hnet = 0

        self.nu = self.add_weight(
            name="Nu",
            shape=(),
            initializer=keras.initializers.Constant(self.nu_init),
            trainable=self.learn_nu,
        )

    def advance_curriculum(self, total_epochs):

        self.backbone.advance_curriculum(total_epochs)

class DeepGreenDirect(DeepGreenBase):
    def __init__(
        self,
        shape_out: int,
        width: int,
        depth: int,
        nblocks: int = 1,
        output_scale=1e2,
        latent_dim: int | None = None,
        transform_width: int | None = None,
        transform_depth: int | None = None,
        source_width: int | None = None,
        source_depth: int | None = None,
        decoder_width: int | None = None,
        decoder_depth: int | None = None,
        n_green_functions: int = 16,
        n_sources: int = 32,
        kernel_type: str = "matern52",
        state_dim: int = 8,
        activation: str = "tanh",
        encoder_skip_connections: bool = False,
        decoder_skip_connections: bool = False,
        family_names=None,
        learn_nu: bool = False,
        nu_init: float = 0.0,
        **kwargs,
    ):
        del nblocks, state_dim, family_names

        super().__init__(
            learn_nu=learn_nu,
            nu_init=nu_init,
            **kwargs,
        )

        transform_width = int(width if transform_width is None else transform_width)
        transform_depth = int(depth if transform_depth is None else transform_depth)
        source_width = int(width if source_width is None else source_width)
        source_depth = int(depth if source_depth is None else source_depth)
        latent_dim = int(width if latent_dim is None else latent_dim)

        self.backbone = DeepGreenBackbone(
            encoder_width=transform_width,
            encoder_depth=transform_depth,
            source_width=source_width,
            source_depth=source_depth,
            latent_dim=latent_dim,
            n_green_functions=n_green_functions,
            n_sources=n_sources,
            kernel_type=kernel_type,
            activation=activation,
            encoder_skip_connections=encoder_skip_connections,
            decoder_skip_connections=decoder_skip_connections,
            decoder_width=decoder_width,
            decoder_depth=decoder_depth,
            output_dim=int(shape_out),
            output_scale=output_scale,
            name="deepgreen",
        )

    def predict_fields(self, inputs, training: bool = False):

        #del training

        return self.backbone(inputs)
    
    @tf.function
    def call(self, inputs, training: bool = False, **_kwargs):

        return self.predict_fields(inputs, training=training)


def build_deepgreen_model(network: dict[str, object], shape_out: int = 3):

    parameterization = network.get("parameterization", {})
    backbone = network.get("backbone", {})
    encoder = network.get("Encoder", {})
    decoder = network.get("Decoder", {})
    kernel = network.get("KernelFunction", {})
    sources = network.get("SourceGenerator", {})

    normalized = str(parameterization.get("wind", "direct")).lower()

    if normalized != "direct":
        raise ValueError(f"Unsupported DeepGreen parameterization '{normalized}' in hyperMLT.")

    return DeepGreenDirect(
        shape_out=shape_out,
        width=int(encoder.get("width", 64)),
        depth=int(encoder.get("depth", 4)),
        nblocks=int(backbone.get("nblocks", 1)),
        output_scale=parameterization.get("output_scale", [10.0, 10.0, 1.0]),
        latent_dim=int(kernel.get("latent_dim", encoder.get("width", 64))),
        transform_width=int(encoder.get("width", 64)),
        transform_depth=int(encoder.get("depth", 4)),
        source_width=int(sources.get("width", 16)),
        source_depth=int(sources.get("depth", 3)),
        decoder_width=int(decoder.get("width", encoder.get("width", 64))),
        decoder_depth=int(decoder.get("depth", encoder.get("depth", 4))),
        n_green_functions=int(kernel.get("n_green_functions", 16)),
        n_sources=int(sources.get("n_sources", 32)),
        kernel_type=str(kernel.get("type", "matern52")),
        activation=str(backbone.get("activation", "sin")),
        encoder_skip_connections=bool(encoder.get("use_skip_connections", False)),
        decoder_skip_connections=bool(decoder.get("use_skip_connections", False)),
    )
