from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras


class SIRENLayerInitializer(keras.initializers.Initializer):
    def __init__(self, w0: float = 1.0):

        self.w0 = float(w0)

    def __call__(self, shape, dtype=None):

        in_features = int(shape[1])
        limit = tf.sqrt(2.0 / float(in_features)) / self.w0

        return tf.random.uniform(shape, -limit, limit, dtype=dtype)


class DropoutLayer(keras.layers.Layer):
    def __init__(self, n: int = 5000, **kwargs):

        super().__init__(**kwargs)
        self.max_n = int(n)

    def build(self, input_shape):

        in_dim = int(input_shape[-1])
        self.in_dim = in_dim

        self.step_count = tf.Variable(0.0, trainable=False, dtype=tf.float32, name=f"{self.name}_step")
        self.mask = tf.Variable(
            tf.ones((in_dim,), dtype=tf.float32),
            trainable=False,
            name=f"{self.name}_mask",
        )

    def update_mask(self, n: int, percent: float = 0.2):

        frac = tf.minimum(self.step_count / (0.9 * float(max(n, 1))), 1.0)
        wcut = percent + frac * (1.0 - percent)
        ramp = tf.linspace(1.0, 0.0, self.in_dim)
        shifted = 50.0 * (ramp + wcut - 1.0)
        mask = tf.clip_by_value(shifted, 0.0, 1.0)
        mask = mask / tf.reduce_mean(mask)

        self.mask.assign(mask)
        self.step_count.assign_add(1.0)

    def call(self, inputs, training: bool = False):

        del training

        return inputs * self.mask


class Embedding(keras.layers.Layer):
    def __init__(
        self,
        n_neurons: int,
        *,
        bias_initializer: str = "zeros",
        stddev: float | None = 1.0,
        w0: float = 1.0,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.w0 = float(w0)
        self.n_neurons = int(n_neurons)
        self.bias_initializer = bias_initializer
        self.stddev = stddev
        self.kernel_initializer = SIRENLayerInitializer(self.w0)

        self.layer0 = keras.layers.Dense(
            self.n_neurons,
            activation=None,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            use_bias=False,
            trainable=True,
        )

    def call(self, inputs, alpha: float = 0.0):

        sigma = tf.exp(alpha)
        projected = self.layer0(self.w0 * sigma * inputs)

        sinx = tf.sin(projected)
        cosx = tf.cos(projected)

        return tf.concat([sinx, cosx], axis=1)


class LaafLayer(keras.layers.Layer):
    def __init__(
        self,
        n_neurons: int,
        *,
        activation: str = "sine",
        w0: float = 1.0,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.n_neurons = int(n_neurons)
        self.w0 = float(w0)
        self.kernel_initializer = SIRENLayerInitializer(self.w0)

        normalized = str(activation).lower()

        if normalized == "sine":
            self.activation = tf.sin
        elif normalized == "tanh":
            self.activation = tf.tanh
        elif normalized == "swish":
            self.activation = keras.activations.swish
        elif normalized == "gelu":
            self.activation = keras.activations.gelu
        else:
            raise ValueError(f"Unsupported LAAF activation '{activation}'.")

    def build(self, input_shape):

        nfeatures = int(input_shape[1])

        self.w = self.add_weight(
            shape=(nfeatures, self.n_neurons),
            initializer=self.kernel_initializer,
            trainable=True,
            name="w_laaf",
        )
        self.b = self.add_weight(
            shape=(self.n_neurons,),
            initializer="zeros",
            trainable=True,
            name="b_laaf",
        )

    def call(self, inputs, alpha: float = 0.0):

        sigma = tf.exp(alpha)
        outputs = tf.matmul(inputs, sigma * self.w) + self.b

        return self.activation(outputs)


class Linear(keras.layers.Layer):
    def __init__(
        self,
        noutputs: int = 1,
        *,
        kernel_initializer: str | keras.initializers.Initializer = "glorot_uniform",
        add_bias: bool = False,
        activation: str | None = None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        normalized = None if activation is None else str(activation).lower()

        if normalized == "sine":
            self.activation = tf.sin
        elif normalized == "tanh":
            self.activation = tf.tanh
        elif normalized == "swish":
            self.activation = keras.activations.swish
        elif normalized == "relu":
            self.activation = keras.activations.relu
        else:
            self.activation = keras.activations.linear

        self.noutputs = int(noutputs)
        self.kernel_initializer = kernel_initializer
        self.add_bias = bool(add_bias)

    def build(self, input_shape):

        nnodes = int(input_shape[1])

        self.w = self.add_weight(
            shape=(nnodes, self.noutputs),
            initializer=self.kernel_initializer,
            trainable=True,
            name="w_linear",
        )

        if self.add_bias:
            self.b = self.add_weight(
                shape=(1, self.noutputs),
                initializer="zeros",
                trainable=True,
                name="b_linear",
            )
        else:
            self.b = 0.0

    def call(self, inputs, alpha: float = 1.0):

        outputs = tf.matmul(inputs, self.w) + self.b
        outputs = alpha * outputs

        return self.activation(outputs)


class Scaler(keras.layers.Layer):
    def __init__(
        self,
        values: list[float] | tuple[float, ...] = (1e1, 1e1, 1.0, 1.0),
        *,
        add_bias: bool = True,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.values = [float(value) for value in values]
        self.add_bias = bool(add_bias)

    def build(self, input_shape):

        n_in = int(input_shape[-1])

        self.w = self.add_weight(
            shape=(n_in,),
            initializer="ones",
            trainable=True,
            name="gain",
        )

        if self.add_bias:
            self.b = self.add_weight(
                shape=(n_in,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )
        else:
            self.b = 0.0

        self.scaling = tf.constant(np.asarray(self.values[:n_in], dtype=np.float32), dtype=tf.float32)

    def call(self, inputs):

        outputs = inputs * self.w + self.b

        return outputs * self.scaling
