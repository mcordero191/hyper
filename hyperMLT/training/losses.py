from __future__ import annotations

import tensorflow as tf


def doppler_from_field(field_output: tf.Tensor, braggs: tf.Tensor) -> tf.Tensor:

    radial_angular_frequency = tf.reduce_sum(field_output[:, :3] * braggs, axis=1, keepdims=True)

    return -radial_angular_frequency / tf.constant(2.0 * 3.141592653589793, dtype=field_output.dtype)


def _weighted_reduce(residual: tf.Tensor, weights: tf.Tensor | None = None) -> tf.Tensor:

    if weights is None:
        return tf.reduce_mean(residual)

    weights = tf.cast(weights, residual.dtype)

    return tf.reduce_sum(weights * residual) / tf.maximum(
        tf.reduce_sum(weights),
        tf.constant(1.0, dtype=residual.dtype),
    )


def weighted_mse(observed: tf.Tensor, predicted: tf.Tensor, weights: tf.Tensor | None = None) -> tf.Tensor:

    residual = tf.square(predicted - observed)

    return _weighted_reduce(residual, weights=weights)


def weighted_mae(observed: tf.Tensor, predicted: tf.Tensor, weights: tf.Tensor | None = None) -> tf.Tensor:

    residual = tf.abs(predicted - observed)

    return _weighted_reduce(residual, weights=weights)


def get_data_loss_function(name: str):

    normalized = str(name).strip().lower()

    if normalized == "mae":
        return weighted_mae

    if normalized == "mse":
        return weighted_mse

    raise ValueError(f"Unsupported hyperMLT data loss '{name}'.")
