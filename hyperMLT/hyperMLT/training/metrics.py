from __future__ import annotations

import tensorflow as tf


def doppler_rmse(observed: tf.Tensor, predicted: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.reduce_mean(tf.square(predicted - observed)))
