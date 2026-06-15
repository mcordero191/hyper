from __future__ import annotations

import tensorflow as tf


def eq_continuity(
    u_x: tf.Tensor,
    v_y: tf.Tensor,
    w_z: tf.Tensor,
    *,
    w: tf.Tensor | float = 0.0,
    rho_ratio: tf.Tensor | float = 0.0,
) -> tf.Tensor:

    return rho_ratio * w + u_x + v_y + w_z


def eq_vorticity_rotational_forcing(
    omega_t: tf.Tensor,
    forcing_kj: tf.Tensor,
    forcing_jk: tf.Tensor,
    *,
    theta_j: tf.Tensor | float = 0.0,
    buoyancy_frequency: tf.Tensor | float = 0.0,
) -> tf.Tensor:

    return omega_t + forcing_kj - forcing_jk + buoyancy_frequency * theta_j
