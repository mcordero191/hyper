from __future__ import annotations

import tensorflow as tf

from hyperMLT.physics.equations import eq_continuity, eq_vorticity_rotational_forcing


FORMULATION_ALIASES = {
    "data_only": "data_only",
    "continuity_only": "continuity_only",
    "divergence_free": "continuity_only",
    "vp_div": "continuity_only",
    "vertical_vorticity_inviscid": "vertical_vorticity_inviscid",
    "vv_nonu": "vertical_vorticity_inviscid",
    "vv-nonu": "vertical_vorticity_inviscid",
    "vertical_vorticity_viscous": "vertical_vorticity_viscous",
    "vv": "vertical_vorticity_viscous",
}


FORMULATION_LABELS = {
    "data_only": "Data only",
    "continuity_only": "Continuity only",
    "vertical_vorticity_inviscid": "Vertical vorticity (inviscid)",
    "vertical_vorticity_viscous": "Vertical vorticity (viscous)",
}


def _format_available_formulations() -> str:

    implemented = ", ".join(sorted(FORMULATION_LABELS))
    aliases = ", ".join(sorted(FORMULATION_ALIASES))

    return (
        f"Implemented canonical formulations: {implemented}. "
        f"Accepted names and aliases: {aliases}."
    )


def normalize_formulation_name(name: str) -> str:

    normalized = str(name).strip().lower()

    if normalized not in FORMULATION_ALIASES:
        raise ValueError(
            f"Unsupported hyperMLT PDE formulation '{name}'. "
            f"{_format_available_formulations()}"
        )

    return FORMULATION_ALIASES[normalized]


def describe_formulation(name: str) -> str:

    canonical = normalize_formulation_name(name)

    return FORMULATION_LABELS[canonical]


def data_only_loss(
    model,
    t_norm: tf.Tensor,
    z_norm: tf.Tensor,
    x_norm: tf.Tensor,
    y_norm: tf.Tensor,
    normalized_to_physical_grad_scale: tf.Tensor,
    rho_ratio: tf.Tensor,
) -> dict[str, tf.Tensor]:

    del model
    del t_norm
    del z_norm
    del x_norm
    del y_norm
    del normalized_to_physical_grad_scale
    del rho_ratio

    zero = tf.zeros((), dtype=tf.float32)

    return {
        "div": zero,
        "mom": zero,
        "temp": zero,
    }


def vp_divergence_loss(
    model,
    t_norm: tf.Tensor,
    z_norm: tf.Tensor,
    x_norm: tf.Tensor,
    y_norm: tf.Tensor,
    normalized_to_physical_grad_scale: tf.Tensor,
    rho_ratio: tf.Tensor,
) -> tf.Tensor:

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        coords_spatial = tf.concat([z_norm, x_norm, y_norm], axis=1)
        tape.watch(coords_spatial)

        coords_norm = tf.concat([t_norm, coords_spatial], axis=1)

        outputs = model(coords_norm, training=True)

        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        w = outputs[:, 2:3]

    # `batch_jacobian` returns per-sample derivatives directly.
    # With spatial coordinates ordered as [z, x, y], shapes are:
    #   u_jac_norm: [batch, 1, 3]
    #   v_jac_norm: [batch, 1, 3]
    #   w_jac_norm: [batch, 1, 3]
    u_jac_norm = tape.batch_jacobian(u, coords_spatial, experimental_use_pfor=False)
    v_jac_norm = tape.batch_jacobian(v, coords_spatial, experimental_use_pfor=False)
    w_jac_norm = tape.batch_jacobian(w, coords_spatial, experimental_use_pfor=False)

    del tape

    u_x = u_jac_norm[:, 0, 1:2] * normalized_to_physical_grad_scale[2]
    v_y = v_jac_norm[:, 0, 2:3] * normalized_to_physical_grad_scale[3]
    w_z = w_jac_norm[:, 0, 0:1] * normalized_to_physical_grad_scale[1]

    div = eq_continuity(u_x, v_y, w_z, w=w, rho_ratio=rho_ratio)

    return {
        "div": tf.reduce_mean(tf.square(div)),
        "mom": tf.zeros((), dtype=div.dtype),
        "temp": tf.zeros((), dtype=div.dtype),
    }


def vertical_vorticity_inviscid_loss(
    model,
    t_norm: tf.Tensor,
    z_norm: tf.Tensor,
    x_norm: tf.Tensor,
    y_norm: tf.Tensor,
    normalized_to_physical_grad_scale: tf.Tensor,
    rho_ratio: tf.Tensor,
) -> tf.Tensor:

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as outer_tape:
        outer_tape.watch(t_norm)
        outer_tape.watch(x_norm)
        outer_tape.watch(y_norm)

        coords_spatial = tf.concat([z_norm, x_norm, y_norm], axis=1)

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as inner_tape:
            inner_tape.watch(coords_spatial)

            coords_norm = tf.concat([t_norm, coords_spatial], axis=1)
            outputs = model(coords_norm, training=True)

            u = outputs[:, 0:1]
            v = outputs[:, 1:2]
            w = outputs[:, 2:3]

        spatial_jac_norm = inner_tape.batch_jacobian(
            outputs,
            coords_spatial,
            experimental_use_pfor=False,
        )

        del inner_tape

        dz_scale = normalized_to_physical_grad_scale[1]
        dx_scale = normalized_to_physical_grad_scale[2]
        dy_scale = normalized_to_physical_grad_scale[3]

        u_z = spatial_jac_norm[:, 0, 0:1] * dz_scale
        u_x = spatial_jac_norm[:, 0, 1:2] * dx_scale
        u_y = spatial_jac_norm[:, 0, 2:3] * dy_scale

        v_z = spatial_jac_norm[:, 1, 0:1] * dz_scale
        v_x = spatial_jac_norm[:, 1, 1:2] * dx_scale
        v_y = spatial_jac_norm[:, 1, 2:3] * dy_scale

        w_z = spatial_jac_norm[:, 2, 0:1] * dz_scale
        w_x = spatial_jac_norm[:, 2, 1:2] * dx_scale
        w_y = spatial_jac_norm[:, 2, 2:3] * dy_scale

        omega_x = w_y - v_z
        omega_y = u_z - w_x
        omega_z = v_x - u_y

        force_x = w * omega_y - v * omega_z
        force_y = u * omega_z - w * omega_x

    omega_z_t_norm = outer_tape.batch_jacobian(omega_z, t_norm, experimental_use_pfor=False)
    force_x_y_norm = outer_tape.batch_jacobian(force_x, y_norm, experimental_use_pfor=False)
    force_y_x_norm = outer_tape.batch_jacobian(force_y, x_norm, experimental_use_pfor=False)

    del outer_tape

    omega_z_t = omega_z_t_norm[:, 0, 0:1] * normalized_to_physical_grad_scale[0]
    force_x_y = force_x_y_norm[:, 0, 0:1] * normalized_to_physical_grad_scale[3]
    force_y_x = force_y_x_norm[:, 0, 0:1] * normalized_to_physical_grad_scale[2]

    div = eq_continuity(u_x, v_y, w_z, w=w, rho_ratio=rho_ratio)
    momentum_z = eq_vorticity_rotational_forcing(omega_z_t, force_y_x, force_x_y)

    return {
        "div": tf.reduce_mean(tf.square(div)),
        "mom": tf.reduce_mean(tf.square(momentum_z)),
        "temp": tf.zeros((), dtype=div.dtype),
    }


def compute_pde_loss(
    formulation: str,
    model,
    *,
    t_norm: tf.Tensor,
    z_norm: tf.Tensor,
    x_norm: tf.Tensor,
    y_norm: tf.Tensor,
    normalized_to_physical_grad_scale: tf.Tensor,
    rho_ratio: tf.Tensor,
) -> dict[str, tf.Tensor]:

    canonical = normalize_formulation_name(formulation)

    return get_pde_loss_function(canonical)(
        model,
        t_norm=t_norm,
        z_norm=z_norm,
        x_norm=x_norm,
        y_norm=y_norm,
        normalized_to_physical_grad_scale=normalized_to_physical_grad_scale,
        rho_ratio=rho_ratio,
    )


def get_pde_loss_function(formulation: str):

    canonical = normalize_formulation_name(formulation)

    if canonical == "data_only":
        return data_only_loss

    if canonical == "continuity_only":
        return vp_divergence_loss

    if canonical == "vertical_vorticity_inviscid":
        return vertical_vorticity_inviscid_loss

    raise NotImplementedError(
        f"hyperMLT formulation '{canonical}' is recognized but not implemented yet."
    )
