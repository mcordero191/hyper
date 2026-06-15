from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from hyperMLT.artifacts.io import write_json
from hyperMLT.datasets.mean_winds import mean_wind_from_fields
from hyperMLT.models import build_model
from hyperMLT.plotting.mean_winds import plot_mean_winds
from hyperMLT.physics.formulations import get_pde_loss_function, normalize_formulation_name
from hyperMLT.physics.sampling import sample_pde_points
from hyperMLT.utils.console import format_progress_report, format_summary_table

from .losses import doppler_from_field, get_data_loss_function
from .metrics import doppler_rmse
from .scheduling import current_loss_weights


@dataclass
class TrainingArtifacts:
    run_dir: Path
    history_path: Path
    weights_path: Path
    manifest_path: Path
    summary_plot_path: Path
    doppler_comparison_plot_path: Path | None
    mean_wind_plot_path: Path | None
    mean_wind_data_path: Path | None


@dataclass
class MeasurementBatch:
    coords_raw: tf.Tensor
    coords_norm: tf.Tensor
    dops: tf.Tensor
    weights: tf.Tensor | None
    braggs: tf.Tensor


@dataclass
class PDEBatch:
    coords_raw: tf.Tensor
    coords_norm: tf.Tensor
    t_norm: tf.Tensor
    z_norm: tf.Tensor
    x_norm: tf.Tensor
    y_norm: tf.Tensor
    rho_ratio: tf.Tensor


@dataclass
class PreparedTrainingData:
    train: MeasurementBatch
    validation: MeasurementBatch | None
    validation_inner: MeasurementBatch | None
    validation_outer: MeasurementBatch | None
    pde: PDEBatch | None
    lower: tf.Tensor
    upper: tf.Tensor
    normalized_to_physical_grad_scale: tf.Tensor
    time_base: float


def _count_variables(variables) -> int:

    return int(sum(np.prod(variable.shape) for variable in variables))


def _build_training_setup_summary(
    config,
    model,
    prepared: PreparedTrainingData,
    *,
    model_family: str,
    formulation_label: str,
    formulation_name: str,
    total_epochs: int,
) -> str:

    pde_samples = 0 if prepared.pde is None else int(prepared.pde.coords_norm.shape[0])
    validation_samples = 0 if prepared.validation is None else int(prepared.validation.coords_norm.shape[0])
    validation_inner_samples = 0 if prepared.validation_inner is None else int(prepared.validation_inner.coords_norm.shape[0])
    validation_outer_samples = 0 if prepared.validation_outer is None else int(prepared.validation_outer.coords_norm.shape[0])

    rows = [
        ("Architecture", model_family),
        ("PDE formulation", f"{formulation_label} ({formulation_name})"),
        ("Data loss", str(config.training.data_loss).upper()),
        ("Optimizer", str(config.training.optimizer)),
        ("Learning rate", f"{float(config.training.learning_rate):.1e}"),
        ("Epochs", total_epochs),
        ("Random seed", int(config.training.random_seed)),
        ("Train meteors", int(prepared.train.coords_norm.shape[0])),
        ("Validation meteors", validation_samples),
        ("Validation inner", validation_inner_samples),
        ("Validation outer", validation_outer_samples),
        ("PDE samples", pde_samples),
        ("Scheduler", str(config.training.scheduling.get("type", "fixed"))),
        ("Trainable params", _count_variables(model.trainable_variables)),
        ("Non-trainable params", _count_variables(model.non_trainable_variables)),
        ("Total params", model.count_params()),
    ]

    return format_summary_table("Training Setup Summary", rows)


def _build_timing_summary(title: str, timings: dict[str, float]) -> str:

    rows = [
        (label.replace("_s", "").replace("_", " "), f"{float(value):.2f}s")
        for label, value in timings.items()
    ]

    return format_summary_table(title, rows)


def _build_feature_bounds(domain: dict[str, Any]) -> tuple[tf.Tensor, tf.Tensor]:

    norm = domain["normalization"]

    lower = tf.constant(norm["lower_bounds"], dtype=tf.float32)
    upper = tf.constant(norm["upper_bounds"], dtype=tf.float32)

    return lower, upper


def _build_normalized_to_physical_grad_scale(lower: tf.Tensor, upper: tf.Tensor) -> tf.Tensor:

    return 2.0 / (upper - lower)


def _normalize_coords(coords: tf.Tensor, lower: tf.Tensor, upper: tf.Tensor) -> tf.Tensor:

    return 2.0 * (coords - lower) / (upper - lower) - 1.0


def _window_time_base(window) -> float:

    central_date = window.metadata.get("central_date")

    if central_date:
        return pd.Timestamp(central_date).timestamp()

    return float(window.training_df["times"].min())


def _schedule_requires_pde(config) -> bool:

    residual_weights = dict(config.physics.residual_weights or {})

    for term in ("div", "mom", "temp"):
        if float(residual_weights.get(term, 0.0)) > 0.0:
            return True

    scheduling = dict(config.training.scheduling or {})
    stages = list(scheduling.get("stages", []))

    for stage in stages:
        weights = dict(stage.get("weights", {}))

        for term in ("div", "mom", "temp"):
            if float(weights.get(term, 0.0)) > 0.0:
                return True

    return False


def _weights_from_df(df: pd.DataFrame) -> np.ndarray | None:

    if "weights" in df.columns:
        return df["weights"].to_numpy(dtype=np.float32)

    return None


def _measurement_batch_from_df(
    df: pd.DataFrame,
    *,
    time_base: float,
    lower: tf.Tensor,
    upper: tf.Tensor,
) -> MeasurementBatch:

    t = (df["times"].to_numpy(dtype=np.float64) - float(time_base)).astype(np.float32)
    z = df["z"].to_numpy(dtype=np.float32)
    x = df["x"].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)

    coords_raw_np = np.stack([t, z, x, y], axis=1).astype(np.float32)
    coords_raw = tf.constant(coords_raw_np, dtype=tf.float32)
    coords_norm = _normalize_coords(coords_raw, lower, upper)

    dops = tf.constant(df["dops"].to_numpy(dtype=np.float32).reshape(-1, 1), dtype=tf.float32)
    weights_np = _weights_from_df(df)
    weights = None

    if weights_np is not None:
        weights = tf.constant(weights_np.reshape(-1, 1), dtype=tf.float32)

    braggs = tf.constant(
        df[["braggs_x", "braggs_y", "braggs_z"]].to_numpy(dtype=np.float32),
        dtype=tf.float32,
    )

    return MeasurementBatch(
        coords_raw=coords_raw,
        coords_norm=coords_norm,
        dops=dops,
        weights=weights,
        braggs=braggs,
    )


def _pde_batch_from_window(
    config,
    window,
    *,
    time_base: float,
    lower: tf.Tensor,
    upper: tf.Tensor,
) -> PDEBatch | None:

    if not _schedule_requires_pde(config):
        return None

    sampling = dict(config.physics.pde_sampling or {})
    sample_count = int(sampling.get("sample_count", 0) or 0)

    if sample_count <= 0:
        return None

    coords_raw_np = sample_pde_points(
        window.training_df,
        lower,
        upper,
        sample_count=sample_count,
        method=str(sampling.get("method", "random")),
        time_base=time_base,
        random_seed=int(config.training.random_seed),
    )

    coords_raw = tf.constant(coords_raw_np, dtype=tf.float32)
    coords_norm = _normalize_coords(coords_raw, lower, upper)
    rho_ratio = tf.zeros((sample_count, 1), dtype=tf.float32)

    return PDEBatch(
        coords_raw=coords_raw,
        coords_norm=coords_norm,
        t_norm=coords_norm[:, 0:1],
        z_norm=coords_norm[:, 1:2],
        x_norm=coords_norm[:, 2:3],
        y_norm=coords_norm[:, 3:4],
        rho_ratio=rho_ratio,
    )


def prepare_training_data(config, window) -> PreparedTrainingData:

    lower, upper = _build_feature_bounds(config.to_dict()["domain"])
    normalized_to_physical_grad_scale = _build_normalized_to_physical_grad_scale(lower, upper)
    time_base = _window_time_base(window)

    train_batch = _measurement_batch_from_df(
        window.training_df,
        time_base=time_base,
        lower=lower,
        upper=upper,
    )

    validation_batch = None

    if window.validation_df is not None and len(window.validation_df) > 0:
        validation_batch = _measurement_batch_from_df(
            window.validation_df,
            time_base=time_base,
            lower=lower,
            upper=upper,
        )

    validation_inner_batch = None

    if getattr(window, "validation_inner_df", None) is not None and len(window.validation_inner_df) > 0:
        validation_inner_batch = _measurement_batch_from_df(
            window.validation_inner_df,
            time_base=time_base,
            lower=lower,
            upper=upper,
        )

    validation_outer_batch = None

    if getattr(window, "validation_outer_df", None) is not None and len(window.validation_outer_df) > 0:
        validation_outer_batch = _measurement_batch_from_df(
            window.validation_outer_df,
            time_base=time_base,
            lower=lower,
            upper=upper,
        )

    pde_batch = _pde_batch_from_window(
        config,
        window,
        time_base=time_base,
        lower=lower,
        upper=upper,
    )

    return PreparedTrainingData(
        train=train_batch,
        validation=validation_batch,
        validation_inner=validation_inner_batch,
        validation_outer=validation_outer_batch,
        pde=pde_batch,
        lower=lower,
        upper=upper,
        normalized_to_physical_grad_scale=normalized_to_physical_grad_scale,
        time_base=time_base,
    )


def _plot_training_summary(history: dict[str, list[float]], path: Path) -> None:

    epochs = np.asarray(history["epoch"], dtype=np.int32)

    def _bounded_log_limits(series_list: list[np.ndarray]) -> tuple[float, float]:

        positive = []

        for series in series_list:
            values = np.asarray(series, dtype=np.float64)
            values = values[np.isfinite(values) & (values > 0.0)]

            if values.size > 0:
                positive.append(values)

        if not positive:
            return 1e-4, 1.0

        merged = np.concatenate(positive)
        ymax = float(np.nanmax(merged))
        ymin = float(np.nanmin(merged))

        if ymax <= 0.0:
            return 1e-4, 1.0

        if ymin == ymax:
            return max(ymin / 10.0, 1e-12), ymax * 10.0

        ymin = max(ymin, ymax / 1e4)

        return ymin, ymax

    def _safe_positive(values: list[float] | np.ndarray) -> np.ndarray:

        array = np.asarray(values, dtype=np.float64).copy()
        array[~np.isfinite(array)] = np.nan
        array[array <= 0.0] = np.nan

        return array

    total_loss = np.asarray(history["train_total_loss"], dtype=np.float64)
    data_loss = np.asarray(history["train_data_loss"], dtype=np.float64)
    div_loss = np.asarray(history["train_div_loss"], dtype=np.float64)
    mom_loss = np.asarray(history["train_mom_loss"], dtype=np.float64)
    temp_loss = np.asarray(history["train_temp_loss"], dtype=np.float64)
    val_rmse = np.asarray(history["val_doppler_rmse"], dtype=np.float64)
    val_inner_rmse = np.asarray(history["val_inner_doppler_rmse"], dtype=np.float64)
    val_outer_rmse = np.asarray(history["val_outer_doppler_rmse"], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    loss_ax = axes[0]
    pde_ax = loss_ax.twinx()

    dense_left_series = [
        (_safe_positive(total_loss), "Total", "black"),
        (_safe_positive(data_loss), "Data", "tab:red"),
    ]
    sparse_left_series = [
        (_safe_positive(val_rmse), "Val RMSE", "tab:purple"),
        (_safe_positive(val_inner_rmse), "Val Inner", "tab:green"),
        (_safe_positive(val_outer_rmse), "Val Outer", "tab:pink"),
    ]
    right_series = [
        (_safe_positive(div_loss), "DIV", "tab:blue"),
        (_safe_positive(mom_loss), "Momt", "tab:orange"),
        (_safe_positive(temp_loss), "Temp", "tab:brown"),
    ]

    left_handles = []

    for values, label, color in dense_left_series:
        (line,) = loss_ax.plot(epochs, values, label=label, color=color)
        left_handles.append(line)

    for values, label, color in sparse_left_series:
        valid = np.isfinite(values)
        (line,) = loss_ax.plot(
            epochs[valid],
            values[valid],
            label=label,
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.5,
        )
        left_handles.append(line)

    right_handles = []

    for values, label, color in right_series:
        (line,) = pde_ax.plot(epochs, values, label=label, color=color, linestyle="--")
        right_handles.append(line)

    left_ymin, left_ymax = _bounded_log_limits(
        [series for series, _, _ in dense_left_series + sparse_left_series]
    )
    right_ymin, right_ymax = _bounded_log_limits([series for series, _, _ in right_series])

    loss_ax.set_title("Losses")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Data / Validation")
    loss_ax.set_yscale("log")
    loss_ax.set_ylim(left_ymin, left_ymax)
    loss_ax.grid(True, alpha=0.3)

    pde_ax.set_ylabel("PDE")
    pde_ax.set_yscale("log")
    pde_ax.set_ylim(right_ymin, right_ymax)

    loss_ax.legend(left_handles + right_handles, [line.get_label() for line in left_handles + right_handles], loc="best")

    weight_data = np.asarray(history["w_data"], dtype=np.float64)
    weight_div = np.asarray(history["w_div"], dtype=np.float64)
    weight_mom = np.asarray(history["w_mom"], dtype=np.float64)
    weight_temp = np.asarray(history["w_temp"], dtype=np.float64)

    def _plot_weight(values: np.ndarray, *, label: str, color: str) -> None:

        safe = values.astype(np.float64, copy=True)
        safe[safe <= 0.0] = np.nan
        axes[1].plot(epochs, safe, label=label, color=color)

    _plot_weight(weight_data, label="w_data", color="tab:red")
    _plot_weight(weight_div, label="w_div", color="tab:blue")
    _plot_weight(weight_mom, label="w_mom", color="tab:orange")
    _plot_weight(weight_temp, label="w_temp", color="tab:brown")

    weight_ymin, weight_ymax = _bounded_log_limits([weight_data, weight_div, weight_mom, weight_temp])

    axes[1].set_title("Weights")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weight")
    axes[1].set_yscale("log")
    axes[1].set_ylim(weight_ymin, weight_ymax)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_mean_wind_outputs(df_winds: dict[str, Any], plot_path: Path, data_path: Path) -> None:

    plot_mean_winds(
        df_winds["times"],
        df_winds["alts"],
        df_winds["u0"],
        df_winds["v0"],
        df_winds["w0"],
        vmins=[-100, -100, -10],
        vmaxs=[100, 100, 10],
        figfile=str(plot_path),
        histogram=True,
    )

    with h5py.File(data_path, "w") as fp:
        for key, value in df_winds.items():
            fp[key] = value


def _plot_doppler_comparison(
    train_observed: np.ndarray,
    train_predicted: np.ndarray,
    path: Path,
    *,
    validation_observed: np.ndarray | None = None,
    validation_predicted: np.ndarray | None = None,
) -> None:

    panels = [("Train", train_observed, train_predicted)]

    if validation_observed is not None and validation_predicted is not None:
        panels.append(("Validation", validation_observed, validation_predicted))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    axes = axes[0]

    all_values = [train_observed, train_predicted]

    if validation_observed is not None:
        all_values.append(validation_observed)

    if validation_predicted is not None:
        all_values.append(validation_predicted)

    combined = np.concatenate([np.asarray(values, dtype=np.float64).ravel() for values in all_values])
    combined = combined[np.isfinite(combined)]

    if combined.size == 0:
        plt.close(fig)
        return

    lower = float(np.min(combined))
    upper = float(np.max(combined))

    if lower == upper:
        delta = max(abs(lower) * 0.05, 1.0)
        lower -= delta
        upper += delta

    for axis, (title, observed, predicted) in zip(axes, panels):
        observed = np.asarray(observed, dtype=np.float64).ravel()
        predicted = np.asarray(predicted, dtype=np.float64).ravel()
        valid = np.isfinite(observed) & np.isfinite(predicted)

        if np.count_nonzero(valid) > 0:
            axis.hist2d(
                observed[valid],
                predicted[valid],
                bins=60,
                range=[[lower, upper], [lower, upper]],
                cmap="inferno",
                norm="log",
            )

        axis.plot([lower, upper], [lower, upper], color="cyan", linestyle="--", linewidth=1.0)
        axis.set_title(f"{title} Doppler")
        axis.set_xlabel("Measured (Hz)")
        axis.set_ylabel("Predicted (Hz)")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _predicted_mean_wind(
    model,
    df_reference: pd.DataFrame,
    *,
    time_base: float,
    lower: tf.Tensor,
    upper: tf.Tensor,
):

    reference_batch = _measurement_batch_from_df(
        df_reference,
        time_base=time_base,
        lower=lower,
        upper=upper,
    )

    outputs = model(reference_batch.coords_norm, training=False).numpy()

    df_model = df_reference.copy()
    df_model["u_pred"] = outputs[:, 0]
    df_model["v_pred"] = outputs[:, 1]
    df_model["w_pred"] = outputs[:, 2]

    return mean_wind_from_fields(df_model)


def _build_optimizer(training_config):

    learning_rate = float(training_config.learning_rate)
    optimizer_name = str(training_config.optimizer).lower()

    if optimizer_name != "adam":
        raise ValueError(f"Unsupported hyperMLT optimizer '{training_config.optimizer}'.")

    legacy_optimizers = getattr(tf.keras.optimizers, "legacy", None)

    if legacy_optimizers is not None and hasattr(legacy_optimizers, "Adam"):
        return legacy_optimizers.Adam(learning_rate=learning_rate)

    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def _global_gradient_norm(gradients) -> tf.Tensor:

    finite_grads = [gradient for gradient in gradients if gradient is not None]

    if not finite_grads:
        return tf.zeros((), dtype=tf.float32)

    return tf.linalg.global_norm(finite_grads)


def _shared_trunk_drift(
    trunk_variables,
    trunk_reference_weights: list[np.ndarray] | None,
) -> tuple[float | None, float | None]:

    if trunk_reference_weights is None or not trunk_variables:
        return None, None

    if len(trunk_reference_weights) != len(trunk_variables):
        return None, None

    sq_sum = 0.0

    for variable, initial in zip(trunk_variables, trunk_reference_weights):
        current = variable.numpy()
        delta = np.asarray(current) - np.asarray(initial)
        sq_sum += float(np.sum(np.square(delta)))

    abs_drift = float(np.sqrt(sq_sum))

    init_sq_sum = 0.0
    for initial in trunk_reference_weights:
        init_sq_sum += float(np.sum(np.square(np.asarray(initial))))

    rel_drift = abs_drift / max(float(np.sqrt(init_sq_sum)), 1e-12)

    return abs_drift, rel_drift


def train_model(
    config,
    window,
    run_dir: str | Path,
    *,
    model_family: str | None = None,
    formulation_label: str | None = None,
    epochs_override: int | None = None,
    model=None,
    trunk_anchor_reference_weights: list[np.ndarray] | None = None,
    trunk_anchor_weight: float = 0.0,
    freeze_trunk_until_epoch: int = 0,
) -> TrainingArtifacts:

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    startup_timings: dict[str, float] = {}

    prepare_start = perf_counter()
    prepared = prepare_training_data(config, window)
    startup_timings["prepare_data_s"] = perf_counter() - prepare_start

    tf.keras.utils.set_random_seed(int(config.training.random_seed))

    model_start = perf_counter()
    if model is None:
        model = build_model(config.model, shape_out=3)
        _ = model(prepared.train.coords_norm[:2], training=False)
    else:
        _ = model(prepared.train.coords_norm[:2], training=False)
    startup_timings["build_model_s"] = perf_counter() - model_start

    full_optimizer_variables = list(model.trainable_variables)

    optimizer_start = perf_counter()
    optimizer = _build_optimizer(config.training)
    startup_timings["optimizer_setup_s"] = perf_counter() - optimizer_start
    total_epochs = int(epochs_override or config.training.epochs)
    formulation = normalize_formulation_name(config.physics.formulation)
    pde_loss_function = get_pde_loss_function(formulation)
    data_loss_function = get_data_loss_function(config.training.data_loss)
    enable_gradient_diagnostics = bool(getattr(config.training, "gradient_diagnostics", True))
    formulation_label = formulation_label or formulation
    startup_timings["setup_total_s"] = float(np.sum(list(startup_timings.values())))

    trunk_anchor_reference_tensors: tuple[tf.Tensor, ...] = ()
    trunk_trainable_variables = []
    trunk_layer = None

    if hasattr(model, "backbone") and hasattr(model.backbone, "trunk"):
        trunk_layer = model.backbone.trunk
        trunk_trainable_variables = list(model.backbone.trunk.trainable_variables)

    if trunk_anchor_reference_weights is not None:
        if len(trunk_anchor_reference_weights) != len(trunk_trainable_variables):
            raise ValueError(
                "trunk_anchor_reference_weights must match model.backbone.trunk.trainable_variables length."
            )

        trunk_anchor_reference_tensors = tuple(
            tf.convert_to_tensor(weight, dtype=variable.dtype)
            for weight, variable in zip(trunk_anchor_reference_weights, trunk_trainable_variables)
        )

    trunk_anchor_weight_value = float(trunk_anchor_weight)
    trunk_anchor_weight_tensor = tf.constant(trunk_anchor_weight_value, dtype=tf.float32)
    trunk_anchor_enabled = bool(trunk_anchor_reference_tensors) and bool(trunk_trainable_variables) and trunk_anchor_weight_value > 0.0
    freeze_trunk_until_epoch = int(max(freeze_trunk_until_epoch, 0))
    trunk_is_frozen = bool(trunk_layer is not None and freeze_trunk_until_epoch > 0 and trunk_anchor_reference_tensors)

    if trunk_is_frozen:
        trunk_layer.trainable = False

    def _shared_trunk_anchor_loss() -> tf.Tensor:

        if not trunk_anchor_enabled:
            return tf.zeros((), dtype=tf.float32)

        penalties = [
            tf.reduce_mean(tf.square(variable - reference))
            for variable, reference in zip(trunk_trainable_variables, trunk_anchor_reference_tensors)
        ]

        return tf.add_n(penalties) / tf.cast(len(penalties), tf.float32)

    print("")
    print(
        _build_training_setup_summary(
            config,
            model,
            prepared,
            model_family=model_family or str(config.model.architecture),
            formulation_label=formulation_label,
            formulation_name=formulation,
            total_epochs=total_epochs,
        )
    )
    print("")
    print(_build_timing_summary("Training Startup Timing", startup_timings))

    def _should_report_epoch(epoch: int) -> bool:

        return epoch == 0 or (epoch + 1) % 200 == 0 or epoch == total_epochs - 1

    def _compute_loss_terms(
        train_coords_norm,
        train_braggs,
        train_dops,
        train_weights,
        pde_t_norm,
        pde_z_norm,
        pde_x_norm,
        pde_y_norm,
        pde_rho_ratio,
        normalized_to_physical_grad_scale,
    ):

        outputs = model(train_coords_norm, training=True)
        predicted_dops = doppler_from_field(outputs, train_braggs)

        loss_data = data_loss_function(train_dops, predicted_dops, weights=train_weights)

        if pde_t_norm is None:
            pde_terms = {
                "div": tf.zeros((), dtype=tf.float32),
                "mom": tf.zeros((), dtype=tf.float32),
                "temp": tf.zeros((), dtype=tf.float32),
            }
        else:
            pde_terms = pde_loss_function(
                model,
                t_norm=pde_t_norm,
                z_norm=pde_z_norm,
                x_norm=pde_x_norm,
                y_norm=pde_y_norm,
                normalized_to_physical_grad_scale=normalized_to_physical_grad_scale,
                rho_ratio=pde_rho_ratio,
            )

        return loss_data, pde_terms["div"], pde_terms["mom"], pde_terms["temp"]

    def _make_train_step():

        @tf.function
        def train_step(
            train_coords_norm,
            train_braggs,
            train_dops,
            train_weights,
            pde_t_norm,
            pde_z_norm,
            pde_x_norm,
            pde_y_norm,
            pde_rho_ratio,
            normalized_to_physical_grad_scale,
            w_data,
            w_div,
            w_mom,
            w_temp,
        ):

            with tf.GradientTape() as tape:
                loss_data, loss_div, loss_mom, loss_temp = _compute_loss_terms(
                    train_coords_norm,
                    train_braggs,
                    train_dops,
                    train_weights,
                    pde_t_norm,
                    pde_z_norm,
                    pde_x_norm,
                    pde_y_norm,
                    pde_rho_ratio,
                    normalized_to_physical_grad_scale,
                )

                total_loss = (
                    w_data * loss_data
                    + w_div * loss_div
                    + w_mom * loss_mom
                    + w_temp * loss_temp
                )
                total_loss = total_loss + trunk_anchor_weight_tensor * _shared_trunk_anchor_loss()

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            return total_loss, loss_data, loss_div, loss_mom, loss_temp

        return train_step

    @tf.function(reduce_retracing=True)
    def eval_step(coords_norm, braggs, dops):

        outputs = model(coords_norm, training=False)
        predicted_dops = doppler_from_field(outputs, braggs)

        return doppler_rmse(dops, predicted_dops)

    def _make_gradient_diagnostics_step():

        @tf.function
        def gradient_diagnostics_step(
            train_coords_norm,
            train_braggs,
            train_dops,
            train_weights,
            pde_t_norm,
            pde_z_norm,
            pde_x_norm,
            pde_y_norm,
            pde_rho_ratio,
            normalized_to_physical_grad_scale,
            w_data,
            w_div,
            w_mom,
            w_temp,
        ):

            with tf.GradientTape(persistent=True) as tape:
                loss_data, loss_div, loss_mom, loss_temp = _compute_loss_terms(
                    train_coords_norm,
                    train_braggs,
                    train_dops,
                    train_weights,
                    pde_t_norm,
                    pde_z_norm,
                    pde_x_norm,
                    pde_y_norm,
                    pde_rho_ratio,
                    normalized_to_physical_grad_scale,
                )

                weighted_data = w_data * loss_data
                weighted_div = w_div * loss_div
                weighted_mom = w_mom * loss_mom
                weighted_temp = w_temp * loss_temp

            grad_data = tape.gradient(weighted_data, model.trainable_variables)
            grad_div = tape.gradient(weighted_div, model.trainable_variables)
            grad_mom = tape.gradient(weighted_mom, model.trainable_variables)
            grad_temp = tape.gradient(weighted_temp, model.trainable_variables)
            del tape

            return (
                _global_gradient_norm(grad_data),
                _global_gradient_norm(grad_div),
                _global_gradient_norm(grad_mom),
                _global_gradient_norm(grad_temp),
            )

        return gradient_diagnostics_step

    if hasattr(optimizer, "build"):
        optimizer.build(full_optimizer_variables)

    train_step = _make_train_step()
    gradient_diagnostics_step = _make_gradient_diagnostics_step() if enable_gradient_diagnostics else None

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_total_loss": [],
        "train_data_loss": [],
        "train_div_loss": [],
        "train_mom_loss": [],
        "train_temp_loss": [],
        "val_doppler_rmse": [],
        "val_inner_doppler_rmse": [],
        "val_outer_doppler_rmse": [],
        "w_data": [],
        "w_div": [],
        "w_mom": [],
        "w_temp": [],
        "grad_data_norm": [],
        "grad_div_norm": [],
        "grad_mom_norm": [],
        "grad_temp_norm": [],
        "trunk_drift": [],
        "trunk_relative_drift": [],
    }

    train_start = perf_counter()
    weights_path = run_dir / "model.weights.h5"
    best_epoch: int | None = None
    best_val_rmse = float("inf")
    has_saved_best = False

    for epoch in range(total_epochs):
        if trunk_is_frozen and epoch >= freeze_trunk_until_epoch:
            trunk_layer.trainable = True
            trunk_is_frozen = False
            train_step = _make_train_step()
            if enable_gradient_diagnostics:
                gradient_diagnostics_step = _make_gradient_diagnostics_step()

        if hasattr(model, "advance_curriculum"):
            model.advance_curriculum(total_epochs)

        weights_cfg = current_loss_weights(config, epoch, total_epochs)

        pde_t_norm = None if prepared.pde is None else prepared.pde.t_norm
        pde_z_norm = None if prepared.pde is None else prepared.pde.z_norm
        pde_x_norm = None if prepared.pde is None else prepared.pde.x_norm
        pde_y_norm = None if prepared.pde is None else prepared.pde.y_norm
        pde_rho_ratio = None if prepared.pde is None else prepared.pde.rho_ratio

        total_loss, loss_data, loss_div, loss_mom, loss_temp = train_step(
            prepared.train.coords_norm,
            prepared.train.braggs,
            prepared.train.dops,
            prepared.train.weights,
            pde_t_norm,
            pde_z_norm,
            pde_x_norm,
            pde_y_norm,
            pde_rho_ratio,
            prepared.normalized_to_physical_grad_scale,
            tf.constant(float(weights_cfg["data"]), dtype=tf.float32),
            tf.constant(float(weights_cfg["div"]), dtype=tf.float32),
            tf.constant(float(weights_cfg["mom"]), dtype=tf.float32),
            tf.constant(float(weights_cfg["temp"]), dtype=tf.float32),
        )

        should_report = _should_report_epoch(epoch)

        if prepared.validation is not None and should_report:
            val_rmse = eval_step(
                prepared.validation.coords_norm,
                prepared.validation.braggs,
                prepared.validation.dops,
            )
            val_rmse_value = float(val_rmse.numpy())
        else:
            val_rmse_value = float("nan")

        if prepared.validation_inner is not None and should_report:
            val_inner_rmse = eval_step(
                prepared.validation_inner.coords_norm,
                prepared.validation_inner.braggs,
                prepared.validation_inner.dops,
            )
            val_inner_rmse_value = float(val_inner_rmse.numpy())
        else:
            val_inner_rmse_value = float("nan")

        if prepared.validation_outer is not None and should_report:
            val_outer_rmse = eval_step(
                prepared.validation_outer.coords_norm,
                prepared.validation_outer.braggs,
                prepared.validation_outer.dops,
            )
            val_outer_rmse_value = float(val_outer_rmse.numpy())
        else:
            val_outer_rmse_value = float("nan")

        history["epoch"].append(epoch)
        history["train_total_loss"].append(float(total_loss.numpy()))
        history["train_data_loss"].append(float(loss_data.numpy()))
        history["train_div_loss"].append(float(loss_div.numpy()))
        history["train_mom_loss"].append(float(loss_mom.numpy()))
        history["train_temp_loss"].append(float(loss_temp.numpy()))
        history["val_doppler_rmse"].append(val_rmse_value)
        history["val_inner_doppler_rmse"].append(val_inner_rmse_value)
        history["val_outer_doppler_rmse"].append(val_outer_rmse_value)
        history["w_data"].append(float(weights_cfg["data"]))
        history["w_div"].append(float(weights_cfg["div"]))
        history["w_mom"].append(float(weights_cfg["mom"]))
        history["w_temp"].append(float(weights_cfg["temp"]))

        grad_data_norm_value = float("nan")
        grad_div_norm_value = float("nan")
        grad_mom_norm_value = float("nan")
        grad_temp_norm_value = float("nan")
        trunk_drift_value = float("nan")
        trunk_relative_drift_value = float("nan")

        if should_report and enable_gradient_diagnostics:
            (
                grad_data_norm,
                grad_div_norm,
                grad_mom_norm,
                grad_temp_norm,
            ) = gradient_diagnostics_step(
                                            prepared.train.coords_norm,
                                            prepared.train.braggs,
                                            prepared.train.dops,
                                            prepared.train.weights,
                                            pde_t_norm,
                                            pde_z_norm,
                                            pde_x_norm,
                                            pde_y_norm,
                                            pde_rho_ratio,
                                            prepared.normalized_to_physical_grad_scale,
                                            tf.constant(float(weights_cfg["data"]), dtype=tf.float32),
                                            tf.constant(float(weights_cfg["div"]), dtype=tf.float32),
                                            tf.constant(float(weights_cfg["mom"]), dtype=tf.float32),
                                            tf.constant(float(weights_cfg["temp"]), dtype=tf.float32),
                                        )
            grad_data_norm_value = float(grad_data_norm.numpy())
            grad_div_norm_value = float(grad_div_norm.numpy())
            grad_mom_norm_value = float(grad_mom_norm.numpy())
            grad_temp_norm_value = float(grad_temp_norm.numpy())

        if should_report:
            trunk_drift, trunk_relative_drift = _shared_trunk_drift(
                trunk_trainable_variables,
                trunk_anchor_reference_weights,
            )
            if trunk_drift is not None:
                trunk_drift_value = float(trunk_drift)
            if trunk_relative_drift is not None:
                trunk_relative_drift_value = float(trunk_relative_drift)

            if prepared.validation is not None and np.isfinite(val_rmse_value) and val_rmse_value < best_val_rmse:
                best_val_rmse = val_rmse_value
                best_epoch = epoch
                model.save_weights(weights_path)
                has_saved_best = True

            elapsed = perf_counter() - train_start
            print("")
            print(
                format_progress_report(
                    epoch=epoch,
                    elapsed_seconds=elapsed,
                    total_loss=float(total_loss.numpy()),
                    data_loss=float(loss_data.numpy()),
                    div_loss=float(loss_div.numpy()),
                    mom_loss=float(loss_mom.numpy()),
                    temp_loss=float(loss_temp.numpy()),
                    w_data=float(weights_cfg["data"]),
                    w_div=float(weights_cfg["div"]),
                    w_mom=float(weights_cfg["mom"]),
                    w_temp=float(weights_cfg["temp"]),
                    val_doppler_rmse=val_rmse_value,
                    val_inner_doppler_rmse=val_inner_rmse_value,
                    val_outer_doppler_rmse=val_outer_rmse_value,
                    grad_data_norm=grad_data_norm_value,
                    grad_div_norm=grad_div_norm_value,
                    grad_mom_norm=grad_mom_norm_value,
                    grad_temp_norm=grad_temp_norm_value,
                    trunk_drift=trunk_drift_value,
                    trunk_relative_drift=trunk_relative_drift_value,
                )
            )

        history["grad_data_norm"].append(grad_data_norm_value)
        history["grad_div_norm"].append(grad_div_norm_value)
        history["grad_mom_norm"].append(grad_mom_norm_value)
        history["grad_temp_norm"].append(grad_temp_norm_value)
        history["trunk_drift"].append(trunk_drift_value)
        history["trunk_relative_drift"].append(trunk_relative_drift_value)

    history_path = write_json(run_dir, "history.json", history)

    summary_plot_path = run_dir / "training_summary.png"
    _plot_training_summary(history, summary_plot_path)

    if not has_saved_best:
        model.save_weights(weights_path)
        best_epoch = total_epochs - 1 if total_epochs > 0 else 0
        best_val_rmse = float("nan")

    if prepared.validation is not None and has_saved_best:
        model.load_weights(weights_path)

    doppler_comparison_plot_path = None

    if bool(config.output.write_plots):
        train_outputs = model(prepared.train.coords_norm, training=False).numpy()
        train_predicted_dops = doppler_from_field(
            tf.convert_to_tensor(train_outputs, dtype=tf.float32),
            prepared.train.braggs,
        ).numpy()

        validation_observed = None
        validation_predicted = None

        if prepared.validation is not None:
            validation_outputs = model(prepared.validation.coords_norm, training=False).numpy()
            validation_predicted = doppler_from_field(
                tf.convert_to_tensor(validation_outputs, dtype=tf.float32),
                prepared.validation.braggs,
            ).numpy()
            validation_observed = prepared.validation.dops.numpy()

        doppler_comparison_plot_path = run_dir / "doppler_comparison.png"
        _plot_doppler_comparison(
            prepared.train.dops.numpy(),
            train_predicted_dops,
            doppler_comparison_plot_path,
            validation_observed=validation_observed,
            validation_predicted=validation_predicted,
        )

    model_summary_path = run_dir / "model_summary.txt"
    with model_summary_path.open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))

    mean_wind_plot_path = None
    mean_wind_data_path = None

    if bool(config.output.write_plots):
        selected_df = window.selection_df.copy()

        if "used_for_training" in selected_df.columns:
            selected_df = selected_df[selected_df["used_for_training"]].copy()
        else:
            selected_df = window.training_df.copy()

        df_winds = _predicted_mean_wind(
            model,
            selected_df,
            time_base=prepared.time_base,
            lower=prepared.lower,
            upper=prepared.upper,
        )

        mean_wind_plot_path = run_dir / "mean_wind_trained_model.png"
        mean_wind_data_path = run_dir / "mean_wind_trained_model.hdf5"

        _save_mean_wind_outputs(df_winds, mean_wind_plot_path, mean_wind_data_path)

    manifest = {
        "experiment": config.experiment.name,
        "architecture": str(config.model.architecture),
        "formulation": formulation,
        "weights_file": weights_path.name,
        "best_epoch": best_epoch,
        "best_val_doppler_rmse": best_val_rmse,
        "history_file": history_path.name,
        "time_base": float(prepared.time_base),
        "lat_ref": float(window.metadata["active_center"][1]),
        "lon_ref": float(window.metadata["active_center"][0]),
        "alt_ref_km": float(window.metadata["active_center"][2]),
        "normalization": config.to_dict()["domain"]["normalization"],
        "window_metadata": window.metadata,
    }
    manifest_path = write_json(run_dir, "manifest.json", manifest)

    return TrainingArtifacts(
        run_dir=run_dir,
        history_path=history_path,
        weights_path=weights_path,
        manifest_path=manifest_path,
        summary_plot_path=summary_plot_path,
        doppler_comparison_plot_path=doppler_comparison_plot_path,
        mean_wind_plot_path=mean_wind_plot_path,
        mean_wind_data_path=mean_wind_data_path,
    )
