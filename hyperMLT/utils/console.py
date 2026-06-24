from __future__ import annotations

from typing import Any

import numpy as np


def format_summary_table(title: str, rows: list[tuple[str, Any]]) -> str:

    label_width = max(len(str(label)) for label, _ in rows)
    value_width = max(len(str(value)) for _, value in rows)
    border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"

    lines = [title, border]

    for label, value in rows:
        lines.append(f"| {str(label):<{label_width}} | {str(value):<{value_width}} |")

    lines.append(border)

    return "\n".join(lines)


def format_dataset_summary(
    *,
    window,
    dataset_source: str,
    validation_config: dict[str, Any] | None = None,
    noise_config: dict[str, Any] | None = None,
) -> str:

    stats = dict(window.metadata.get("filter_stats", {}))
    cache_info = dict(window.metadata.get("cache", {}))
    timings = dict(window.metadata.get("timings", {}))
    center = tuple(window.metadata.get("active_center", (0.0, 0.0, 0.0)))
    validation_config = dict(validation_config or {})
    noise_config = dict(noise_config or {})

    validation_mode = "-"

    if float(validation_config.get("holdout_fraction", 0.0) or 0.0) > 0.0:
        radial_mode = validation_config.get("radial_mode")

        if radial_mode:
            validation_mode = f"{radial_mode} ({float(validation_config.get('radial_quantile', 0.5)):.2f})"
        else:
            validation_mode = f"random ({float(validation_config.get('holdout_fraction', 0.0)):.2f})"

    rows = [
        ("Source", dataset_source),
        ("File", window.metadata.get("file_name", "-")),
        ("Time window (h)", window.metadata.get("time_window", "-")),
        ("Center", f"{center[0]:.2f}, {center[1]:.2f}, {center[2]:.1f}"),
        ("Meteors", f"{stats.get('raw_samples', 0)} raw / {stats.get('deduplicated_samples', 0)} dedup / {stats.get('used_samples', 0)} used"),
        ("Train / Val", f"{len(window.training_df)} / {0 if window.validation_df is None else len(window.validation_df)}"),
        ("Val inner / outer", f"{0 if getattr(window, 'validation_inner_df', None) is None else len(window.validation_inner_df)} / {0 if getattr(window, 'validation_outer_df', None) is None else len(window.validation_outer_df)}"),
        ("Links", f"{len(stats.get('raw_links', []))} raw / {len(stats.get('used_links', []))} used"),
        ("Validation split", validation_mode),
        (
            "Doppler noise",
            (
                f"std={float(noise_config.get('doppler_std', 0.0) or 0.0):.2f}, "
                f"train={bool(noise_config.get('apply_to_train', True))}, "
                f"val={bool(noise_config.get('apply_to_validation', True))}"
            ),
        ),
        ("Cache used", bool(cache_info.get("used", False))),
        ("Prep time", f"{float(timings.get('total_s', 0.0)):.2f}s"),
    ]

    return format_summary_table("Dataset Summary", rows)


def format_domain_summary(
    *,
    center: tuple[float, float, float],
    center_mode: str,
    normalization: dict[str, Any],
) -> str:

    rows = [
        ("Center mode", center_mode),
        ("Center", f"{center[0]:.2f}, {center[1]:.2f}, {center[2]:.1f}"),
        ("Normalization", str(normalization.get("mode", "fixed"))),
        ("Lower bounds", normalization.get("lower_bounds", "-")),
        ("Upper bounds", normalization.get("upper_bounds", "-")),
    ]

    return format_summary_table("Domain Summary", rows)


def format_physics_summary(
    *,
    formulation_label: str,
    formulation_name: str,
    sampling: dict[str, Any],
    scheduling: dict[str, Any],
    residual_weights: dict[str, Any] | None = None,
    filter_config: dict[str, Any] | None = None,
) -> str:

    filter_config = dict(filter_config or {})
    residual_weights = dict(residual_weights or {})
    stages = list(scheduling.get("stages", []))

    if stages:
        stage_text = "; ".join(
            (
                f"{int(stage.get('start', 0))}:"
                f"d={float(dict(stage.get('weights', {})).get('data', 0.0)):.0e},"
                f"div={float(dict(stage.get('weights', {})).get('div', 0.0)):.0e},"
                f"mom={float(dict(stage.get('weights', {})).get('mom', 0.0)):.0e}"
            )
            for stage in stages
        )
    else:
        weights = dict(scheduling.get("weights", {})) or residual_weights
        stage_text = (
            f"fixed:"
            f"d={float(weights.get('data', 0.0)):.0e},"
            f"div={float(weights.get('div', 0.0)):.0e},"
            f"mom={float(weights.get('mom', 0.0)):.0e}"
        )

    rows = [
        ("Formulation", f"{formulation_label} ({formulation_name})"),
        ("PDE sampling", str(sampling.get("method", "random"))),
        ("PDE samples", int(sampling.get("sample_count", 0) or 0)),
        ("Scheduler", str(scheduling.get("type", "fixed"))),
        ("Schedule", stage_text),
        ("SMR_like only", bool(filter_config.get("require_smr_like", False))),
        ("Cluster filter", bool(filter_config.get("enable_doppler_clustering_filter", False))),
        ("Mean-wind filter", bool(filter_config.get("enable_mean_wind_quality_filter", False))),
    ]

    return format_summary_table("Physics Summary", rows)


def format_progress_report(
    *,
    epoch: int,
    elapsed_seconds: float,
    total_loss: float,
    data_loss: float,
    div_loss: float,
    mom_loss: float,
    temp_loss: float,
    w_data: float,
    w_div: float,
    w_mom: float,
    w_temp: float,
    val_doppler_rmse: float,
    val_inner_doppler_rmse: float | None = None,
    val_outer_doppler_rmse: float | None = None,
    grad_data_norm: float | None = None,
    grad_div_norm: float | None = None,
    grad_mom_norm: float | None = None,
    grad_temp_norm: float | None = None,
    trunk_drift: float | None = None,
    trunk_relative_drift: float | None = None,
) -> str:

    loss_labels = ["Total", "Data", "DIV", "Momt", "Temp"]
    losses = [total_loss, data_loss, div_loss, mom_loss, temp_loss]
    weights = [w_data, w_div, w_mom, w_temp]

    lines = [f"epoch: {epoch}, elps: {elapsed_seconds:.1f}s"]
    lines.append("\t\t\t" + "\t".join(loss_labels))
    lines.append("\tlosses: \t" + "\t".join(f"{value:.1e}" for value in losses))
    lines.append("\tweights: \t\t" + "\t".join(f"{value:.1e}" for value in weights))

    if any(
        value is not None and np.isfinite(value)
        for value in (grad_data_norm, grad_div_norm, grad_mom_norm, grad_temp_norm)
    ):
        labels = [" "]
        values = [grad_data_norm, grad_div_norm, grad_mom_norm, grad_temp_norm]
        lines.append("\t\t\t" + "\t".join(labels))
        lines.append(
            "\tgrads:  \t\t"
            + "\t".join(
                f"{float(value):.1e}" if value is not None and np.isfinite(value) else "nan"
                for value in values
            )
        )
    
    if np.isfinite(val_doppler_rmse):
        lines.append(f" ")
        lines.append(f"\tval rmse: {val_doppler_rmse:.2e}")
    if val_inner_doppler_rmse is not None and np.isfinite(val_inner_doppler_rmse):
        lines.append(f"\t\tinner rmse: {val_inner_doppler_rmse:.2e}")
    if val_outer_doppler_rmse is not None and np.isfinite(val_outer_doppler_rmse):
        lines.append(f"\t\touter rmse: {val_outer_doppler_rmse:.2e}")

    if trunk_drift is not None and np.isfinite(trunk_drift):
        lines.append(f" ")
        lines.append(f"\ttrunk drift: {float(trunk_drift):.2e}")
    if trunk_relative_drift is not None and np.isfinite(trunk_relative_drift):
        lines.append(f"\ttrunk rel drift: {float(trunk_relative_drift):.2e}")

    return "\n".join(lines)
