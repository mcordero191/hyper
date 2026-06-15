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

    if np.isfinite(val_doppler_rmse):
        lines.append(f"\tval rmse: {val_doppler_rmse:.2e}")
    if val_inner_doppler_rmse is not None and np.isfinite(val_inner_doppler_rmse):
        lines.append(f"\tval inner rmse: {val_inner_doppler_rmse:.2e}")
    if val_outer_doppler_rmse is not None and np.isfinite(val_outer_doppler_rmse):
        lines.append(f"\tval outer rmse: {val_outer_doppler_rmse:.2e}")
    if any(
        value is not None and np.isfinite(value)
        for value in (grad_data_norm, grad_div_norm, grad_mom_norm, grad_temp_norm)
    ):
        labels = ["Data", "DIV", "Momt", "Temp"]
        values = [grad_data_norm, grad_div_norm, grad_mom_norm, grad_temp_norm]
        lines.append("\t\t\t" + "\t".join(labels))
        lines.append(
            "\tgrads:  \t"
            + "\t".join(
                f"{float(value):.1e}" if value is not None and np.isfinite(value) else "nan"
                for value in values
            )
        )
    if trunk_drift is not None and np.isfinite(trunk_drift):
        lines.append(f"\ttrunk drift: {float(trunk_drift):.2e}")
    if trunk_relative_drift is not None and np.isfinite(trunk_relative_drift):
        lines.append(f"\ttrunk rel drift: {float(trunk_relative_drift):.2e}")

    return "\n".join(lines)
