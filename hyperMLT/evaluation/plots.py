from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import FieldData


def _masked_nanmean(field: np.ndarray, horizontal_mask: np.ndarray) -> np.ndarray:
    mask4 = horizontal_mask[None, :, :, None]
    masked = np.where(mask4, np.asarray(field, dtype=np.float64), np.nan)
    return np.nanmean(masked, axis=(1, 2, 3))


def plot_metric_profiles(
    field_truth: FieldData,
    field_pred: FieldData,
    horizontal_mask: np.ndarray,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    times_h = (field_truth.t - field_truth.t[0]) / 3600.0
    components = ("u", "v", "w")
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    for axis, component in zip(axes, components):
        truth_mean = _masked_nanmean(getattr(field_truth, component), horizontal_mask)
        pred_mean = _masked_nanmean(getattr(field_pred, component), horizontal_mask)
        axis.plot(times_h, truth_mean, label=f"{component} truth", linewidth=1.5)
        axis.plot(times_h, pred_mean, label=f"{component} recon", linewidth=1.2, linestyle="--")
        axis.set_ylabel(f"{component} (m/s)")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Time from start (h)")
    fig.suptitle("Central-domain mean winds: truth vs reconstruction")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_xy_snapshot_comparison(
    field_truth: FieldData,
    field_pred: FieldData,
    horizontal_mask: np.ndarray,
    output_path: str | Path,
    *,
    time_index: int | None = None,
    alt_index: int | None = None,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_index = len(field_truth.t) // 2 if time_index is None else int(time_index)
    z_index = len(field_truth.alt_km) // 2 if alt_index is None else int(alt_index)

    lon_mesh, lat_mesh = np.meshgrid(field_truth.lon, field_truth.lat, indexing="ij")
    mask = np.asarray(horizontal_mask, dtype=bool)
    components = ("u", "v", "w")
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)

    for row, component in enumerate(components):
        truth_slice = np.asarray(getattr(field_truth, component), dtype=np.float64)[t_index, :, :, z_index]
        pred_slice = np.asarray(getattr(field_pred, component), dtype=np.float64)[t_index, :, :, z_index]
        err_slice = pred_slice - truth_slice

        vmax = np.nanmax(np.abs(np.concatenate([truth_slice.ravel(), pred_slice.ravel()])))
        emax = np.nanmax(np.abs(err_slice))

        panels = [
            (truth_slice, f"{component} truth", "RdBu_r", -vmax, vmax),
            (pred_slice, f"{component} recon", "RdBu_r", -vmax, vmax),
            (err_slice, f"{component} error", "RdBu_r", -emax, emax),
        ]

        for col, (panel, title, cmap, vmin, vmax_panel) in enumerate(panels):
            axis = axes[row, col]
            mesh = axis.pcolormesh(lon_mesh, lat_mesh, panel.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax_panel)
            axis.contour(lon_mesh, lat_mesh, mask.T.astype(float), levels=[0.5], colors="k", linewidths=0.8)
            axis.set_title(title)
            axis.grid(True, alpha=0.2)
            fig.colorbar(mesh, ax=axis, shrink=0.82)

    for axis in axes[-1, :]:
        axis.set_xlabel("Longitude")
    for axis in axes[:, 0]:
        axis.set_ylabel("Latitude")

    fig.suptitle(
        f"Truth vs reconstruction at t={field_truth.t[t_index]:.0f}s, z={field_truth.alt_km[z_index]:.1f} km"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_uncertainty_snapshot(
    lon: np.ndarray,
    lat: np.ndarray,
    alt_km: np.ndarray,
    t: np.ndarray,
    std_fields: dict[str, np.ndarray],
    horizontal_mask: np.ndarray,
    output_path: str | Path,
    *,
    time_index: int | None = None,
    alt_index: int | None = None,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_index = len(t) // 2 if time_index is None else int(time_index)
    z_index = len(alt_km) // 2 if alt_index is None else int(alt_index)
    lon_mesh, lat_mesh = np.meshgrid(lon, lat, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharex=True, sharey=True)
    for axis, component in zip(axes, ("u", "v", "w")):
        panel = np.asarray(std_fields[component], dtype=np.float64)[t_index, :, :, z_index]
        vmax = float(np.nanmax(panel)) if np.isfinite(panel).any() else 1.0
        mesh = axis.pcolormesh(lon_mesh, lat_mesh, panel.T, shading="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        axis.contour(lon_mesh, lat_mesh, horizontal_mask.T.astype(float), levels=[0.5], colors="w", linewidths=0.8)
        axis.set_title(f"std({component})")
        axis.grid(True, alpha=0.2)
        fig.colorbar(mesh, ax=axis, shrink=0.85)

    for axis in axes:
        axis.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.suptitle(f"Uncertainty snapshot at t={t[t_index]:.0f}s, z={alt_km[z_index]:.1f} km")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
