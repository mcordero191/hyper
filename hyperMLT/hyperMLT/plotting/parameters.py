from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _zenith_degrees(df: pd.DataFrame) -> np.ndarray:

    dxy = np.sqrt(df["dcosx"].to_numpy(dtype=np.float64) ** 2 + df["dcosy"].to_numpy(dtype=np.float64) ** 2)

    return np.degrees(np.arcsin(np.clip(dxy, 0.0, 1.0)))


def plot_parameter_histograms(
    selected_df: pd.DataFrame,
    discarded_df: pd.DataFrame,
    figfile: str | Path,
    bins: int = 40,
) -> None:

    if selected_df.empty and discarded_df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    selected = {
        "dops": selected_df["dops"].to_numpy(dtype=np.float64) if "dops" in selected_df else np.array([]),
        "dop_errs": selected_df["dop_errs"].to_numpy(dtype=np.float64) if "dop_errs" in selected_df else np.array([]),
        "zenith": _zenith_degrees(selected_df) if not selected_df.empty else np.array([]),
        "braggs_x": selected_df["braggs_x"].to_numpy(dtype=np.float64) if "braggs_x" in selected_df else np.array([]),
        "braggs_y": selected_df["braggs_y"].to_numpy(dtype=np.float64) if "braggs_y" in selected_df else np.array([]),
        "braggs_z": selected_df["braggs_z"].to_numpy(dtype=np.float64) if "braggs_z" in selected_df else np.array([]),
    }
    discarded = {
        "dops": discarded_df["dops"].to_numpy(dtype=np.float64) if "dops" in discarded_df else np.array([]),
        "dop_errs": discarded_df["dop_errs"].to_numpy(dtype=np.float64) if "dop_errs" in discarded_df else np.array([]),
        "zenith": _zenith_degrees(discarded_df) if not discarded_df.empty else np.array([]),
        "braggs_x": discarded_df["braggs_x"].to_numpy(dtype=np.float64) if "braggs_x" in discarded_df else np.array([]),
        "braggs_y": discarded_df["braggs_y"].to_numpy(dtype=np.float64) if "braggs_y" in discarded_df else np.array([]),
        "braggs_z": discarded_df["braggs_z"].to_numpy(dtype=np.float64) if "braggs_z" in discarded_df else np.array([]),
    }

    panels = [
        ("Doppler (Hz)", "dops"),
        ("Doppler error (Hz)", "dop_errs"),
        ("Zenith (deg)", "zenith"),
        (r"$k_x$", "braggs_x"),
        (r"$k_y$", "braggs_y"),
        (r"$k_z$", "braggs_z"),
    ]

    for axis, (xlabel, key) in zip(axes.ravel(), panels):
        selected_values = selected[key]
        discarded_values = discarded[key]

        if selected_values.size > 0:
            axis.hist(
                selected_values[np.isfinite(selected_values)],
                bins=bins,
                alpha=0.55,
                color="tab:blue",
                label="Selected",
            )

        if discarded_values.size > 0:
            axis.hist(
                discarded_values[np.isfinite(discarded_values)],
                bins=bins,
                alpha=0.55,
                color="tab:orange",
                label="Discarded",
            )

        axis.set_xlabel(xlabel)
        axis.set_ylabel("Counts")
        axis.set_yscale("log")
        axis.grid(True)
        axis.legend()

    fig.tight_layout()
    fig.savefig(figfile)
    plt.close(fig)


def _time_hours(df: pd.DataFrame) -> np.ndarray:

    times = df["times"].to_numpy(dtype=np.float64)

    if times.size == 0:
        return np.array([], dtype=np.float64)

    return (times - np.nanmin(times)) / 3600.0


def _sample_rows(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:

    if len(df) <= max_points:
        return df

    return df.sample(n=max_points, random_state=seed)


def plot_spacetime_sampling(
    selected_df: pd.DataFrame,
    discarded_df: pd.DataFrame,
    figfile: str | Path,
    *,
    max_points: int = 30000,
) -> None:

    if selected_df.empty and discarded_df.empty:
        return

    selected_plot = _sample_rows(selected_df, max_points=max_points, seed=0)
    discarded_plot = _sample_rows(discarded_df, max_points=max_points, seed=1)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    def _coords(df: pd.DataFrame):

        return {
            "t": _time_hours(df),
            "x": df["x"].to_numpy(dtype=np.float64) * 1e-3,
            "y": df["y"].to_numpy(dtype=np.float64) * 1e-3,
            "z": df["z"].to_numpy(dtype=np.float64) * 1e-3,
        }

    selected = _coords(selected_plot)
    discarded = _coords(discarded_plot)

    panels = [
        ("t", "x", "Time (h)", "X (km)"),
        ("t", "y", "Time (h)", "Y (km)"),
        ("t", "z", "Time (h)", "Z (km)"),
        ("x", "y", "X (km)", "Y (km)"),
        ("x", "z", "X (km)", "Z (km)"),
        ("y", "z", "Y (km)", "Z (km)"),
    ]

    for axis, (xkey, ykey, xlabel, ylabel) in zip(axes.ravel(), panels):
        if discarded[xkey].size > 0:
            axis.scatter(
                discarded[xkey],
                discarded[ykey],
                s=3,
                alpha=0.08,
                color="tab:orange",
                label="Discarded",
                rasterized=True,
            )

        if selected[xkey].size > 0:
            axis.scatter(
                selected[xkey],
                selected[ykey],
                s=3,
                alpha=0.10,
                color="tab:blue",
                label="Selected",
                rasterized=True,
            )

        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(figfile, dpi=200)
    plt.close(fig)
