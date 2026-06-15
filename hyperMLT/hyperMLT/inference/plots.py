from __future__ import annotations

from pathlib import Path

import numpy as np

from hyperMLT.plotting.mean_winds import plot_mean_winds

from .runner import InferenceResult


def _absolute_times(result: InferenceResult) -> np.ndarray:

    return np.asarray(result.t, dtype=np.float64) + float(result.manifest.get("time_base", 0.0))


def _nearest_index(values: np.ndarray, target: float | None) -> int:

    if target is None:
        return len(values) // 2

    return int(np.abs(np.asarray(values, dtype=np.float64) - float(target)).argmin())


def _window_bounds(center: int, size: int, limit: int) -> tuple[int, int]:

    half = max(0, int(size) // 2)
    start = max(0, center - half)
    stop = min(limit, center + half + 1)

    if start >= stop:
        return center, min(limit, center + 1)

    return start, stop


def _nanmean_filter1d(data: np.ndarray, window: int, axis: int) -> np.ndarray:

    if int(window) <= 1:
        return np.asarray(data, dtype=np.float64)

    kernel = np.ones(int(window), dtype=np.float64)
    moved = np.moveaxis(np.asarray(data, dtype=np.float64), axis, 0)
    output = np.empty_like(moved, dtype=np.float64)

    for idx in np.ndindex(moved.shape[1:]):
        vector = moved[(slice(None),) + idx]
        valid = np.isfinite(vector).astype(np.float64)
        filled = np.nan_to_num(vector, nan=0.0)
        numerator = np.convolve(filled, kernel, mode="same")
        denominator = np.convolve(valid, kernel, mode="same")
        output[(slice(None),) + idx] = np.where(denominator > 0.0, numerator / denominator, np.nan)

    return np.moveaxis(output, 0, axis)


def _remove_background(
    data: np.ndarray,
    *,
    t_seconds: np.ndarray,
    alt_km: np.ndarray,
    background_time_window_hours: float,
    background_alt_window_km: float,
) -> np.ndarray:

    if len(t_seconds) > 1:
        dt_hours = float(np.nanmedian(np.diff(t_seconds))) / 3600.0
    else:
        dt_hours = float(background_time_window_hours)

    if len(alt_km) > 1:
        dz_km = float(np.nanmedian(np.diff(alt_km)))
    else:
        dz_km = float(background_alt_window_km)

    time_window = max(1, int(round(float(background_time_window_hours) / max(dt_hours, 1e-6))))
    alt_window = max(1, int(round(float(background_alt_window_km) / max(dz_km, 1e-6))))

    background = _nanmean_filter1d(data, time_window, axis=0)
    background = _nanmean_filter1d(background, alt_window, axis=-1)
    return np.asarray(data, dtype=np.float64) - background


def _plot_mean_wind_diagnostic(
    result: InferenceResult,
    output_dir: Path,
    extension: str,
    plotting_cfg: dict,
) -> list[Path]:

    mean_cfg = dict(plotting_cfg.get("mean_winds", {}))

    if not bool(mean_cfg.get("enabled", True)):
        return []

    times = _absolute_times(result)
    u_mean = np.nanmean(result.u, axis=(1, 2))
    v_mean = np.nanmean(result.v, axis=(1, 2))
    w_mean = np.nanmean(result.w, axis=(1, 2))

    output_path = output_dir / f"mean_winds.{extension}"
    plot_mean_winds(
        times,
        result.alt_km,
        u_mean,
        v_mean,
        w_mean,
        figfile=output_path,
        histogram=bool(mean_cfg.get("histogram", True)),
        vmins=mean_cfg.get("vmins", [-100, -100, -10]),
        vmaxs=mean_cfg.get("vmaxs", [100, 100, 10]),
        figtitle=str(mean_cfg.get("title", "Mean winds from inferred field")),
    )
    return [output_path]


def _slice_indices(result: InferenceResult, slices_cfg: dict) -> tuple[int, int, int]:

    lon0 = slices_cfg.get("lon0")
    lat0 = slices_cfg.get("lat0")
    alt0 = slices_cfg.get("alt0")

    lon_index = _nearest_index(result.lon, lon0)
    lat_index = _nearest_index(result.lat, lat0)
    alt_index = _nearest_index(result.alt_km, alt0)
    return lon_index, lat_index, alt_index


def _slice_windows(
    result: InferenceResult,
    lon_index: int,
    lat_index: int,
    alt_index: int,
    slices_cfg: dict,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:

    if not bool(slices_cfg.get("plot_mean", False)):
        return (
            _window_bounds(lon_index, 1, len(result.lon)),
            _window_bounds(lat_index, 1, len(result.lat)),
            _window_bounds(alt_index, 1, len(result.alt_km)),
        )

    average_fraction = float(slices_cfg.get("average_fraction", 0.7))
    lon_window = max(1, int(round(average_fraction * len(result.lon))))
    lat_window = max(1, int(round(average_fraction * len(result.lat))))
    alt_window = max(1, int(round(average_fraction * len(result.alt_km))))

    return (
        _window_bounds(lon_index, lon_window, len(result.lon)),
        _window_bounds(lat_index, lat_window, len(result.lat)),
        _window_bounds(alt_index, alt_window, len(result.alt_km)),
    )


def _plot_time_coordinate_slices(
    result: InferenceResult,
    output_dir: Path,
    extension: str,
    plotting_cfg: dict,
) -> list[Path]:

    slices_cfg = dict(plotting_cfg.get("time_coordinate_slices", {}))

    if not bool(slices_cfg.get("enabled", True)):
        return []

    lon_index, lat_index, alt_index = _slice_indices(result, slices_cfg)
    (lon0, lon1), (lat0, lat1), (alt0, alt1) = _slice_windows(
        result,
        lon_index,
        lat_index,
        alt_index,
        slices_cfg,
    )

    u = np.asarray(result.u, dtype=np.float64)
    v = np.asarray(result.v, dtype=np.float64)
    w = np.asarray(result.w, dtype=np.float64)

    if bool(slices_cfg.get("plot_residual", False)):
        residual_cfg = dict(slices_cfg.get("residual", {}))
        u = _remove_background(
            u,
            t_seconds=np.asarray(result.t, dtype=np.float64),
            alt_km=np.asarray(result.alt_km, dtype=np.float64),
            background_time_window_hours=float(residual_cfg.get("background_time_window_hours", 4.0)),
            background_alt_window_km=float(residual_cfg.get("background_alt_window_km", 2.0)),
        )
        v = _remove_background(
            v,
            t_seconds=np.asarray(result.t, dtype=np.float64),
            alt_km=np.asarray(result.alt_km, dtype=np.float64),
            background_time_window_hours=float(residual_cfg.get("background_time_window_hours", 4.0)),
            background_alt_window_km=float(residual_cfg.get("background_alt_window_km", 2.0)),
        )
        w = _remove_background(
            w,
            t_seconds=np.asarray(result.t, dtype=np.float64),
            alt_km=np.asarray(result.alt_km, dtype=np.float64),
            background_time_window_hours=float(residual_cfg.get("background_time_window_hours", 4.0)),
            background_alt_window_km=float(residual_cfg.get("background_alt_window_km", 2.0)),
        )

    times = _absolute_times(result)
    suffix = "residual" if bool(slices_cfg.get("plot_residual", False)) else "raw"
    vmins = slices_cfg.get("vmins", [-100, -100, -10])
    vmaxs = slices_cfg.get("vmaxs", [100, 100, 10])

    plots: list[Path] = []

    slices = [
        (
            "keox",
            result.lon,
            np.nanmean(u[:, :, lat0:lat1, alt0:alt1], axis=(2, 3)),
            np.nanmean(v[:, :, lat0:lat1, alt0:alt1], axis=(2, 3)),
            np.nanmean(w[:, :, lat0:lat1, alt0:alt1], axis=(2, 3)),
            f"[lat={result.lat[lat_index]:.2f}, alt={result.alt_km[alt_index]:.1f} km]",
            "Longitude",
        ),
        (
            "keoy",
            result.lat,
            np.nanmean(u[:, lon0:lon1, :, alt0:alt1], axis=(1, 3)),
            np.nanmean(v[:, lon0:lon1, :, alt0:alt1], axis=(1, 3)),
            np.nanmean(w[:, lon0:lon1, :, alt0:alt1], axis=(1, 3)),
            f"[lon={result.lon[lon_index]:.2f}, alt={result.alt_km[alt_index]:.1f} km]",
            "Latitude",
        ),
        (
            "keoz",
            result.alt_km,
            np.nanmean(u[:, lon0:lon1, lat0:lat1, :], axis=(1, 2)),
            np.nanmean(v[:, lon0:lon1, lat0:lat1, :], axis=(1, 2)),
            np.nanmean(w[:, lon0:lon1, lat0:lat1, :], axis=(1, 2)),
            f"[lon={result.lon[lon_index]:.2f}, lat={result.lat[lat_index]:.2f}]",
            "Altitude (km)",
        ),
    ]

    for name, coords, u_slice, v_slice, w_slice, title, ylabel in slices:
        output_path = output_dir / f"{name}_{suffix}.{extension}"
        plot_mean_winds(
            times,
            coords,
            u_slice,
            v_slice,
            w_slice,
            figfile=output_path,
            histogram=True,
            vmins=vmins,
            vmaxs=vmaxs,
            figtitle=title,
            ylabel=ylabel,
            xlabel="Universal time",
        )
        plots.append(output_path)

    return plots


def plot_inference_diagnostics(
    result: InferenceResult,
    output_dir: str | Path,
    plotting_cfg: dict | None = None,
) -> list[Path]:

    plotting_cfg = dict(plotting_cfg or {})
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = str(plotting_cfg.get("extension", "png"))

    outputs: list[Path] = []
    outputs.extend(_plot_mean_wind_diagnostic(result, output_dir, extension, plotting_cfg))
    outputs.extend(_plot_time_coordinate_slices(result, output_dir, extension, plotting_cfg))
    return outputs
