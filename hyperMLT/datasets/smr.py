from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import pickle
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import pandas as pd

from hyperMLT.utils.coordinates import lla2enu
from hyperMLT.utils.console import format_summary_table

from .preprocess import FilterResult, apply_filter_pipeline
from .selection import add_selection_columns, apply_selection_operations, build_selection_frame
from .validation import split_holdout


@dataclass
class PreparedMeteorWindow:
    full_df: pd.DataFrame
    selection_df: pd.DataFrame
    training_df: pd.DataFrame
    validation_df: pd.DataFrame | None
    validation_inner_df: pd.DataFrame | None
    validation_outer_df: pd.DataFrame | None
    mean_winds: dict[str, Any] | None
    metadata: dict[str, Any] = field(default_factory=dict)


CACHE_FORMAT_VERSION = 4


def _format_link_list(values: list[str], limit: int = 5) -> str:

    if not values:
        return "-"

    if len(values) <= limit:
        return ", ".join(values)

    return f"{', '.join(values[:limit])}, ... (+{len(values) - limit})"


def _decode_strings(values) -> list[str]:

    decoded = []

    for value in values:
        if isinstance(value, (bytes, bytearray)):
            decoded.append(value.decode("ascii", "replace"))
        else:
            decoded.append(str(value))

    return decoded


def _read_optional_meteor_series(fp: h5py.File, name: str, sample_count: int) -> np.ndarray:

    if name not in fp:
        return np.full(sample_count, np.nan, dtype=np.float64)

    values = np.asarray(fp[name][()])

    if values.ndim != 1 or values.shape[0] != sample_count:
        return np.full(sample_count, np.nan, dtype=np.float64)

    return values.astype(np.float64, copy=False)


def _read_smr_file(
    file_path: Path,
    *,
    lon_center: float,
    lat_center: float,
    alt_center_km: float,
) -> pd.DataFrame:

    with h5py.File(file_path, "r") as fp:
        times = fp["t"][()]
        links = fp["link"][()]
        alt_km = fp["heights"][:] * 1e-3
        lat = fp["lats"][:]
        lon = fp["lons"][:]
        dopplers = fp["dops"][:]
        dop_errs = fp["dop_errs"][:]
        braggs = fp["braggs"][:]

        if "dcos" in fp:
            dcosx = fp["dcos"][:, 0]
            dcosy = fp["dcos"][:, 1]
        else:
            dcosx = np.zeros_like(alt_km)
            dcosy = np.zeros_like(alt_km)

        sample_count = len(times)

        u = _read_optional_meteor_series(fp, "u", sample_count)
        v = _read_optional_meteor_series(fp, "v", sample_count)
        w = _read_optional_meteor_series(fp, "w", sample_count)

        T = _read_optional_meteor_series(fp, "temp", sample_count)
        P = _read_optional_meteor_series(fp, "pres", sample_count)
        rho = _read_optional_meteor_series(fp, "rho", sample_count)
        tke = _read_optional_meteor_series(fp, "tke", sample_count)

        if "SMR_like" in fp:
            smr_like = np.asarray(fp["SMR_like"][()]).reshape(-1)
            if smr_like.shape[0] != sample_count:
                smr_like = np.ones(sample_count, dtype=np.float64)
        else:
            smr_like = np.ones(sample_count, dtype=np.float64)

    links_text = _decode_strings(links)
    tx_station = []
    rx_station = []

    for link in links_text:
        left, _, right = link.partition("_")
        tx_station.append(left)
        rx_station.append(right if right else left)

    x, y, _ = lla2enu(
        lat,
        lon,
        alt_km,
        lat_ref=lat_center,
        lon_ref=lon_center,
        alt_ref=alt_center_km,
        units="m",
    )

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    df = pd.DataFrame(
        {
            "t": pd.to_datetime(times, unit="s"),
            "times": times,
            "link": links,
            "tx_station": tx_station,
            "rx_station": rx_station,
            "lats": lat,
            "lons": lon,
            "heights": alt_km,
            "dop_errs": dop_errs,
            "braggs_x": braggs[:, 0],
            "braggs_y": braggs[:, 1],
            "braggs_z": braggs[:, 2],
            "dcosx": dcosx,
            "dcosy": dcosy,
            "dops": dopplers,
            "u": u,
            "v": v,
            "w": w,
            "T": T,
            "P": P,
            "rho": rho,
            "tke": tke,
            "SMR_like": smr_like,
            "weights": np.ones_like(times, dtype=np.float32),
            "x": x,
            "y": y,
            "z": alt_km * 1e3,
        }
    )

    df = df[df["dops"].notnull()].copy()
    df.sort_values(["times"], inplace=True, ignore_index=True)

    return df


def _cache_root(output_dir: str | Path) -> Path:

    return Path(output_dir).expanduser().resolve().parent / "_cache"


def _cache_key(
    *,
    file_path: Path,
    active_center: tuple[float, float, float],
    filter_config: dict[str, Any],
    selection_config: dict[str, Any],
    validation_config: dict[str, Any],
) -> str:

    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "file_path": str(file_path.resolve()),
        "file_size": int(file_path.stat().st_size),
        "file_mtime_ns": int(file_path.stat().st_mtime_ns),
        "active_center": list(active_center),
        "filter": filter_config,
        "selection": selection_config,
        "validation": validation_config,
    }

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()[:16]


def _cache_path(
    output_dir: str | Path,
    *,
    file_path: Path,
    active_center: tuple[float, float, float],
    filter_config: dict[str, Any],
    selection_config: dict[str, Any],
    validation_config: dict[str, Any],
) -> Path:

    cache_dir = _cache_root(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(
        file_path=file_path,
        active_center=active_center,
        filter_config=filter_config,
        selection_config=selection_config,
        validation_config=validation_config,
    )

    return cache_dir / f"prepared_window_{key}.pkl"


def _load_cached_window(path: Path) -> PreparedMeteorWindow:

    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_cached_window(path: Path, window: PreparedMeteorWindow) -> None:

    with path.open("wb") as handle:
        pickle.dump(window, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_window_summary(window: PreparedMeteorWindow) -> str:

    stats = window.metadata.get("filter_stats", {})
    default_center = tuple(window.metadata.get("default_center", (0.0, 0.0, 0.0)))
    active_center = tuple(window.metadata.get("active_center", (0.0, 0.0, 0.0)))

    rows = [
        ("File", window.metadata.get("file_name", "-")),
        ("Central date", window.metadata.get("central_date", "-")),
        ("Time window (h)", window.metadata.get("time_window", "-")),
        ("Center default", f"{default_center[0]:.2f}, {default_center[1]:.2f}, {default_center[2]:.1f}"),
        ("Center active", f"{active_center[0]:.2f}, {active_center[1]:.2f}, {active_center[2]:.1f}"),
        ("Meteors raw", stats.get("raw_samples", 0)),
        ("Meteors dedup", stats.get("deduplicated_samples", 0)),
        ("Meteors used", stats.get("used_samples", 0)),
        ("Links raw", len(stats.get("raw_links", []))),
        ("Links used", len(stats.get("used_links", []))),
        ("Links removed", _format_link_list(stats.get("removed_links", []))),
        (
            "TX stations",
            f"{len(stats.get('raw_tx_stations', []))} raw / {len(stats.get('used_tx_stations', []))} used",
        ),
        (
            "RX stations",
            f"{len(stats.get('raw_rx_stations', []))} raw / {len(stats.get('used_rx_stations', []))} used",
        ),
    ]

    return format_summary_table("Meteor Window Summary", rows)


def load_first_window(
    dataset_config: dict[str, Any],
    domain_config: dict[str, Any],
    *,
    output_dir: str | Path,
    enable_plots: bool = True,
) -> PreparedMeteorWindow:

    dataset_path = Path(dataset_config["path"]).expanduser().resolve()
    pattern = str(dataset_config.get("pattern", "*"))
    candidates = sorted(dataset_path.glob(f"{pattern}.h5"))

    if not candidates:
        raise RuntimeError(f"No SMR files matching '{pattern}.h5' in '{dataset_path}'.")

    return load_window_for_file(
        candidates[0],
        dataset_config,
        domain_config,
        output_dir=output_dir,
        enable_plots=enable_plots,
    )


def load_window_for_file(
    file_path: str | Path,
    dataset_config: dict[str, Any],
    domain_config: dict[str, Any],
    *,
    output_dir: str | Path,
    enable_plots: bool = True,
) -> PreparedMeteorWindow:

    load_start = perf_counter()
    file_path = Path(file_path).expanduser().resolve()

    with h5py.File(file_path, "r") as fp:
        default_center = (
            float(np.round(np.median(fp["lons"][:]), 1)),
            float(np.round(np.median(fp["lats"][:]), 1)),
            float(np.round(np.median(fp["heights"][:] * 1e-3), 1)),
        )

    region = domain_config.get("region", {})
    active_center = (
        float(region.get("lon_center", default_center[0])),
        float(region.get("lat_center", default_center[1])),
        float(region.get("alt_center_km", default_center[2])),
    )

    filter_config = dict(dataset_config.get("filter", {}))
    time_config = domain_config.get("time", {})
    selection_config = dict(dataset_config.get("selection", {}))
    validation_config = dict(dataset_config.get("validation", {}))

    filter_config["start_hour"] = time_config.get("start_hour", 0.0)
    filter_config["duration_hours"] = time_config.get("duration_hours", 24.0)
    filter_config["padding_fraction"] = time_config.get("padding_fraction", 0.0)

    cache_path = _cache_path(
        output_dir,
        file_path=file_path,
        active_center=active_center,
        filter_config=filter_config,
        selection_config=selection_config,
        validation_config=validation_config,
    )

    cache_lookup_elapsed = perf_counter() - load_start

    if cache_path.exists():
        cache_load_start = perf_counter()
        cached_window = _load_cached_window(cache_path)
        cache_load_elapsed = perf_counter() - cache_load_start
        cached_window.metadata = dict(cached_window.metadata)
        cached_window.metadata["cache"] = {
            "used": True,
            "path": str(cache_path),
        }
        cached_window.metadata["timings"] = {
            "cache_lookup_s": cache_lookup_elapsed,
            "cache_load_s": cache_load_elapsed,
            "total_s": perf_counter() - load_start,
        }
        return cached_window

    read_start = perf_counter()
    full_df = _read_smr_file(
        file_path,
        lon_center=active_center[0],
        lat_center=active_center[1],
        alt_center_km=active_center[2],
    )
    read_elapsed = perf_counter() - read_start

    central_date = datetime.utcfromtimestamp(float(full_df["times"].iloc[0]))

    filter_start = perf_counter()
    filter_result: FilterResult = apply_filter_pipeline(
        full_df,
        central_date=central_date,
        output_dir=str(output_dir),
        enable_plots=enable_plots,
        filter_config=filter_config,
    )
    filter_elapsed = perf_counter() - filter_start

    selection_start = perf_counter()
    selection_df = build_selection_frame(full_df, filter_result.filtered_df)

    selection_ops = selection_config.get("operations", [])
    selection_seed = int(selection_config.get("random_seed", 0))
    selection_df = apply_selection_operations(selection_df, operations=selection_ops, random_seed=selection_seed)
    selection_df = add_selection_columns(selection_df)
    selection_elapsed = perf_counter() - selection_start

    selected_training_df = selection_df[selection_df["used_for_training"]].copy()
    split_start = perf_counter()

    validation_split = split_holdout(
        selected_training_df,
        holdout_fraction=float(validation_config.get("holdout_fraction", 0.0) or 0.0),
        random_seed=int(validation_config.get("random_seed", 0)),
        radial_mode=validation_config.get("radial_mode"),
        radial_quantile=float(validation_config.get("radial_quantile", 0.5)),
    )
    split_elapsed = perf_counter() - split_start

    time_window = (
        f"{float(time_config.get('start_hour', 0.0)):.1f} -> "
        f"{float(time_config.get('start_hour', 0.0)) + float(time_config.get('duration_hours', 24.0)):.1f}"
    )

    metadata = {
        "central_date": central_date.isoformat(),
        "file_name": file_path.name,
        "time_window": time_window,
        "default_center": default_center,
        "active_center": active_center,
        "filter_stats": filter_result.stats,
        "cache": {
            "used": False,
            "path": str(cache_path),
        },
        "timings": {
            "cache_lookup_s": cache_lookup_elapsed,
            "read_file_s": read_elapsed,
            "filter_pipeline_s": filter_elapsed,
            "selection_s": selection_elapsed,
            "validation_split_s": split_elapsed,
        },
    }

    window = PreparedMeteorWindow(
        full_df=full_df,
        selection_df=selection_df,
        training_df=validation_split.training_df,
        validation_df=validation_split.validation_df,
        validation_inner_df=validation_split.validation_inner_df,
        validation_outer_df=validation_split.validation_outer_df,
        mean_winds=filter_result.mean_winds,
        metadata=metadata,
    )

    window.metadata["timings"]["total_s"] = perf_counter() - load_start
    _write_cached_window(cache_path, window)

    return window
