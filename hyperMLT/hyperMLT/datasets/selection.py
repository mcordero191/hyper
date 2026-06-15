from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def ensure_text_series(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: value.decode("ascii", "replace")
        if isinstance(value, (bytes, bytearray))
        else value
    )


def add_selection_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy().reset_index(drop=True)
    if "link_id" not in result.columns and "link" in result.columns:
        result["link_id"] = ensure_text_series(result["link"])
    if "tx_id" not in result.columns and "tx_station" in result.columns:
        result["tx_id"] = ensure_text_series(result["tx_station"])
    if "rx_id" not in result.columns and "rx_station" in result.columns:
        result["rx_id"] = ensure_text_series(result["rx_station"])
    if "used_for_training" not in result.columns:
        result["used_for_training"] = False
    if "split_reason" not in result.columns:
        result["split_reason"] = ""
    return result


def attach_selection_key(df: pd.DataFrame) -> pd.DataFrame:
    result = add_selection_columns(df)
    key_columns = [
        column
        for column in (
            "times",
            "lats",
            "lons",
            "heights",
            "dops",
            "braggs_x",
            "braggs_y",
            "braggs_z",
            "link_id",
        )
        if column in result.columns
    ]
    if not key_columns:
        raise ValueError("Unable to derive selection keys from dataframe columns.")
    result["_selection_key"] = result[key_columns].astype(str).agg("|".join, axis=1)
    result["_selection_rank"] = result.groupby("_selection_key").cumcount()
    return result


def build_selection_frame(full_df: pd.DataFrame, training_df: pd.DataFrame) -> pd.DataFrame:
    full_df = attach_selection_key(full_df)
    training_df = attach_selection_key(training_df)

    selected = set(
        training_df[["_selection_key", "_selection_rank"]].itertuples(index=False, name=None)
    )

    selection_df = full_df.copy()
    selection_df["used_for_training"] = [
        item in selected
        for item in selection_df[["_selection_key", "_selection_rank"]].itertuples(
            index=False,
            name=None,
        )
    ]
    selection_df["split_reason"] = np.where(
        selection_df["used_for_training"],
        "selected",
        "legacy_filter_removed",
    )

    extra_columns = [column for column in training_df.columns if column not in selection_df.columns]
    if extra_columns:
        selection_df = selection_df.merge(
            training_df[["_selection_key", "_selection_rank", *extra_columns]],
            on=["_selection_key", "_selection_rank"],
            how="left",
        )
    return selection_df


def _append_reason(df: pd.DataFrame, mask: np.ndarray, reason: str) -> None:
    if not np.any(mask):
        return
    existing = df.loc[mask, "split_reason"].fillna("")
    df.loc[mask, "split_reason"] = np.where(existing == "", reason, existing + ";" + reason)
    df.loc[mask, "used_for_training"] = False


def _mask_remove_random_fraction(
    df: pd.DataFrame,
    operation: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    fraction = float(operation.get("fraction", 0.0))
    active_indices = np.flatnonzero(df["used_for_training"].to_numpy().astype(bool))
    n_remove = int(np.floor(len(active_indices) * fraction))
    if n_remove <= 0:
        return np.zeros(len(df), dtype=bool)
    chosen = rng.choice(active_indices, size=n_remove, replace=False)
    mask = np.zeros(len(df), dtype=bool)
    mask[chosen] = True
    return mask


def _mask_remove_links(df: pd.DataFrame, operation: dict[str, Any]) -> np.ndarray:
    return df["link_id"].isin(set(operation.get("links", []))).to_numpy()


def _mask_remove_time_ranges(df: pd.DataFrame, operation: dict[str, Any]) -> np.ndarray:
    values = df["times"].to_numpy()
    mask = np.zeros(len(df), dtype=bool)
    for item in operation.get("ranges", []):
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        mask |= (values >= start) & (values <= end)
    return mask


def _mask_remove_altitude_ranges(df: pd.DataFrame, operation: dict[str, Any]) -> np.ndarray:
    values = df["heights"].to_numpy() if "heights" in df.columns else df["z"].to_numpy() * 1e-3
    mask = np.zeros(len(df), dtype=bool)
    for item in operation.get("ranges", []):
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        mask |= (values >= start) & (values <= end)
    return mask


def _mask_remove_spatial_sectors(df: pd.DataFrame, operation: dict[str, Any]) -> np.ndarray:
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    center_x = float(operation.get("center_x", 0.0))
    center_y = float(operation.get("center_y", 0.0))
    azimuth_min = float(operation.get("azimuth_min", -180.0))
    azimuth_max = float(operation.get("azimuth_max", 180.0))
    radius_min = float(operation.get("radius_min", 0.0))
    radius_max = float(operation.get("radius_max", np.inf))

    dx = x - center_x
    dy = y - center_y
    radius = np.sqrt(dx**2 + dy**2)
    azimuth = np.degrees(np.arctan2(dy, dx))

    if azimuth_min <= azimuth_max:
        azimuth_mask = (azimuth >= azimuth_min) & (azimuth <= azimuth_max)
    else:
        azimuth_mask = (azimuth >= azimuth_min) | (azimuth <= azimuth_max)
    return azimuth_mask & (radius >= radius_min) & (radius <= radius_max)


def apply_selection_operations(
    selection_df: pd.DataFrame,
    operations: list[dict[str, Any]] | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    result = add_selection_columns(selection_df)
    operations = list(operations or [])
    if not operations:
        return result

    rng = np.random.default_rng(random_seed)
    dispatch = {
        "remove_random_fraction": lambda op: _mask_remove_random_fraction(result, op, rng),
        "remove_links": lambda op: _mask_remove_links(result, op),
        "remove_time_ranges": lambda op: _mask_remove_time_ranges(result, op),
        "remove_altitude_ranges": lambda op: _mask_remove_altitude_ranges(result, op),
        "remove_spatial_sectors": lambda op: _mask_remove_spatial_sectors(result, op),
    }

    for operation in operations:
        op_type = operation.get("type")
        if op_type not in dispatch:
            raise ValueError(f"Unsupported selection operation: {op_type}")
        mask = dispatch[op_type](operation)
        mask &= result["used_for_training"].to_numpy().astype(bool)
        _append_reason(result, mask, operation.get("reason", op_type))

    return result
