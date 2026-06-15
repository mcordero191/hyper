from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ValidationSplit:
    training_df: pd.DataFrame
    validation_df: pd.DataFrame | None
    validation_inner_df: pd.DataFrame | None
    validation_outer_df: pd.DataFrame | None


def _radius_values(df: pd.DataFrame) -> np.ndarray:

    return np.sqrt(df["x"].to_numpy(dtype=np.float64) ** 2 + df["y"].to_numpy(dtype=np.float64) ** 2)


def _sample_holdout(
    df: pd.DataFrame,
    *,
    holdout_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:

    if df is None or df.empty:
        return df, None

    if holdout_fraction <= 0.0:
        return df.reset_index(drop=True), None

    total_samples = len(df)
    holdout_size = int(round(total_samples * holdout_fraction))
    holdout_size = max(1, min(holdout_size, total_samples - 1))

    validation_df = df.sample(n=holdout_size, random_state=random_seed)
    training_only_df = df.drop(index=validation_df.index)

    return training_only_df.reset_index(drop=True), validation_df.reset_index(drop=True)


def split_holdout(
    training_df: pd.DataFrame,
    *,
    holdout_fraction: float = 0.0,
    random_seed: int = 0,
    radial_mode: str | None = None,
    radial_quantile: float = 0.5,
) -> ValidationSplit:
    if training_df is None or training_df.empty:
        return ValidationSplit(training_df, None, None, None)

    if holdout_fraction <= 0.0:
        return ValidationSplit(training_df.reset_index(drop=True), None, None, None)

    normalized_mode = str(radial_mode or "none").strip().lower()

    if normalized_mode in {"", "none", "disabled"}:
        training_only_df, validation_df = _sample_holdout(
            training_df,
            holdout_fraction=holdout_fraction,
            random_seed=random_seed,
        )
        return ValidationSplit(training_only_df, validation_df, None, None)

    if normalized_mode != "radius_quantile":
        raise ValueError(f"Unsupported radial validation mode '{radial_mode}'.")

    radius = _radius_values(training_df)
    threshold = float(np.quantile(radius, float(radial_quantile)))

    inner_mask = radius <= threshold
    outer_mask = ~inner_mask

    inner_df = training_df.loc[inner_mask].copy()
    outer_df = training_df.loc[outer_mask].copy()

    inner_train, inner_val = _sample_holdout(
        inner_df,
        holdout_fraction=holdout_fraction,
        random_seed=random_seed,
    )
    outer_train, outer_val = _sample_holdout(
        outer_df,
        holdout_fraction=holdout_fraction,
        random_seed=random_seed + 1,
    )

    training_only_df = pd.concat([inner_train, outer_train], axis=0).reset_index(drop=True)

    validation_parts = [frame for frame in (inner_val, outer_val) if frame is not None and not frame.empty]
    validation_df = None

    if validation_parts:
        validation_df = pd.concat(validation_parts, axis=0).reset_index(drop=True)

    inner_val = None if inner_val is None or inner_val.empty else inner_val.reset_index(drop=True)
    outer_val = None if outer_val is None or outer_val.empty else outer_val.reset_index(drop=True)

    return ValidationSplit(training_only_df, validation_df, inner_val, outer_val)
