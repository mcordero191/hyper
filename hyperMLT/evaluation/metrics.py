from __future__ import annotations

from typing import Any

import numpy as np


def _finite_pair_mask(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(truth) & np.isfinite(pred)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return valid


def component_metrics(truth: np.ndarray, pred: np.ndarray, *, mask: np.ndarray | None = None) -> dict[str, float]:
    valid = _finite_pair_mask(truth, pred, mask=mask)
    if not np.any(valid):
        return {
            "count": 0.0,
            "cc": float("nan"),
            "rae": float("nan"),
            "nmae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
        }

    truth_v = np.asarray(truth, dtype=np.float64)[valid]
    pred_v = np.asarray(pred, dtype=np.float64)[valid]
    error = pred_v - truth_v

    if truth_v.size < 2 or np.nanstd(truth_v) <= 0.0 or np.nanstd(pred_v) <= 0.0:
        cc = float("nan")
    else:
        cc = float(np.corrcoef(truth_v, pred_v)[0, 1])

    denom_rae = float(np.sum(np.abs(truth_v - np.mean(truth_v))))
    denom_nmae = float(np.sum(np.abs(truth_v)))

    rae = float(np.sum(np.abs(error)) / denom_rae) if denom_rae > 0.0 else float("nan")
    nmae = float(np.sum(np.abs(error)) / denom_nmae) if denom_nmae > 0.0 else float("nan")
    rmse = float(np.sqrt(np.mean(error**2)))
    bias = float(np.mean(error))

    return {
        "count": float(truth_v.size),
        "cc": cc,
        "rae": rae,
        "nmae": nmae,
        "rmse": rmse,
        "bias": bias,
    }


def summarize_metrics(
    truth_fields: dict[str, np.ndarray],
    pred_fields: dict[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    return {
        component: component_metrics(truth_fields[component], pred_fields[component], mask=mask)
        for component in ("u", "v", "w")
    }
