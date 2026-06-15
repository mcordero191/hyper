from __future__ import annotations

import numpy as np
import pandas as pd

from hyperMLT.utils.coordinates import lat2km, lon2km


def distance_weight(df, t0, h0, dt, dh, *, window="gaussian", sigma=0.5):

    if window == "rectangular":
        mask = (np.abs(df["times"] - t0) <= dt / 2) & (np.abs(df["heights"] - h0) <= dh / 2)
        weights = np.ones_like(np.where(mask)[0], dtype=np.float64)
    elif window == "gaussian":
        mask = (np.abs(df["times"] - t0) <= 3 * dt / 2) & (np.abs(df["heights"] - h0) <= 3 * dh / 2)
        dfm = df[mask]

        sigma_t = dt / 2
        sigma_h = dh / 2

        dt_ = (dfm["times"].to_numpy() - t0) / sigma_t
        dh_ = (dfm["heights"].to_numpy() - h0) / sigma_h
        weights = np.exp(-0.5 * (dt_**2 + dh_**2))
    else:
        raise ValueError(f"Unsupported mean-wind window '{window}'.")

    return mask, weights


def mean_wind_grad(
    df,
    *,
    times=None,
    alts=None,
    dt=60 * 60,
    dh=1.0,
    outlier_sigma=4.0,
    gradients=False,
    min_number_of_measurements=10,
    overlapping=2.0,
    window="gaussian",
):

    df = df.copy()

    lat0 = df["lats"].median()
    lon0 = df["lons"].median()

    if alts is None:
        hmin = int(df["heights"].min() * 2) // 2
        hmax = np.ceil(df["heights"].max() * 2) // 2
        alts = np.arange(hmin, hmax + dh, dh / overlapping)

    if times is None:
        tmin = int(df["times"].min() / (30 * 60)) * 30 * 60
        tmax = np.ceil(df["times"].max() / (30 * 60)) * 30 * 60
        times = np.arange(tmin, tmax + dt, dt / overlapping)

    ntimes = len(times)
    nalts = len(alts)
    n_par = 6 if gradients else 2

    u = np.zeros((n_par, ntimes, nalts)) + np.nan
    ue = np.zeros((n_par, ntimes, nalts)) + np.nan

    df.loc[:, "quality"] = 0.0

    for ti, t0 in enumerate(times):
        for hi, h0 in enumerate(alts):
            mask_i, weights = distance_weight(df, t0, h0, dt, dh, window=window)
            n_meas = np.count_nonzero(mask_i)

            if n_meas < min_number_of_measurements:
                continue

            if np.sum(weights) < min_number_of_measurements:
                continue

            df_i = df[mask_i]
            A = np.empty((n_meas, n_par))
            m = -2.0 * np.pi * df_i["dops"].values

            A[:, 0] = df_i["braggs_x"].values
            A[:, 1] = df_i["braggs_y"].values

            if gradients:
                latkm = lat2km(df_i["lats"].values, lat0)
                lonkm = lon2km(df_i["lons"].values, df_i["lats"].values, lon0)

                A[:, 2] = df_i["braggs_x"].values * lonkm
                A[:, 3] = df_i["braggs_x"].values * latkm
                A[:, 4] = df_i["braggs_y"].values * lonkm
                A[:, 5] = df_i["braggs_y"].values * latkm

            W = np.diag(np.sqrt(weights))
            A = W @ A
            m = W @ m

            try:
                uhat = np.linalg.lstsq(A, m, rcond=None)[0]
            except Exception:
                continue

            resid = np.abs(m - np.dot(A, uhat))
            resid_std = 0.7 * np.median(resid)
            gidx = np.where(resid < outlier_sigma * resid_std)[0]

            if len(gidx) < min_number_of_measurements:
                continue

            A2 = A[gidx, :]
            m2 = m[gidx]

            try:
                uhat2 = np.linalg.lstsq(A2, m2, rcond=None)[0]
                stdev = np.sqrt(np.diag(np.linalg.inv(np.dot(np.transpose(A2), A2)))) * resid_std
            except Exception:
                continue

            resid = np.abs(m - np.dot(A, uhat2))
            resid_std = 0.7 * np.median(resid)
            valid = np.where(resid < outlier_sigma * resid_std, 1.0, 0.0)
            df.loc[mask_i, "quality"] += valid

            for pi in range(n_par):
                u[pi, ti, hi] = uhat2[pi]
                ue[pi, ti, hi] = stdev[pi]

    df_filtered = df[df["quality"] > overlapping]

    return {
        "u0": u[0],
        "v0": u[1],
        "w0": u[0] * 0.0,
        "u0_err": ue[0],
        "v0_err": ue[1],
        "w0_err": ue[0] * 0.0,
        "times": times,
        "alts": alts,
    }, df_filtered


def mean_wind_from_fields(
    df: pd.DataFrame,
    *,
    u_key: str = "u_pred",
    v_key: str = "v_pred",
    w_key: str = "w_pred",
    times=None,
    alts=None,
    dt=60 * 60,
    dh=1.0,
    min_number_of_measurements=10,
    overlapping=2.0,
    window="gaussian",
    gradients=True,
):

    df = df.copy()

    if alts is None:
        hmin = int(df["heights"].min() * 2) // 2
        hmax = np.ceil(df["heights"].max() * 2) // 2
        alts = np.arange(hmin, hmax + dh, dh / overlapping)

    if times is None:
        tmin = int(df["times"].min() / (30 * 60)) * 30 * 60
        tmax = np.ceil(df["times"].max() / (30 * 60)) * 30 * 60
        times = np.arange(tmin, tmax + dt, dt / overlapping)

    ntimes = len(times)
    nalts = len(alts)

    u0 = np.zeros((ntimes, nalts)) + np.nan
    v0 = np.zeros((ntimes, nalts)) + np.nan
    w0 = np.zeros((ntimes, nalts)) + np.nan
    u0_err = np.zeros((ntimes, nalts)) + np.nan
    v0_err = np.zeros((ntimes, nalts)) + np.nan
    w0_err = np.zeros((ntimes, nalts)) + np.nan

    lat0 = df["lats"].median()
    lon0 = df["lons"].median()

    for ti, t0 in enumerate(times):
        for hi, h0 in enumerate(alts):
            mask_i, weights = distance_weight(df, t0, h0, dt, dh, window=window)
            n_meas = np.count_nonzero(mask_i)

            if n_meas < min_number_of_measurements:
                continue

            df_i = df[mask_i]
            weights = np.asarray(weights, dtype=np.float64)

            if np.sum(weights) < min_number_of_measurements:
                continue

            latkm = lat2km(df_i["lats"].to_numpy(dtype=np.float64), lat0)
            lonkm = lon2km(
                df_i["lons"].to_numpy(dtype=np.float64),
                df_i["lats"].to_numpy(dtype=np.float64),
                lon0,
            )

            for key, out, out_err in (
                (u_key, u0, u0_err),
                (v_key, v0, v0_err),
                (w_key, w0, w0_err),
            ):
                values = df_i[key].to_numpy(dtype=np.float64)
                valid = np.isfinite(values)

                if np.count_nonzero(valid) < min_number_of_measurements:
                    continue

                values = values[valid]
                local_weights = weights[valid]
                weight_sum = np.sum(local_weights)

                if weight_sum <= 0.0:
                    continue
                
                if gradients:
                    local_lonkm = lonkm[valid]
                    local_latkm = latkm[valid]
                
                    A = np.column_stack(
                        [
                            np.ones_like(values),
                            local_lonkm,
                            local_latkm,
                        ]
                    )
                else:
                    A = np.ones((len(values), 1), dtype=np.float64)

                Aw = A * np.sqrt(local_weights)[:, None]
                bw = values * np.sqrt(local_weights)

                try:
                    coeffs = np.linalg.lstsq(Aw, bw, rcond=None)[0]
                except Exception:
                    continue

                fitted = A @ coeffs
                resid = values - fitted
                variance = np.sum(local_weights * resid**2) / weight_sum

                if len(values) > A.shape[1]:
                    sigma = np.sqrt(variance)
                    try:
                        cov = np.linalg.inv(Aw.T @ Aw) * sigma**2
                        error = np.sqrt(max(cov[0, 0], 0.0))
                    except Exception:
                        error = np.sqrt(max(variance, 0.0))
                else:
                    error = np.sqrt(max(variance, 0.0))

                out[ti, hi] = coeffs[0]
                out_err[ti, hi] = error

    return {
        "u0": u0,
        "v0": v0,
        "w0": w0,
        "u0_err": u0_err,
        "v0_err": v0_err,
        "w0_err": w0_err,
        "times": times,
        "alts": alts,
    }
