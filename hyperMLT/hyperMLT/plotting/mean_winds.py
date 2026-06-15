from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def _to_datetime_numbers(times) -> np.ndarray:

    array = np.asarray(times)

    if np.issubdtype(array.dtype, np.number):
        datetimes = array.astype("datetime64[s]").astype("datetime64[ms]").astype(object)
    else:
        datetimes = array.astype("datetime64[ms]").astype(object)

    return mdates.date2num(datetimes)


def plot_mean_winds(
    t,
    h,
    u,
    v,
    w,
    *,
    figfile=None,
    vmins=None,
    vmaxs=None,
    cmap="seismic",
    ylabel="Altitude (km)",
    xlabel="Universal time",
    histogram=False,
    df_ref=None,
    titles=("Zonal wind", "Meridional wind", "Vertical wind"),
    figtitle="",
    bins=40,
):

    num = _to_datetime_numbers(t)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)

    ncols = 2 if histogram else 1
    data_fields = [u, v, w]

    ref_fields = None
    if df_ref is not None:
        alt_ref = np.asarray(df_ref["alt"])
        valid_h = (alt_ref >= np.min(h)) & (alt_ref <= np.max(h))
        ref_fields = [
            np.asarray(df_ref["u0"])[valid_h],
            np.asarray(df_ref["v0"])[valid_h],
            np.asarray(df_ref["w0"])[valid_h],
        ]

    fig = plt.figure(figsize=(8, 6))
    plt.suptitle(figtitle)

    axes = []
    images = {}

    for i, field in enumerate(data_fields):
        vmin = None if vmins is None else vmins[i]
        vmax = None if vmaxs is None else vmaxs[i]
        std = np.nanstd(field)

        ax = plt.subplot2grid((3, 2 + ncols), (i, 0), colspan=3, rowspan=1)
        ax.set_title(titles[i])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        xmin = float(np.min(num))
        xmax = float(np.max(num))

        if xmin == xmax:
            delta = 1.0 / (24.0 * 60.0)
            xmin -= delta
            xmax += delta

        ax.set_xlim(xmin, xmax)

        im = ax.pcolormesh(num, h, np.asarray(field).T, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, linestyle="--")
        ax.text(
            0.85,
            0.85,
            f"std={std:3.2f}",
            bbox=dict(boxstyle="round", fc="blanchedalmond", ec="orange", alpha=0.5),
            transform=ax.transAxes,
        )

        axes.append(ax)
        images[i] = im

        if histogram:
            ax_hist = plt.subplot2grid((3, 2 + ncols), (i, 3), colspan=1, rowspan=1)
            ax_hist.set_xlabel(f"{titles[i]} (m/s)")
            values = np.asarray(field)
            values = values[np.isfinite(values)]

            ax_hist.hist(values, bins=bins, color="grey", alpha=0.5, zorder=1, label="hyperMLT")
            ax_hist.grid(True, linestyle="--")
            ax_hist.set_ylabel("Counts")

            if vmin is not None and vmax is not None:
                ax_hist.set_xlim(vmin, vmax)
            elif values.size > 0:
                max_edge = np.nanmax(np.abs(values))
                ax_hist.set_xlim(-max_edge, max_edge)

            if ref_fields is not None:
                ref_values = ref_fields[i]
                ref_values = ref_values[np.isfinite(ref_values)]
                ax_hist.hist(ref_values, bins=bins, color="c", alpha=0.3, zorder=0, label="reference")

            ax_hist.locator_params(tight=True, nbins=4)

    for ax in axes:
        ax.label_outer()

    for i, im in images.items():
        plt.colorbar(im, ax=axes[i], label="m/s")

    plt.tight_layout(pad=0.1, rect=[0.02, 0.0, 0.99, 0.97])

    if figfile is not None:
        fig.savefig(Path(figfile))
        plt.close(fig)

        return

    plt.show()
