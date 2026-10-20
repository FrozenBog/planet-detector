"""Diagnostic plots for extracted light curves."""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import QTable
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib.patches import Rectangle


def plot_aperture_lightcurve(curve_file: Path, output: Path) -> None:
    """Plot the recorded aperture and the extracted, undetrended flux."""

    curve = QTable.read(curve_file, format="ascii.ecsv")
    source = Path(curve.meta["source_file"])

    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()

    if source_hash != curve.meta["source_sha256"]:
        raise ValueError("Source FITS file differs from the extraction source.")

    valid = np.asarray(curve["valid"], dtype=bool)
    time = u.Quantity(curve["time_bkjd"]).to_value(u.day)
    flux = u.Quantity(curve["flux"]).to_value(u.electron / u.s)
    error = u.Quantity(curve["flux_err"]).to_value(u.electron / u.s)
    aperture = np.asarray(curve.meta["aperture_mask"], dtype=bool)

    if not np.any(valid):
        raise ValueError("The light curve contains no valid observations.")

    first_good = int(np.flatnonzero(valid)[0])

    with fits.open(source, mode="readonly", memmap=False) as hdus:
        pixel_hdu = hdus[1]

        if not isinstance(pixel_hdu, fits.BinTableHDU):
            raise ValueError("Expected a pixel binary table.")

        data = pixel_hdu.data

        if data is None:
            raise ValueError("Source pixel table is empty.")

        if not np.array_equal(
            data["CADENCENO"],
            np.asarray(curve["cadence"]),
        ):
            raise ValueError("Light curve and source cadence ordering differ.")

        image = np.array(data["FLUX"][first_good], dtype=float, copy=True)

    if aperture.shape != image.shape:
        raise ValueError("Recorded aperture does not match the source image.")

    median_flux = float(np.median(flux[valid]))

    if not np.isfinite(median_flux) or median_flux <= 0:
        raise ValueError("Expected a positive, finite median stellar flux.")

    # Normalization is for display only; the saved measurements stay unchanged.
    relative_flux = flux / median_flux
    relative_error = error / median_flux

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.5),
        gridspec_kw={"width_ratios": [1, 2]},
        layout="constrained",
    )

    image_axis, curve_axis = axes

    norm = ImageNormalize(
        vmin=float(np.nanmin(image)),
        vmax=float(np.nanmax(image)),
        stretch=AsinhStretch(),
    )

    displayed_image = image_axis.imshow(
        image,
        origin="lower",
        cmap="magma",
        norm=norm,
        interpolation="nearest",
    )

    for row, column in np.argwhere(aperture):
        image_axis.add_patch(
            Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                fill=False,
                edgecolor="cyan",
                linewidth=1.2,
            )
        )

    image_axis.set_title("First usable image; aperture in cyan")
    image_axis.set_xlabel("Postage-stamp column")
    image_axis.set_ylabel("Postage-stamp row")

    fig.colorbar(
        displayed_image,
        ax=image_axis,
        label=r"Pixel flux [$e^-\,s^{-1}$]",
        shrink=0.8,
    )

    curve_axis.errorbar(
        time[valid],
        relative_flux[valid],
        yerr=relative_error[valid],
        fmt=".",
        markersize=2,
        color="black",
        ecolor="0.65",
        elinewidth=0.5,
        capsize=0,
    )

    curve_axis.set_title("Extracted aperture flux — no detrending")
    curve_axis.set_xlabel("Time [BKJD, TDB]")
    curve_axis.set_ylabel("Flux / median valid flux")
    curve_axis.grid(alpha=0.2)

    fig.suptitle(
        f"KIC {curve.meta['kic_id']} | Quarter {curve.meta['quarter']} | "
        f"{np.count_nonzero(valid)} usable cadences"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

def plot_detrending(curve_file: Path, output: Path) -> None:
    """Show original flux, fitted trend, and detrended residuals."""

    curve = QTable.read(curve_file, format="ascii.ecsv")

    required_columns = {
        "time_bkjd",
        "flux",
        "trend_flux",
        "flat_flux",
        "flat_flux_err",
        "valid",
        "search_valid",
    }

    if not required_columns.issubset(curve.colnames):
        raise ValueError("Input file is missing detrending columns.")

    time = u.Quantity(curve["time_bkjd"]).to_value(u.day)
    flux = u.Quantity(curve["flux"]).to_value(u.electron / u.s)
    trend = u.Quantity(curve["trend_flux"]).to_value(u.electron / u.s)
    flat = u.Quantity(curve["flat_flux"]).to_value(u.dimensionless_unscaled)
    flat_error = u.Quantity(
        curve["flat_flux_err"]
    ).to_value(u.dimensionless_unscaled)

    valid = np.asarray(curve["valid"], dtype=bool)
    search_valid = np.asarray(curve["search_valid"], dtype=bool)

    if not np.any(valid) or not np.any(search_valid):
        raise ValueError("No usable observations to plot.")

    median_flux = float(np.median(flux[valid]))

    if not np.isfinite(median_flux) or median_flux <= 0:
        raise ValueError("Expected positive, finite median flux.")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        layout="constrained",
    )

    raw_axis, flat_axis = axes

    raw_axis.scatter(
        time[valid],
        (flux[valid] / median_flux - 1) * 1_000_000,
        s=5,
        color="0.35",
        label="Extracted flux",
    )

    raw_axis.plot(
        time,
        (trend / median_flux - 1) * 1_000_000,
        color="tab:orange",
        linewidth=1.8,
        label="Fitted trend",
    )

    raw_axis.set_title("Original measurements and fitted trend")
    raw_axis.set_ylabel("Relative flux [ppm]")
    raw_axis.legend()
    raw_axis.grid(alpha=0.2)

    flat_axis.errorbar(
        time[search_valid],
        (flat[search_valid] - 1) * 1_000_000,
        yerr=flat_error[search_valid] * 1_000_000,
        fmt=".",
        markersize=2,
        color="black",
        ecolor="0.7",
        elinewidth=0.5,
        capsize=0,
    )

    flat_axis.axhline(0, color="tab:orange", linewidth=1)
    flat_axis.set_title("Detrended measurements")
    flat_axis.set_xlabel("Time [BKJD, TDB]")
    flat_axis.set_ylabel("Relative flux [ppm]")
    flat_axis.grid(alpha=0.2)

    settings = curve.meta["detrending"]

    fig.suptitle(
        f"KIC {curve.meta['kic_id']} | Quarter {curve.meta['quarter']} | "
        f"Window: {settings['window_length']} observations"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)