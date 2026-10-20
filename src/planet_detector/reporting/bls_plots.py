"""Periodogram and phase-folded diagnostics for BLS searches."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import QTable
from astropy.timeseries import BoxLeastSquares


def plot_bls_search(search_dir: Path, output: Path) -> None:
    """Plot a saved search using its exact input light curve."""

    candidate = json.loads(
        (search_dir / "strongest_candidate.json").read_text(
            encoding="utf-8"
        )
    )

    periodogram = QTable.read(
        search_dir / "periodogram.ecsv",
        format="ascii.ecsv",
    )

    curve_file = Path(candidate["input_file"])

    with curve_file.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()

    if checksum != candidate["input_sha256"]:
        raise ValueError("Input light curve differs from the searched file.")

    curve = QTable.read(curve_file, format="ascii.ecsv")
    valid = np.asarray(curve["search_valid"], dtype=bool)

    time = u.Quantity(curve["time_bkjd"]).to_value(u.day)[valid]
    flux = u.Quantity(
        curve["flat_flux"]
    ).to_value(u.dimensionless_unscaled)[valid]
    error = u.Quantity(
        curve["flat_flux_err"]
    ).to_value(u.dimensionless_unscaled)[valid]

    period = float(candidate["period_days"])
    duration = float(candidate["duration_hours"]) / 24
    transit_time = float(candidate["transit_time_bkjd"])

    # Center each predicted transit at zero.
    phase_hours = (
        (time - transit_time + period / 2) % period - period / 2
    ) * 24

    half_width = min(period * 12, duration * 24 * 3)
    zoom = np.abs(phase_hours) <= half_width

    if not np.any(zoom):
        raise ValueError("No observations lie in the transit plotting window.")

    # Reconstruct the fitted box from the same data and parameters.
    model = BoxLeastSquares(time, flux, dy=error)
    phase_grid = np.linspace(-half_width, half_width, 2000)

    fitted_flux = model.model(
        transit_time + phase_grid / 24,
        period,
        duration,
        transit_time,
    )

    # Weighted phase bins are a display aid; all data enter the BLS fit.
    edges = np.linspace(-half_width, half_width, 41)
    bin_centers = []
    bin_flux = []
    bin_error = []

    for left, right in zip(edges[:-1], edges[1:]):
        selected = (phase_hours >= left) & (phase_hours < right)

        if np.count_nonzero(selected) < 3:
            continue

        weights = 1 / error[selected] ** 2

        bin_centers.append((left + right) / 2)
        bin_flux.append(
            np.sum(weights * flux[selected]) / np.sum(weights)
        )
        bin_error.append(np.sqrt(1 / np.sum(weights)))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        layout="constrained",
    )

    period_axis, folded_axis = axes

    periods = u.Quantity(periodogram["period"]).to_value(u.day)

    period_axis.plot(
        periods,
        np.asarray(periodogram["power"]),
        color="0.25",
        linewidth=0.7,
    )

    period_axis.axvline(
        period,
        color="tab:red",
        linestyle="--",
        label=f"Best period: {period:.8f} days",
    )

    period_axis.set_xscale("log")
    period_axis.set_xlabel("Trial period [days]")
    period_axis.set_ylabel("BLS power — likelihood objective")
    period_axis.set_title("Full searched period range")
    period_axis.legend()
    period_axis.grid(alpha=0.2)

    residual_ppm = (flux - 1) * 1_000_000

    folded_axis.scatter(
        phase_hours[zoom],
        residual_ppm[zoom],
        s=8,
        color="0.65",
        alpha=0.5,
        label="Individual observations",
    )

    folded_axis.errorbar(
        bin_centers,
        (np.asarray(bin_flux) - 1) * 1_000_000,
        yerr=np.asarray(bin_error) * 1_000_000,
        fmt="o",
        markersize=4,
        color="black",
        capsize=2,
        label="Weighted phase bins",
    )

    folded_axis.plot(
        phase_grid,
        (np.asarray(fitted_flux) - 1) * 1_000_000,
        color="tab:red",
        linewidth=1.8,
        label="Fitted box",
    )

    # Crop the display range while explicitly counting off-scale points.
    low, high = np.percentile(residual_ppm[zoom], [1, 99])
    padding = max(25.0, 0.2 * (high - low))
    lower_limit = low - padding
    upper_limit = high + padding

    off_scale = np.count_nonzero(
        zoom
        & (
            (residual_ppm < lower_limit)
            | (residual_ppm > upper_limit)
        )
    )

    folded_axis.set_xlim(-half_width, half_width)
    folded_axis.set_ylim(lower_limit, upper_limit)
    folded_axis.set_xlabel("Time from fitted transit center [hours]")
    folded_axis.set_ylabel("Relative flux [ppm]")
    folded_axis.set_title(
        f"Phase-folded signal; {off_scale} observations outside y-axis view"
    )
    folded_axis.legend(fontsize=8)
    folded_axis.grid(alpha=0.2)

    fig.suptitle(
        f"KIC {candidate['kic_id']} | "
        f"P = {period:.8f} days | "
        f"Box depth = {candidate['depth_ppm']:.1f} ppm"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)