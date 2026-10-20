"""Initial weighted Box Least Squares transit search."""

import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import QTable
from astropy.timeseries import BoxLeastSquares


def search_bls(
    curve_file: Path,
    minimum_period: float = 0.5,
    maximum_period: float = 10.0,
) -> tuple[QTable, dict]:
    """Return the full periodogram and strongest unvetted hypothesis."""

    curve = QTable.read(curve_file, format="ascii.ecsv")

    if not 0 < minimum_period < maximum_period:
        raise ValueError("Require 0 < minimum period < maximum period.")

    if curve.meta["time_scale"] != "tdb":
        raise ValueError("Expected timestamps in the TDB time scale.")

    valid = np.asarray(curve["search_valid"], dtype=bool)

    time = u.Quantity(curve["time_bkjd"]).to_value(u.day)[valid]
    flux = u.Quantity(
        curve["flat_flux"]
    ).to_value(u.dimensionless_unscaled)[valid]
    error = u.Quantity(
        curve["flat_flux_err"]
    ).to_value(u.dimensionless_unscaled)[valid]

    if len(time) < 20:
        raise ValueError("Too few usable observations for this search.")

    if not np.all(
        np.isfinite(time)
        & np.isfinite(flux)
        & np.isfinite(error)
        & (error > 0)
    ):
        raise ValueError("Search data contain invalid values.")

    if np.any(np.diff(time) <= 0):
        raise ValueError("Search timestamps must be strictly increasing.")

    baseline = float(np.ptp(time))

    if maximum_period > baseline / 3:
        raise ValueError(
            "For this initial search, maximum period must be no greater "
            f"than one third of the baseline: {baseline / 3:.3f} days."
        )

    duration_hours = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
    durations = duration_hours / 24

    if minimum_period <= 2 * np.max(durations):
        raise ValueError(
            "Minimum period must exceed twice the longest trial duration."
        )

    model = BoxLeastSquares(time, flux, dy=error)

    print("Computing BLS periodogram...", flush=True)

    result = model.autopower(
        durations,
        minimum_period=minimum_period,
        maximum_period=maximum_period,
        frequency_factor=1.0,
        oversample=20,
        objective="likelihood",
        method="fast",
    )

    power = np.asarray(result.power, dtype=float)
    finite_power = np.isfinite(power)

    if not np.any(finite_power):
        raise RuntimeError("BLS produced no finite search statistics.")

    best = int(np.argmax(np.where(finite_power, power, -np.inf)))

    period = float(result.period[best])
    duration = float(result.duration[best])
    transit_time = float(result.transit_time[best])

    in_transit = np.asarray(
        model.transit_mask(time, period, duration, transit_time),
        dtype=bool,
    )

    cycle_numbers = np.rint(
        (time[in_transit] - transit_time) / period
    ).astype(np.int64)

    cycles, cycle_counts = np.unique(
        cycle_numbers,
        return_counts=True,
    )

    periodogram = QTable()
    periodogram["period"] = np.asarray(result.period) * u.day
    periodogram["duration"] = np.asarray(result.duration) * u.day
    periodogram["transit_time_bkjd"] = (
        np.asarray(result.transit_time) * u.day
    )
    periodogram["depth"] = np.asarray(result.depth)
    periodogram["depth_err"] = np.asarray(result.depth_err)
    periodogram["nominal_depth_snr"] = np.asarray(result.depth_snr)
    periodogram["power"] = power

    periodogram.meta.update({
        "detector": "astropy_bls",
        "objective": "likelihood",
        "time_scale": "tdb",
        "bjd_reference": float(curve.meta["bjd_reference"]),
    })

    with curve_file.open("rb") as stream:
        input_hash = hashlib.file_digest(stream, "sha256").hexdigest()

    candidate = {
        "status": "unvetted_periodic_signal",
        "detector": "astropy_bls",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kic_id": int(curve.meta["kic_id"]),
        "quarter": int(curve.meta["quarter"]),
        "period_days": period,
        "duration_hours": duration * 24,
        "transit_time_bkjd": transit_time,
        "time_scale": "tdb",
        "bjd_reference": float(curve.meta["bjd_reference"]),
        "depth_ppm": float(result.depth[best]) * 1_000_000,
        "depth_err_ppm": float(result.depth_err[best]) * 1_000_000,
        "nominal_depth_snr": float(result.depth_snr[best]),
        "power": float(power[best]),
        "observed_transit_cycles": int(len(cycles)),
        "in_transit_points": int(np.count_nonzero(in_transit)),
        "cycle_numbers": cycles.tolist(),
        "points_per_cycle": cycle_counts.tolist(),
        "input_file": str(curve_file.resolve()),
        "input_sha256": input_hash,
        "astropy_version": version("astropy"),
        "search_settings": {
            "minimum_period_days": minimum_period,
            "maximum_period_days": maximum_period,
            "duration_hours": duration_hours.tolist(),
            "frequency_factor": 1.0,
            "oversample": 20,
            "objective": "likelihood",
            "method": "fast",
            "usable_observations": len(time),
            "baseline_days": baseline,
            "detrending": curve.meta["detrending"],
        },
    }

    return periodogram, candidate