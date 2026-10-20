"""Check transit depth across temporal and odd/even partitions."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import QTable
from astropy.timeseries import BoxLeastSquares


def _measure_depth(
    flux: np.ndarray,
    error: np.ndarray,
    in_transit: np.ndarray,
    selected: np.ndarray,
) -> dict:
    """Fit independent constant levels inside and outside transit."""

    inside = selected & in_transit
    outside = selected & ~in_transit

    n_inside = int(np.count_nonzero(inside))
    n_outside = int(np.count_nonzero(outside))

    if n_inside < 3 or n_outside < 5:
        return {
            "status": "insufficient_data",
            "in_transit_points": n_inside,
            "out_of_transit_points": n_outside,
        }

    inside_weights = 1 / error[inside] ** 2
    outside_weights = 1 / error[outside] ** 2

    inside_mean = (
        np.sum(inside_weights * flux[inside])
        / np.sum(inside_weights)
    )

    outside_mean = (
        np.sum(outside_weights * flux[outside])
        / np.sum(outside_weights)
    )

    depth = outside_mean - inside_mean

    depth_error = np.sqrt(
        1 / np.sum(inside_weights)
        + 1 / np.sum(outside_weights)
    )

    return {
        "status": "ok",
        "in_transit_points": n_inside,
        "out_of_transit_points": n_outside,
        "depth_ppm": float(depth * 1_000_000),
        "formal_depth_err_ppm": float(depth_error * 1_000_000),
        "nominal_depth_snr": float(depth / depth_error),
    }


def check_repeatability(search_dir: Path) -> dict:
    """Measure depth in partitions using the saved BLS ephemeris."""

    candidate_file = search_dir / "strongest_candidate.json"
    candidate = json.loads(candidate_file.read_text(encoding="utf-8"))

    curve_file = Path(candidate["input_file"])

    with curve_file.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()

    if checksum != candidate["input_sha256"]:
        raise ValueError("Light curve differs from the searched input.")

    curve = QTable.read(curve_file, format="ascii.ecsv")
    valid = np.asarray(curve["search_valid"], dtype=bool)

    time = u.Quantity(curve["time_bkjd"]).to_value(u.day)[valid]
    flux = u.Quantity(
        curve["flat_flux"]
    ).to_value(u.dimensionless_unscaled)[valid]
    error = u.Quantity(
        curve["flat_flux_err"]
    ).to_value(u.dimensionless_unscaled)[valid]

    if len(time) < 20:
        raise ValueError("Too few usable observations.")

    if not np.all(
        np.isfinite(time)
        & np.isfinite(flux)
        & np.isfinite(error)
        & (error > 0)
    ):
        raise ValueError("Search observations contain invalid values.")

    period = float(candidate["period_days"])
    duration = float(candidate["duration_hours"]) / 24
    transit_time = float(candidate["transit_time_bkjd"])

    model = BoxLeastSquares(time, flux, dy=error)

    in_transit = np.asarray(
        model.transit_mask(time, period, duration, transit_time),
        dtype=bool,
    )

    cycles = np.floor(
        (time - transit_time) / period + 0.5
    ).astype(np.int64)

    midpoint = float((np.min(time) + np.max(time)) / 2)

    partitions = {
        "all": np.ones(len(time), dtype=bool),
        "first_half": time < midpoint,
        "second_half": time >= midpoint,
        "even_cycles": cycles % 2 == 0,
        "odd_cycles": cycles % 2 != 0,
    }

    measurements = {}

    for name, selected in partitions.items():
        measurement = _measure_depth(
            flux,
            error,
            in_transit,
            selected,
        )

        measurement["sampled_transit_cycles"] = int(
            len(np.unique(cycles[selected & in_transit]))
        )

        measurements[name] = measurement

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_file": str(candidate_file.resolve()),
        "input_file": str(curve_file.resolve()),
        "input_sha256": checksum,
        "period_days": period,
        "duration_hours": duration * 24,
        "transit_time_bkjd": transit_time,
        "split_time_bkjd": midpoint,
        "method": "fixed_ephemeris_weighted_two_level_fit",
        "uncertainty_model": "independent_measurement_errors",
        "measurements": measurements,
    }

    even = measurements["even_cycles"]
    odd = measurements["odd_cycles"]

    if even["status"] == "ok" and odd["status"] == "ok":
        difference_error = np.hypot(
            even["formal_depth_err_ppm"],
            odd["formal_depth_err_ppm"],
        )

        report["odd_even_nominal_difference_sigma"] = float(
            abs(even["depth_ppm"] - odd["depth_ppm"])
            / difference_error
        )

    return report