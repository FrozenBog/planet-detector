"""Initial Savitzky-Golay detrending baseline."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import QTable
from astropy.time import Time
from lightkurve import LightCurve


def detrend_lightcurve(
    curve_file: Path,
    window_length: int = 301,
) -> QTable:
    """Add fitted trend and flattened flux without deleting original rows."""

    if window_length < 5 or window_length % 2 == 0:
        raise ValueError("Window length must be an odd integer of at least 5.")

    curve = QTable.read(curve_file, format="ascii.ecsv")

    if "flat_flux" in curve.colnames:
        raise ValueError("Use the original aperture light curve as input.")

    if curve.meta["time_scale"] != "tdb":
        raise ValueError("Expected the TDB time scale.")

    valid = np.asarray(curve["valid"], dtype=bool)
    time = u.Quantity(curve["time_bkjd"]).to_value(u.day)
    flux = u.Quantity(curve["flux"]).to_value(u.electron / u.s)
    error = u.Quantity(curve["flux_err"]).to_value(u.electron / u.s)

    if np.count_nonzero(valid) < window_length:
        raise ValueError("Too few valid observations for this window length.")

    if not np.all(
        np.isfinite(time[valid])
        & np.isfinite(flux[valid])
        & np.isfinite(error[valid])
        & (error[valid] > 0)
    ):
        raise ValueError("An observation marked valid contains invalid data.")

    time_steps = np.diff(time[valid])

    if np.any(time_steps <= 0):
        raise ValueError("Valid timestamps must be strictly increasing.")

    median_flux = float(np.median(flux[valid]))

    if median_flux <= 0:
        raise ValueError("Expected positive median stellar flux.")

    times = Time(
        time[valid],
        float(curve.meta["bjd_reference"]),
        format="jd",
        scale="tdb",
    )

    lightcurve = LightCurve(
        time=times,
        flux=flux[valid] / median_flux,
        flux_err=error[valid] / median_flux,
    )

    result = lightcurve.flatten(
        window_length=window_length,
        polyorder=2,
        return_trend=True,
        break_tolerance=5,
        niters=3,
        sigma=3,
    )

    if not isinstance(result, tuple):
        raise RuntimeError("Detrending did not return the requested trend.")

    _, trend_lightcurve = result

    fitted_trend = (
        u.Quantity(trend_lightcurve.flux).to_value(u.dimensionless_unscaled)
        * median_flux
    )

    if len(fitted_trend) != np.count_nonzero(valid):
        raise RuntimeError("Detrending changed the observation count.")

    trend = np.full(len(curve), np.nan)
    trend[valid] = fitted_trend

    search_valid = valid & np.isfinite(trend) & (trend > 0)

    flat_flux = np.full(len(curve), np.nan)
    flat_error = np.full(len(curve), np.nan)

    flat_flux[search_valid] = flux[search_valid] / trend[search_valid]
    flat_error[search_valid] = error[search_valid] / trend[search_valid]

    if not np.any(search_valid):
        raise ValueError("Detrending produced no usable observations.")

    curve["trend_flux"] = trend * u.electron / u.s
    curve["flat_flux"] = flat_flux * u.dimensionless_unscaled
    curve["flat_flux_err"] = flat_error * u.dimensionless_unscaled
    curve["search_valid"] = search_valid

    with curve_file.open("rb") as stream:
        input_hash = hashlib.file_digest(stream, "sha256").hexdigest()

    curve.meta["detrending"] = {
        "method": "lightkurve_savgol",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(curve_file.resolve()),
        "input_sha256": input_hash,
        "window_length": window_length,
        "nominal_window_days": float(window_length * np.median(time_steps)),
        "polyorder": 2,
        "break_tolerance": 5,
        "niters": 3,
        "sigma": 3,
        "uncertainty_model": "measurement_error_divided_by_fitted_trend",
        "includes_trend_uncertainty": False,
    }

    return curve