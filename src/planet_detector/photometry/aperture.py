"""Aperture photometry for calibrated Kepler target pixel files."""

import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import QTable
from lightkurve.utils import KeplerQualityFlags


def extract_kepler_aperture(path: Path) -> QTable:
    """Sum the mission aperture and retain all cadences with validity flags."""

    with fits.open(path, mode="readonly", memmap=False) as hdus:
        primary_header = hdus[0].header.copy()
        pixel_hdu = hdus[1]
        aperture_hdu = hdus[2]

        if not isinstance(pixel_hdu, fits.BinTableHDU):
            raise ValueError("Expected a pixel binary table in extension 1.")

        if not isinstance(aperture_hdu, fits.ImageHDU):
            raise ValueError("Expected an aperture image in extension 2.")

        data = pixel_hdu.data
        aperture_image = aperture_hdu.data

        if data is None or aperture_image is None:
            raise ValueError("Pixel data or aperture image is missing.")

        header = pixel_hdu.header.copy()

        if header.get("BACKAPP") is not True:
            raise ValueError(
                "This extractor requires background-subtracted FLUX."
            )

        if str(header["TIMESYS"]).strip().upper() != "TDB":
            raise ValueError("Expected timestamps in the TDB time scale.")

        reference = (
            float(header["BJDREFI"])
            + float(header.get("BJDREFF", 0.0))
        )

        if reference != 2454833.0:
            raise ValueError("Unexpected reference offset for BKJD times.")

        time = np.asarray(data["TIME"], dtype=float)
        flux_cube = np.asarray(data["FLUX"], dtype=float)
        error_cube = np.asarray(data["FLUX_ERR"], dtype=float)
        quality = np.asarray(data["QUALITY"], dtype=np.int64)
        cadence = np.asarray(data["CADENCENO"], dtype=np.int64)

        # Value 2 identifies membership in the mission's optimal aperture.
        aperture = (
            np.asarray(aperture_image, dtype=np.int64) & 2
        ) != 0

        if flux_cube.ndim != 3:
            raise ValueError("Expected a three-dimensional pixel cube.")

        if aperture.shape != flux_cube.shape[1:]:
            raise ValueError("Aperture and pixel image dimensions differ.")

        if error_cube.shape != flux_cube.shape:
            raise ValueError("Flux and uncertainty cube dimensions differ.")

        if not np.any(aperture):
            raise ValueError("The mission aperture contains no pixels.")

        aperture_flux = flux_cube[:, aperture]
        aperture_error = error_cube[:, aperture]

        pixel_ok = np.all(
            np.isfinite(aperture_flux)
            & np.isfinite(aperture_error)
            & (aperture_error > 0),
            axis=1,
        )

        # Ordinary sums deliberately propagate missing pixels to missing totals.
        flux = np.sum(aperture_flux, axis=1)
        flux_error = np.sqrt(np.sum(aperture_error**2, axis=1))

        # Reject incomplete aperture measurements rather than change the aperture.
        flux[~pixel_ok] = np.nan
        flux_error[~pixel_ok] = np.nan

        quality_bitmask = int(KeplerQualityFlags.DEFAULT_BITMASK)
        quality_ok = (quality & quality_bitmask) == 0
        time_ok = np.isfinite(time)
        valid = time_ok & pixel_ok & quality_ok

        if not np.any(valid):
            raise ValueError("No cadences passed the extraction checks.")

    curve = QTable()
    curve["cadence"] = cadence
    curve["time_bkjd"] = time * u.day
    curve["flux"] = flux * u.electron / u.s
    curve["flux_err"] = flux_error * u.electron / u.s
    curve["quality"] = quality
    curve["time_ok"] = time_ok
    curve["pixel_ok"] = pixel_ok
    curve["quality_ok"] = quality_ok
    curve["valid"] = valid

    with path.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()

    curve.meta.update({
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kic_id": int(primary_header["KEPLERID"]),
        "quarter": int(primary_header["QUARTER"]),
        "source_file": str(path.resolve()),
        "source_sha256": source_hash,
        "time_scale": "tdb",
        "bjd_reference": reference,
        "aperture_method": "mission_pipeline",
        "aperture_mask": aperture.astype(int).tolist(),
        "aperture_pixels": int(np.count_nonzero(aperture)),
        "quality_bitmask": quality_bitmask,
        "background_treatment": "already_subtracted_by_mission",
        "uncertainty_model": "quadrature_without_pixel_covariance",
        "planet_detector_version": version("planet-detector"),
        "lightkurve_version": version("lightkurve"),
    })

    return curve