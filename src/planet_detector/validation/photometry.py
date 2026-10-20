"""Compare independently extracted photometry with mission reference data."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import QTable


def compare_kepler_sap(
    curve_file: Path,
    reference_file: Path,
    flux_tolerance_ppm: float = 10.0,
) -> dict:
    """Compare extracted flux with SAP_FLUX at matching cadence numbers."""

    curve = QTable.read(curve_file, format="ascii.ecsv")

    with fits.open(reference_file, mode="readonly", memmap=False) as hdus:
        primary = hdus[0].header
        lightcurve_hdu = hdus[1]

        if int(primary["KEPLERID"]) != int(curve.meta["kic_id"]):
            raise ValueError("Reference file belongs to another star.")

        if int(primary["QUARTER"]) != int(curve.meta["quarter"]):
            raise ValueError("Reference file belongs to another quarter.")

        if not isinstance(lightcurve_hdu, fits.BinTableHDU):
            raise ValueError("Expected a reference light-curve binary table.")

        data = lightcurve_hdu.data

        if data is None:
            raise ValueError("Reference light-curve table is empty.")

        header = lightcurve_hdu.header

        if str(header["TIMESYS"]).strip().upper() != "TDB":
            raise ValueError("Reference timestamps must use TDB.")

        if curve.meta["time_scale"] != "tdb":
            raise ValueError("Extracted timestamps must use TDB.")

        reference_offset = (
            float(header["BJDREFI"])
            + float(header.get("BJDREFF", 0.0))
        )

        reference_cadence = np.array(
            data["CADENCENO"], dtype=np.int64, copy=True
        )
        reference_time = np.array(data["TIME"], dtype=float, copy=True)
        reference_flux = np.array(data["SAP_FLUX"], dtype=float, copy=True)

    extracted_cadence = np.asarray(curve["cadence"], dtype=np.int64)
    extracted_time = u.Quantity(curve["time_bkjd"]).to_value(u.day)
    extracted_flux = u.Quantity(curve["flux"]).to_value(u.electron / u.s)
    extracted_valid = np.asarray(curve["valid"], dtype=bool)

    if len(np.unique(extracted_cadence)) != len(extracted_cadence):
        raise ValueError("Extracted cadence identifiers contain duplicates.")

    if len(np.unique(reference_cadence)) != len(reference_cadence):
        raise ValueError("Reference cadence identifiers contain duplicates.")

    shared, extracted_index, reference_index = np.intersect1d(
        extracted_cadence,
        reference_cadence,
        return_indices=True,
    )

    ours = extracted_flux[extracted_index]
    theirs = reference_flux[reference_index]
    our_time = extracted_time[extracted_index]
    their_time = reference_time[reference_index]

    use = (
        extracted_valid[extracted_index]
        & np.isfinite(ours)
        & np.isfinite(theirs)
        & (theirs > 0)
        & np.isfinite(our_time)
        & np.isfinite(their_time)
    )

    compared_count = int(np.count_nonzero(use))

    if compared_count < 3:
        raise ValueError("Fewer than three usable matched observations.")

    difference_ppm = (
        (ours[use] - theirs[use]) / theirs[use]
    ) * 1_000_000

    offset_difference = (
        float(curve.meta["bjd_reference"]) - reference_offset
    )

    time_difference_seconds = (
        our_time[use] - their_time[use] + offset_difference
    ) * 86400

    max_flux_difference = float(np.max(np.abs(difference_ppm)))
    max_time_difference = float(np.max(np.abs(time_difference_seconds)))

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kic_id": int(curve.meta["kic_id"]),
        "quarter": int(curve.meta["quarter"]),
        "shared_cadences": int(len(shared)),
        "compared_cadences": compared_count,
        "valid_extracted_cadences": int(np.count_nonzero(extracted_valid)),
        "median_difference_ppm": float(np.median(difference_ppm)),
        "rms_difference_ppm": float(np.sqrt(np.mean(difference_ppm**2))),
        "max_abs_difference_ppm": max_flux_difference,
        "max_abs_time_difference_seconds": max_time_difference,
        "flux_tolerance_ppm": flux_tolerance_ppm,
        "time_tolerance_seconds": 0.1,
        "flux_check_passed": max_flux_difference <= flux_tolerance_ppm,
        "time_check_passed": max_time_difference <= 0.1,
        "files": {},
        "valid_extracted_not_compared": (
            int(np.count_nonzero(extracted_valid)) - compared_count
        ),        
    }

    for label, path in (
        ("extracted", curve_file),
        ("reference", reference_file),
    ):
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()

        report["files"][label] = {
            "path": str(path.resolve()),
            "sha256": checksum,
        }

    return report