"""Inspect Kepler target pixel files without modifying their contents."""

from pathlib import Path

import numpy as np
from astropy.io import fits


def inspect_kepler_pixels(path: Path) -> None:
    """Print file structure, pixel dimensions, timing, and quality information."""

    with fits.open(path, mode="readonly", memmap=False) as hdus:
        hdus.verify("exception")

        print("\n--- FITS structure ---")
        hdus.info()

        primary_header = hdus[0].header
        pixel_hdu = hdus[1]

        if not isinstance(pixel_hdu, fits.BinTableHDU):
            raise ValueError("Expected a binary table in FITS extension 1.")

        data = pixel_hdu.data

        if data is None:
            raise ValueError("The pixel table contains no data.")

        required_columns = {"TIME", "FLUX", "FLUX_ERR", "QUALITY"}

        if not required_columns.issubset(pixel_hdu.columns.names):
            raise ValueError("The file is missing required pixel columns.")

        header = pixel_hdu.header

        time = np.asarray(data["TIME"], dtype=float)
        flux = np.asarray(data["FLUX"], dtype=float)
        flux_error = np.asarray(data["FLUX_ERR"], dtype=float)
        quality = np.asarray(data["QUALITY"], dtype=np.int64)

        if flux.ndim != 3:
            raise ValueError("Expected FLUX dimensions: cadence, row, column.")

        if flux_error.shape != flux.shape:
            raise ValueError("FLUX and FLUX_ERR dimensions do not match.")

        if not (len(time) == len(quality) == flux.shape[0]):
            raise ValueError("Time, quality, and pixel cadence counts differ.")

        finite_time = np.isfinite(time)

        if np.count_nonzero(finite_time) < 2:
            raise ValueError("Fewer than two finite timestamps are available.")

        valid_times = time[finite_time]
        time_steps = np.diff(valid_times)

        if np.any(time_steps <= 0):
            raise ValueError("Finite timestamps are not strictly increasing.")

        reference = (
            float(header["BJDREFI"])
            + float(header.get("BJDREFF", 0.0))
        )

        print("\n--- Target and pixels ---")
        print(f"KIC ID: {primary_header['KEPLERID']}")
        print(f"Quarter: {primary_header['QUARTER']}")
        print(f"FLUX shape (cadences, rows, columns): {flux.shape}")
        print(f"FLUX unit: {pixel_hdu.columns['FLUX'].unit}")
        print(f"Finite pixel values: {np.isfinite(flux).mean():.2%}")

        print("\n--- Time ---")
        print(f"Time scale: {header.get('TIMESYS', 'not specified')}")
        print(f"Time unit: {header.get('TIMEUNIT', 'not specified')}")
        print(f"BJD reference offset: {reference:.9f}")
        print(f"First finite TIME: {valid_times[0]:.9f}")
        print(f"Last finite TIME: {valid_times[-1]:.9f}")
        print(f"Observation span: {np.ptp(valid_times):.5f} days")
        print(f"Median time spacing: {np.median(time_steps) * 1440:.5f} minutes")
        print(f"Nonfinite timestamps: {np.count_nonzero(~finite_time)}")

        print("\n--- Quality and calibration ---")
        print(f"Background already subtracted (BACKAPP): {header.get('BACKAPP')}")
        print(f"Cadences with QUALITY = 0: {np.count_nonzero(quality == 0)}")
        print(f"Cadences with QUALITY != 0: {np.count_nonzero(quality != 0)}")

        print("\nQuality flag values and cadence counts:")
        values, counts = np.unique(quality, return_counts=True)

        for value, count in zip(values, counts):
            print(f"  {int(value):>8}: {int(count)}")

        print("\nAvailable pixel-table columns:")
        print(", ".join(pixel_hdu.columns.names))