"""Local integration test using the downloaded Kepler-10 reference files."""

import tempfile
import unittest
from pathlib import Path

from planet_detector.detection.bls import search_bls
from planet_detector.photometry.aperture import extract_kepler_aperture
from planet_detector.preprocessing.detrending import detrend_lightcurve
from planet_detector.validation.photometry import compare_kepler_sap


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIXEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kplr011904151-2009166043257_lpd-targ.fits.gz"
)

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kplr011904151-2009166043257_llc.fits"
)

# NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/overview/Kepler-10
REFERENCE_PERIOD_DAYS = 0.83749070


class Kepler10PipelineTest(unittest.TestCase):
    def test_pixel_extraction_and_known_planet_recovery(self):
        self.assertTrue(
            PIXEL_FILE.is_file(),
            "Download the Kepler-10 Quarter 1 target pixel file first.",
        )

        self.assertTrue(
            REFERENCE_FILE.is_file(),
            "Download the Kepler-10 Quarter 1 reference light curve first.",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            work = Path(temporary_directory)
            aperture_path = work / "aperture.ecsv"
            detrended_path = work / "detrended.ecsv"

            # Recompute from pixels so changes to extraction are actually tested.
            aperture_curve = extract_kepler_aperture(PIXEL_FILE)
            aperture_curve.write(aperture_path, format="ascii.ecsv")

            comparison = compare_kepler_sap(
                aperture_path,
                REFERENCE_FILE,
            )

            self.assertLess(
                comparison["max_abs_difference_ppm"],
                1.0,
                "Pixel sums no longer reproduce mission SAP_FLUX.",
            )

            self.assertTrue(
                comparison["time_check_passed"],
                "Extracted and reference timestamps do not agree.",
            )

            self.assertGreaterEqual(
                comparison["compared_cadences"],
                0.99 * comparison["valid_extracted_cadences"],
                "Too many valid observations were excluded from comparison.",
            )

            detrended_curve = detrend_lightcurve(
                aperture_path,
                window_length=301,
            )

            self.assertEqual(
                len(detrended_curve),
                len(aperture_curve),
                "Detrending must preserve the original rows.",
            )

            detrended_curve.write(detrended_path, format="ascii.ecsv")

            _, candidate = search_bls(detrended_path)

            relative_period_error = abs(
                candidate["period_days"] / REFERENCE_PERIOD_DAYS - 1
            )

            self.assertLess(
                relative_period_error,
                0.001,
                "Known-planet period recovery differs by more than 0.1%.",
            )

            self.assertGreaterEqual(
                candidate["observed_transit_cycles"],
                30,
                "Recovered signal has unexpectedly few sampled cycles.",
            )