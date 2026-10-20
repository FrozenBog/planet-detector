"""Command-line interface for the transit discovery pipeline."""

import argparse
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="planet-detector",
        description="Kepler and TESS transit discovery pipeline.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('planet-detector')}",
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    commands.add_parser(
        "info",
        help="Show the installed package and Python environment.",
    )

    search_parser = commands.add_parser(
        "search",
        help="Find Kepler target pixel files on MAST.",
    )

    search_parser.add_argument(
        "--kic",
        type=int,
        required=True,
        help="Kepler Input Catalog identifier.",
    )

    search_parser.add_argument(
        "--quarter",
        type=int,
        choices=range(18),
        default=None,
        help="One Kepler quarter, from 0 to 17. Default: all quarters.",
    )

    search_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the search metadata, using .ecsv.",
    )

    search_parser.add_argument(
        "--product",
        choices=["pixels", "lightcurve"],
        default="pixels",
        help="Product type to search for. Default: pixels.",
    )

    download_parser = commands.add_parser(
        "download",
        help="Download the single product in a saved search table.",
    )

    download_parser.add_argument(
        "--search-file",
        type=Path,
        required=True,
        help="Previously saved search metadata in ECSV format.",
    )

    download_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Download directory. Default: data/raw.",
    )

    inspect_parser = commands.add_parser(
        "inspect",
        help="Inspect a local Kepler target pixel file.",
    )

    inspect_parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the downloaded target pixel FITS file.",
    )

    extract_parser = commands.add_parser(
        "extract",
        help="Extract aperture photometry from a Kepler pixel file.",
    )

    extract_parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Input target pixel FITS file.",
    )

    extract_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output light curve in ECSV format.",
    )

    plot_parser = commands.add_parser(
        "plot",
        help="Plot an extracted aperture light curve and its aperture.",
    )

    plot_parser.add_argument(
        "--curve",
        type=Path,
        required=True,
        help="Extracted light curve in ECSV format.",
    )

    plot_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination figure, such as results/overview.png.",
    )

    compare_parser = commands.add_parser(
        "compare-sap",
        help="Compare extracted photometry with Kepler SAP_FLUX.",
    )

    compare_parser.add_argument(
        "--curve",
        type=Path,
        required=True,
    )

    compare_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
    )

    compare_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON comparison report.",
    )

    detrend_parser = commands.add_parser(
        "detrend",
        help="Fit and remove a slow light-curve trend.",
    )

    detrend_parser.add_argument(
        "--curve",
        type=Path,
        required=True,
    )

    detrend_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    detrend_parser.add_argument(
        "--window-length",
        type=int,
        default=301,
        help="Odd number of observations in the smoothing window.",
    )

    detrend_plot_parser = commands.add_parser(
        "plot-detrending",
        help="Plot the fitted trend and detrended measurements.",
    )

    detrend_plot_parser.add_argument(
        "--curve",
        type=Path,
        required=True,
    )

    detrend_plot_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    bls_parser = commands.add_parser(
        "bls",
        help="Search a detrended light curve for periodic transits.",
    )

    bls_parser.add_argument(
        "--curve",
        type=Path,
        required=True,
    )

    bls_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    bls_parser.add_argument(
        "--minimum-period",
        type=float,
        default=0.5,
        help="Minimum trial period in days.",
    )

    bls_parser.add_argument(
        "--maximum-period",
        type=float,
        default=10.0,
        help="Maximum trial period in days.",
    )

    bls_plot_parser = commands.add_parser(
        "plot-bls",
        help="Plot a saved BLS periodogram and folded signal.",
    )

    bls_plot_parser.add_argument(
        "--search-dir",
        type=Path,
        required=True,
    )

    bls_plot_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    repeat_parser = commands.add_parser(
        "check-repeatability",
        help="Compare transit depths across temporal and odd/even partitions.",
    )

    repeat_parser.add_argument(
        "--search-dir",
        type=Path,
        required=True,
    )

    repeat_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.command == "info":
        print(f"Planet Detector: {version('planet-detector')}")
        print(f"Python: {platform.python_version()}")
        print(f"Interpreter: {sys.executable}")
        return 0

    if args.command == "search":
        from planet_detector.ingestion.mast import (
            search_kepler_lightcurves,
            search_kepler_pixels,
        )

        print(
            f"Searching MAST for KIC {args.kic}: {args.product}...",
            flush=True,
        )

        if args.product == "pixels":
            result = search_kepler_pixels(
                kic_id=args.kic,
                quarter=args.quarter,
            )
        else:
            result = search_kepler_lightcurves(
                kic_id=args.kic,
                quarter=args.quarter,
            )

        print(f"\nFound {len(result)} product(s).")

        if len(result) == 0:
            print("No matching products. Search metadata was not saved.")
            return 1

        print(result)

        source_table = result.table

        if source_table is None:
            raise RuntimeError(
                "MAST returned search results without a metadata table."
            )

        table = source_table.copy()
        table.meta["queried_at_utc"] = datetime.now(timezone.utc).isoformat()
        table.meta["kic_id"] = args.kic
        table.meta["product_type"] = args.product
        table.meta["quarter_filter"] = (
            "all" if args.quarter is None else str(args.quarter)
        )
        table.meta["lightkurve_version"] = version("lightkurve")

        args.output.parent.mkdir(parents=True, exist_ok=True)

        table.write(
            args.output,
            format="ascii.ecsv",
            overwrite=False,
        )

        print(f"\nSearch metadata saved to: {args.output.resolve()}")
        return 0

    if args.command == "download":
        from planet_detector.ingestion.downloads import download_from_search

        data_path, manifest_path = download_from_search(
            search_file=args.search_file,
            data_dir=args.data_dir,
        )

        print(f"\nFITS file: {data_path}")
        print(f"Manifest: {manifest_path}")
        return 0

    if args.command == "inspect":
        from planet_detector.ingestion.inspection import inspect_kepler_pixels

        inspect_kepler_pixels(args.file)
        return 0

    if args.command == "extract":
        from planet_detector.photometry.aperture import extract_kepler_aperture

        curve = extract_kepler_aperture(args.file)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        curve.write(
            args.output,
            format="ascii.ecsv",
            overwrite=False,
        )

        print(f"Aperture pixels: {curve.meta['aperture_pixels']}")
        print(f"Total cadences: {len(curve)}")
        print(f"Passing time checks: {int(sum(curve['time_ok']))}")
        print(f"Passing pixel checks: {int(sum(curve['pixel_ok']))}")
        print(f"Passing quality checks: {int(sum(curve['quality_ok']))}")
        print(f"Passing all checks: {int(sum(curve['valid']))}")
        print(f"\nLight curve saved to: {args.output.resolve()}")
        return 0

    if args.command == "plot":
        from planet_detector.reporting.plots import plot_aperture_lightcurve

        plot_aperture_lightcurve(
            curve_file=args.curve,
            output=args.output,
        )

        print(f"Figure saved to: {args.output.resolve()}")
        return 0

    if args.command == "compare-sap":
        import json

        from planet_detector.validation.photometry import compare_kepler_sap

        report = compare_kepler_sap(
            curve_file=args.curve,
            reference_file=args.reference,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        print(f"Shared cadences: {report['shared_cadences']}")
        print(f"Compared cadences: {report['compared_cadences']}")
        print(f"Median flux difference: {report['median_difference_ppm']:.6f} ppm")
        print(f"RMS flux difference: {report['rms_difference_ppm']:.6f} ppm")
        print(f"Maximum flux difference: {report['max_abs_difference_ppm']:.6f} ppm")
        print(
            "Maximum time difference: "
            f"{report['max_abs_time_difference_seconds']:.9f} seconds"
        )

        passed = report["flux_check_passed"] and report["time_check_passed"]

        print(f"\nComparison: {'PASS' if passed else 'INVESTIGATE'}")
        print(f"Report saved to: {args.output.resolve()}")

        return 0 if passed else 1

    if args.command == "detrend":
        from planet_detector.preprocessing.detrending import detrend_lightcurve

        curve = detrend_lightcurve(
            curve_file=args.curve,
            window_length=args.window_length,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        curve.write(
            args.output,
            format="ascii.ecsv",
            overwrite=False,
        )

        settings = curve.meta["detrending"]

        print(f"Original rows retained: {len(curve)}")
        print(f"Usable detrended observations: {int(sum(curve['search_valid']))}")
        print(f"Window length: {settings['window_length']} observations")
        print(f"Nominal window: {settings['nominal_window_days']:.3f} days")
        print(f"\nDetrended light curve saved to: {args.output.resolve()}")
        return 0

    if args.command == "plot-detrending":
        from planet_detector.reporting.plots import plot_detrending

        plot_detrending(
            curve_file=args.curve,
            output=args.output,
        )

        print(f"Figure saved to: {args.output.resolve()}")
        return 0

    if args.command == "bls":
        import json

        from planet_detector.detection.bls import search_bls

        periodogram_path = args.output_dir / "periodogram.ecsv"
        candidate_path = args.output_dir / "strongest_candidate.json"

        if periodogram_path.exists() or candidate_path.exists():
            raise FileExistsError(
                "Search outputs already exist. Choose a new output directory."
            )

        periodogram, candidate = search_bls(
            curve_file=args.curve,
            minimum_period=args.minimum_period,
            maximum_period=args.maximum_period,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)

        periodogram.write(
            periodogram_path,
            format="ascii.ecsv",
            overwrite=False,
        )

        candidate_path.write_text(
            json.dumps(candidate, indent=2) + "\n",
            encoding="utf-8",
        )

        print(f"\nTrial periods: {len(periodogram)}")
        print(f"Best period: {candidate['period_days']:.8f} days")
        print(f"Box duration: {candidate['duration_hours']:.3f} hours")
        print(f"Box depth: {candidate['depth_ppm']:.3f} ppm")
        print(f"Nominal depth SNR: {candidate['nominal_depth_snr']:.2f}")
        print(f"Sampled transit cycles: {candidate['observed_transit_cycles']}")
        print(f"In-transit observations: {candidate['in_transit_points']}")
        print(f"\nResults saved to: {args.output_dir.resolve()}")
        return 0

    if args.command == "plot-bls":
        from planet_detector.reporting.bls_plots import plot_bls_search

        plot_bls_search(
            search_dir=args.search_dir,
            output=args.output,
        )

        print(f"Figure saved to: {args.output.resolve()}")
        return 0

    if args.command == "check-repeatability":
        import json

        from planet_detector.validation.repeatability import check_repeatability

        report = check_repeatability(args.search_dir)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        for name, measurement in report["measurements"].items():
            if measurement["status"] != "ok":
                print(f"{name}: insufficient data")
                continue

            print(
                f"{name}: "
                f"{measurement['depth_ppm']:.2f} +/- "
                f"{measurement['formal_depth_err_ppm']:.2f} ppm; "
                f"{measurement['sampled_transit_cycles']} sampled cycles"
            )

        difference = report.get("odd_even_nominal_difference_sigma")

        if difference is not None:
            print(f"\nOdd/even nominal difference: {difference:.2f} sigma")

        print(f"\nReport saved to: {args.output.resolve()}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())