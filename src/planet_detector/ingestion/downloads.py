"""Download archive products and record their provenance."""

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations


def sha256_file(path: Path) -> str:
    """Calculate a file checksum without loading the whole file into memory."""

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_from_search(
    search_file: Path,
    data_dir: Path,
) -> tuple[Path, Path]:
    """Download the single product listed in a saved search table."""

    table = Table.read(search_file, format="ascii.ecsv")

    if len(table) != 1:
        raise ValueError(
            f"Expected exactly one product; the search contains {len(table)}."
        )

    required_columns = {"dataURI", "productFilename"}

    if not required_columns.issubset(table.colnames):
        raise ValueError(
            f"Search table must contain these columns: {required_columns}"
        )

    uri = str(table["dataURI"][0])
    filename = str(table["productFilename"][0])

    if not filename or Path(filename).name != filename:
        raise ValueError("Archive product filename is invalid.")

    expected_kic = int(table.meta["kic_id"])
    expected_quarter = table.meta.get("quarter_filter", "all")

    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    destination = data_dir / filename
    manifest_path = data_dir / f"{filename}.manifest.json"
    temporary_path = data_dir / f"partial-{filename}"

    if destination.exists():
        print(f"Using existing file: {destination}", flush=True)
        candidate_path = destination

        if manifest_path.exists():
            previous = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )

            if sha256_file(destination) != previous["sha256"]:
                raise RuntimeError(
                    "Existing file differs from its recorded checksum."
                )
    else:
        print(f"Downloading: {filename}", flush=True)

        status, message, _ = Observations.download_file(
            uri,
            local_path=str(temporary_path),
            cache=False,
        )

        if status != "COMPLETE":
            raise RuntimeError(f"Download failed: {status}: {message}")

        candidate_path = temporary_path

    # Open read-only and check the FITS structure and target identity.
    with fits.open(candidate_path, mode="readonly", memmap=False) as hdus:
        hdus.verify("exception")

        actual_kic = int(hdus[0].header["KEPLERID"])
        actual_quarter = int(hdus[0].header["QUARTER"])

        if actual_kic != expected_kic:
            raise ValueError("Downloaded file belongs to a different star.")

        if expected_quarter != "all":
            if actual_quarter != int(expected_quarter):
                raise ValueError("Downloaded file belongs to another quarter.")

    if candidate_path != destination:
        candidate_path.replace(destination)

    manifest = {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_uri": uri,
        "product_filename": filename,
        "local_path": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "source_search": str(search_file.resolve()),
        "source_search_sha256": sha256_file(search_file),
        "kic_id": actual_kic,
        "quarter": actual_quarter,
        "software": {
            name: version(name)
            for name in ("planet-detector", "lightkurve", "astroquery", "astropy")
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    return destination, manifest_path