"""Search the MAST archive for mission data products."""

import lightkurve as lk


def search_kepler_pixels(
    kic_id: int,
    quarter: int | None = None,
) -> lk.SearchResult:
    """Find Kepler long-cadence target pixel files for one star."""

    if kic_id <= 0:
        raise ValueError("KIC ID must be positive.")

    if quarter is not None and not 0 <= quarter <= 17:
        raise ValueError("Kepler quarter must be between 0 and 17.")

    return lk.search_targetpixelfile(
        target=f"KIC {kic_id}",  # ty: ignore[unknown-argument]
        mission="Kepler",
        author="Kepler",
        exptime="long",
        quarter=quarter,
    )

def search_kepler_lightcurves(
    kic_id: int,
    quarter: int | None = None,
) -> lk.SearchResult:
    """Find official Kepler long-cadence light-curve files."""

    if kic_id <= 0:
        raise ValueError("KIC ID must be positive.")

    if quarter is not None and not 0 <= quarter <= 17:
        raise ValueError("Kepler quarter must be between 0 and 17.")

    return lk.search_lightcurve(
        f"KIC {kic_id}",
        mission="Kepler",
        author="Kepler",
        exptime="long",
        quarter=quarter,
    )