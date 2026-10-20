# Planet Detector

A modular Python pipeline for extracting Kepler/TESS photometry,
detecting transit candidates, and evaluating their reliability.

## Current implementation

- MAST searches for Kepler pixel and light-curve products
- Downloads with source metadata and SHA-256 manifests
- FITS inspection and mission-aperture photometry
- Cadence quality and missing-data checks
- Comparison with archived SAP_FLUX
- Configurable Savitzky–Golay detrending
- Weighted BLS searches and diagnostic plots
- Temporal and odd/even depth comparisons
- Local end-to-end regression test

## Initial benchmark: Kepler-10, Quarter 1

- Stored cadences: 1,639
- Usable extracted cadences: 1,626
- Cadences compared with SAP_FLUX: 1,624
- Maximum SAP_FLUX difference: 0.107167 ppm

| Detrending window | BLS period (days) | Box depth (ppm) |
|---:|---:|---:|
| 101 | 0.83748617 | 136.87 |
| 301 | 0.83800821 | 134.35 |
| 501 | 0.83761661 | 133.98 |

The detected signal is consistent with an initial recovery of Kepler-10b.
Box-model parameters and nominal significance require further evaluation.

## Environment

The dependency snapshot targets Windows and Python 3.14.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-py314.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

## Commands

```powershell
.\.venv\Scripts\python.exe -m planet_detector.cli --help
```

## Local integration test

The test requires these downloaded files in data/raw:

- kplr011904151-2009166043257_lpd-targ.fits.gz
- kplr011904151-2009166043257_llc.fits

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Next development

- Period and ephemeris refinement
- Transit injection and recovery
- Detrending comparisons with measured signal attenuation
- Physical transit fitting and additional false-positive checks
- Multi-quarter processing, TLS, and ML components

Downloaded data and generated results are excluded from Git.