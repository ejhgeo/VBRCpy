# vbrc_V2Tpy — Bayesian Seismic Inversion (Python)

Python translation of the MATLAB [VBR Calculator](https://github.com/vbr-calc/vbr)
bayesian_fitting project for estimating temperature, melt fraction, grain size,
and **viscosity** from seismic Vs and Q observations using Bayesian inference.

All four anelastic methods (`andrade_psp`, `eburgers_psp`, `xfit_mxw`,
`xfit_premelt`) have been verified against the original MATLAB VBR calculator
to floating-point precision across a full parameter sweep.

Designed to scale from single locations to global tomography models with
built-in **multiprocessing support** for large-scale runs.

## Quick Start

### 1. Install

```bash
pip install -e ./vbrc_V2Tpy
```

This installs the package in **editable mode** so code changes take effect
immediately without reinstalling.  Required dependencies (NumPy, SciPy,
Matplotlib, PyYAML) are installed automatically.

For optional CSV / NetCDF model loading:

```bash
pip install -e "./vbrc_V2Tpy[full]"
```

### 2. Fetch Data

The inversion requires pre-computed VBR sweep data and seismic observations
(~180 MB).  After installing, run:

```bash
fetch-vbr-data          # interactive prompt
# or
python -m bayesian_fitting_py.fetch_data -y   # skip prompt
```

Data is placed in `vbrc_V2Tpy/data/` by default.  Use `--data-dir` to choose
a different location.

### 3. Run

```bash
# Default inversion (Vs + Q, all methods, log-uniform grain-size prior)
python -m bayesian_fitting_py

# Use a YAML configuration file
python -m bayesian_fitting_py --config config_example_bayesian_fitting.yaml

# Parallel processing for large-scale runs (0 = auto-detect all cores)
python -m bayesian_fitting_py --config my_config.yaml --parallel 4
python -m bayesian_fitting_py --config my_config.yaml -j 0

# List available anelastic methods
python -m bayesian_fitting_py --list-methods
```

## Features

- **Bayesian inversion** of Vs and/or Q at arbitrary locations and depth ranges
- **Viscosity estimation**: full posterior for log₁₀(η) using HK2003 composite rheology
- **Four anelastic methods**: andrade_psp, eburgers_psp, xfit_premelt, xfit_mxw
- **Parallel processing**: multiprocessing support for large-scale tomography runs
- **Multiple input modes**: manual locations, CSV, MAT, NetCDF seismic models
- **Pure-Python VBR calculator** — generate parameter sweeps without MATLAB
- **Publication-quality plots** with posterior PDFs, T–φ trade-offs, and ensemble summaries

## Project Layout

```
vbrc_V2Tpy/
├── pyproject.toml                  # Package metadata & dependencies
├── data/                           # Fetched data files (git-ignored)
├── config_example_*.yaml           # Example configuration files
└── bayesian_fitting_py/            # Python package
    ├── run_bayes.py                # Main inversion driver & CLI
    ├── fitting.py                  # Fitting functions & ML estimation
    ├── parallel.py                 # Multiprocessing support for large-scale runs
    ├── data_processing.py          # Seismic data I/O (CSV, MAT, NetCDF)
    ├── probability.py / prior.py   # Probability & prior functions
    ├── plotting.py                 # Visualisation
    ├── fetch_data.py               # Data download utility
    └── vbr/                        # Python VBR calculator
        ├── core.py                 # Elastic, viscous, anelastic calculations
        ├── params.py               # Method parameter defaults
        ├── thermal.py              # Solidus & thermal models
        └── generate_sweep.py       # Parameter sweep generation (incl. viscosity)
```

## Documentation

See [bayesian_fitting_py/README.md](bayesian_fitting_py/README.md) for full
documentation including:

- Detailed usage and CLI options
- All five location modes (manual, locations_file, csv_model, mat_model, netcdf_model)
- YAML / JSON configuration reference
- VBR calculator API and sweep generation
- Available methods and solidus calculations

## Uninstall

```bash
pip uninstall bayesian_fitting_py
```

## Citation

If you use this code, please cite the VBR calculator and relevant papers
describing the anelastic methods used.

## License

Same license as the parent [VBR project](https://github.com/vbr-calc/vbr).
