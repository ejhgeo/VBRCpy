# vbrc_V2Tpy — Bayesian Seismic Inversion (Python)

Python translation of the MATLAB [VBR Calculator](https://github.com/vbr-calc/vbr)
bayesian_fitting project for estimating temperature, melt fraction, grain size,
and **viscosity** from seismic Vs and Q observations using Bayesian inference.

All four anelastic methods (`andrade_psp`, `eburgers_psp`, `xfit_mxw`,
`xfit_premelt`) have been verified against the original MATLAB VBR calculator
to machine precision (0.000000% difference) across a full parameter sweep
of temperature, grain size, melt fraction, and depth.

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
- **YT2016 / YT2024 melt mode**: configurable direct melt effects on anelasticity
  via `include_direct_melt_effect` in sweep config
- **Two elastic backends**: `anharmonic` (linear Taylor, olivine) and
  `cammarano2003` (finite-strain mineral physics with depth-dependent mineralogy)
- **Parallel processing**: multiprocessing support for large-scale tomography runs
- **Multiple input modes**: manual locations, CSV, MAT, NetCDF seismic models
- **Pure-Python VBR calculator** — generate parameter sweeps without MATLAB
- **MATLAB benchmark validation** — automated comparison against original VBRc output
- **Publication-quality plots** with posterior PDFs, T–φ trade-offs, and ensemble summaries

## Project Layout

```
vbrc_V2Tpy/
├── pyproject.toml                  # Package metadata & dependencies
├── data/                           # Fetched data files (git-ignored)
├── config_example_*.yaml           # Example configuration files
├── validation/                     # Validation & testing framework
│   ├── syntheticTest_geotherm/     # Geotherm-based validation (SC2006 continental)
│   └── benchmarkTest_vsMatlab/     # Python vs MATLAB VBRc benchmark
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
        ├── plot_lut.py             # Look-up table plotting & comparison
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

## Validation

```bash
cd /Users/ehightow/Research/V2T_Inversion

# Benchmark: compare Python vs MATLAB VBRc (sweep comparison, LUT plots, inversion)
python -m vbrc_V2Tpy.validation.benchmarkTest_vsMatlab

# Synthetic geotherm test (SC2006 continental geotherm, configurable grain size)
python -m vbrc_V2Tpy.validation.syntheticTest_geotherm
python -m vbrc_V2Tpy.validation.syntheticTest_geotherm --gs-um 800

# Regenerate LUT diagnostic plots from existing sweep (no re-computation)
python -m vbrc_V2Tpy.validation.syntheticTest_geotherm --replot-lut
```

The synthetic geotherm test uses an independent VBR core computation (not grid
lookup) to generate observations from a realistic continental geotherm (SC2006),
avoiding the "inverse crime" of inverting on the same grid used to generate the
data.  It includes a cold lithospheric lid, thermal boundary layer, and
convecting interior.  See
[validation/syntheticTest_geotherm/README.md](validation/syntheticTest_geotherm/README.md)
for details.

All validation output is written to `validation_tests/` in the workspace
root (outside the git repository).

## Uninstall

```bash
pip uninstall bayesian_fitting_py
```

## Citation

If you use this code, please cite the VBR calculator and relevant papers
describing the anelastic methods used.

## License

Same license as the parent [VBR project](https://github.com/vbr-calc/vbr).
