# VBRCpy — Bayesian Seismic Inversion (Python)

Python translation of the MATLAB [VBR Calculator](https://github.com/vbr-calc/vbr)
bayesian_fitting project for estimating temperature, melt fraction, grain size,
and **viscosity** from seismic Vs and Q observations using Bayesian inference.

All four anelastic methods (`andrade_psp`, `eburgers_psp`, `xfit_mxw`,
`xfit_premelt`) have been verified against the original MATLAB VBR calculator
to machine precision (0.000000% difference) across a full parameter sweep
of temperature, grain size, melt fraction, and depth.

Designed to scale from single locations to global tomography models with
built-in **multiprocessing support** for large-scale runs.  Depth-batched
streaming dispatch with resume capability enables inversion of 23M+ grid
points on HPC clusters with bounded memory.

## Quick Start

### 1. Environment Setup

Create a conda environment with all required packages (including plotting):

```bash
conda create -n vbrcpy -c conda-forge python=3.12 numpy scipy matplotlib pyyaml pandas xarray netcdf4 pygmt
conda activate vbrcpy
```

### 2. Install

```bash
cd VBRCpy
pip install -e .
```

> **Important:** The `-e` (editable) flag is **required**, not optional.
> Reference data files and downloaded datasets live outside the Python
> package tree (in `data/`), so a regular `pip install` will not work.
> The package checks for this at startup and will raise an error if
> installed non-editably.

### 3. Fetch Data

The inversion requires pre-computed VBR sweep data and seismic observations
(~180 MB).  After installing, run:

```bash
fetch-vbr-data          # interactive prompt
# or
python -m vbrcpy.fetch_data -y   # skip prompt
```

Data is placed in `VBRCpy/data/` by default.  Use `--data-dir` to choose
a different location.

### 4. Run

```bash
# Default inversion (Vs + Q, all methods, log-uniform grain-size prior)
python -m vbrcpy

# Use a YAML configuration file
python -m vbrcpy --config config_example_bayesian_fitting.yaml

# Parallel processing for large-scale runs (0 = auto-detect all cores)
python -m vbrcpy --config my_config.yaml --parallel 4
python -m vbrcpy --config my_config.yaml -j 0

# List available anelastic methods
python -m vbrcpy --list-methods
```

### 5. Post-Processing

After a large-scale run, convert results to NetCDF and generate maps:

```bash
# Convert split-file CSVs to compressed 3-D NetCDF files
vbrc-to-netcdf --csv path/to/ml_estimates/

# Plot global Robinson-projection maps at selected depths
vbrc-plot-maps --depth 100 200 400 --vars T_mean log10_eta_mean phi_mean gs_mean_mm
```

## Features

- **Bayesian inversion** of Vs and/or Q at arbitrary locations and depth ranges
- **Viscosity estimation**: full posterior for log₁₀(η) using method-consistent
  rheology (xfit_premelt uses its own viscosity; other methods use HK2003)
- **Four anelastic methods**: andrade_psp, eburgers_psp, xfit_premelt, xfit_mxw
- **YT2016 / YT2024 melt mode**: configurable direct melt effects on anelasticity
  via `include_direct_melt_effect` in sweep config
- **Two elastic backends**: `anharmonic` (linear Taylor, olivine) and
  `cammarano2003` (finite-strain mineral physics with depth-dependent mineralogy)
- **Geotherm-based temperature prior**: Gaussian prior centered on a reference
  geotherm (e.g. Steinberger & Calderwood 2006 continental) with configurable σ, or flat uniform prior
- **Built-in 1D Earth models**: `prem`, `prem_nocrust`, `stw105`, `stw105_nocrust`
  usable as `vs_file`, `q_file`, `reference_model`, or `density_model`
- **Run tagging**: `run_tag` organizes multiple inversion runs (different priors,
  obs types, etc.) under the same sweep directory without overwriting results
- **Depth-batched streaming**: processes one depth layer at a time, flushing
  results to split-file CSVs after each batch — bounds peak memory regardless
  of total model size (tested on 23M-point global model, 128 cores, ~5 min)
- **Resume support**: `resume: true` config option resumes interrupted runs by
  reading the progress file and skipping completed depths
- **Post-processing pipeline**: CSV → NetCDF conversion (`csv_to_netcdf`) and
  global Robinson-projection map plotting (`plot_global_maps`) via PyGMT
- **CLI entry points**: `vbrc-to-netcdf` and `vbrc-plot-maps` for post-processing
- **Parallel processing**: multiprocessing support with shared-memory Pool
  initializer pattern (eliminates per-task pickle overhead)
- **Flexible input**: manual locations, or `model` mode with auto-detected format
  (.csv, .mat, .nc) via `vs_file` / `q_file`
- **Pure-Python VBR calculator** — generate parameter sweeps without MATLAB
- **MATLAB benchmark validation** — automated comparison against original VBRc output
- **Auto-convert to NetCDF**: `save_ml_netcdf: true` in config automatically
  converts CSVs to compressed 3-D NetCDF after streaming completes
- **Publication-quality plots** with posterior PDFs, T–φ trade-offs, ensemble summaries,
  and PyGMT map-view post-processing (Robinson projection for global models)

## Project Layout

```
VBRCpy/
├── pyproject.toml                  # Package metadata & dependencies
├── data/                           # Fetched data files (git-ignored)
│   └── reference_models/           # Bundled 1D Earth models & geotherms
├── config_example_*.yaml           # Example configuration files
├── validation/                     # Validation & testing framework
│   ├── syntheticTest_geotherm/     # Geotherm-based validation (SC2006 continental)
│   └── benchmarkTest_vsMatlab/     # Python vs MATLAB VBRc benchmark
└── vbrcpy/            # Python package
    ├── run_bayes.py                # Main inversion driver & CLI
    ├── fitting.py                  # Fitting functions & ML estimation
    ├── parallel.py                 # Depth-batched streaming parallel dispatch
    ├── postprocessing.py           # CSV→NetCDF conversion & global map plotting
    ├── data_processing.py          # Seismic data I/O (CSV, MAT, NetCDF)
    ├── prior.py                    # Prior probability (incl. geotherm T prior)
    ├── probability.py              # Likelihood & posterior calculations
    ├── plotting.py                 # Visualisation
    ├── orchestration.py            # Reusable sweep/inversion workflow helpers
    ├── io.py                       # Split-file CSV I/O for ML estimates
    ├── fetch_data.py               # Data download utility
    └── vbr/                        # Python VBR calculator
        ├── core.py                 # Elastic, viscous, anelastic calculations
        ├── cammarano.py            # Finite-strain mineral physics (Cammarano et al., 2003)
        ├── params.py               # Method parameter defaults
        ├── thermal.py              # Solidus, thermal models, Earth model I/O
        ├── plot_lut.py             # Look-up table plotting & comparison
        └── generate_sweep.py       # Parameter sweep generation (incl. viscosity)
```

## Documentation

See [vbrcpy/README.md](vbrcpy/README.md) for full
documentation including:

- Detailed usage and CLI options
- Location modes (`manual`, `locations_file`, `model`) and input formats
- YAML configuration reference
- VBR calculator API and sweep generation
- Available methods and solidus calculations

## Validation

From the repository root:

```bash
# Benchmark: compare Python vs MATLAB VBRc (sweep comparison, LUT plots, inversion)
python -m VBRCpy.validation.benchmarkTest_vsMatlab

# Synthetic geotherm test (Steinberger & Calderwood 2006 geotherm, configurable grain size)
python -m VBRCpy.validation.syntheticTest_geotherm
python -m VBRCpy.validation.syntheticTest_geotherm --gs-um 800

# Regenerate LUT diagnostic plots from existing sweep (no re-computation)
python -m VBRCpy.validation.syntheticTest_geotherm --replot-lut
```

The synthetic geotherm test uses an independent VBR core computation (not grid
lookup) to generate observations from a realistic continental geotherm (Steinberger & Calderwood, 2006),
avoiding the "inverse crime" of inverting on the same grid used to generate the
data.  It includes a cold lithospheric lid, thermal boundary layer, and
convecting interior.  See
[validation/syntheticTest_geotherm/README.md](validation/syntheticTest_geotherm/README.md)
for details.

All validation output is written to `validation_tests/` in the workspace
root (outside the git repository).

## Uninstall

```bash
pip uninstall vbrcpy
```

## Citation

If you use this software, please cite it using the metadata in
[CITATION.cff](CITATION.cff).  Please also cite the VBR calculator and
relevant papers describing the anelastic methods used.

## License

MIT License. See [LICENSE](LICENSE) for details.
