# Bayesian Fitting for Seismic Inversion (Python)

Python translation of the MATLAB VBR bayesian_fitting project for estimating
temperature, melt fraction, grain size, and **viscosity** from seismic Vs and Q
observations using Bayesian inference.

All four anelastic methods (`andrade_psp`, `eburgers_psp`, `xfit_mxw`,
`xfit_premelt`) have been verified against the original MATLAB VBR calculator
to floating-point precision across a full parameter sweep.

## Overview

This package provides tools to:
- Load and process seismic velocity (Vs) and attenuation (Q) observations
- Calculate posterior probability distributions for state variables (T, φ, gs, η)
- Estimate **viscosity** (log₁₀ η) with full posterior uncertainty from HK2003 composite rheology
- Combine results across multiple anelastic calculation methods
- Scale from single locations to **23M+ grid point** global models with
  **depth-batched streaming** parallel dispatch and resume support
- Convert results to compressed 3-D **NetCDF** files and plot global
  **Robinson-projection maps** via PyGMT
- Generate publication-quality figures

## Installation

### Environment Setup

Create a conda environment with all required packages (including plotting):

```bash
conda create -n vbrc_v2t -c conda-forge python=3.12 numpy scipy matplotlib pyyaml pandas xarray netcdf4 pygmt
conda activate vbrc_v2t
```

### Install the package

From the parent directory of `vbrc_V2Tpy/`, run:

```bash
pip install -e ./vbrc_V2Tpy
```

This installs the package in **editable mode** (`-e`), meaning:
- `python -m bayesian_fitting_py` works from any directory on your system
- Code changes in `vbrc_V2Tpy/bayesian_fitting_py/` take effect immediately (no reinstall needed)

## Usage

### Basic Usage

```python
from bayesian_fitting_py import run_bayesian_inversion
from bayesian_fitting_py.run_bayes import InversionConfig

# Use default configuration
results = run_bayesian_inversion()

# Or customize
config = InversionConfig(
    gs_prior_type='log_uniform',  # or 'uniform' (alias) or 'log_normal'
    anelastic_methods=['xfit_premelt', 'eburgers_psp'],
    obs_types='VsQ',  # 'Vs', 'Q', or 'VsQ'
    output_dir='./my_output',
)
results = run_bayesian_inversion(config)
```

### Command Line

Run from the `V2T_Inversion` directory (parent of `bayesian_fitting_py`):

```bash
cd /Users/ehightow/Research/V2T_Inversion

# Use both Vs and Q with all anelastic methods (default)
python -m bayesian_fitting_py --gs-prior log_uniform --output-dir plots/

# Use only Vs data
python -m bayesian_fitting_py --obs-types Vs --output-dir plots/

# Use only Q data
python -m bayesian_fitting_py --obs-types Q --output-dir plots/

# Use a single anelastic method
python -m bayesian_fitting_py --anelastic-methods xfit_premelt

# Use multiple specific anelastic methods
python -m bayesian_fitting_py --anelastic-methods xfit_premelt,eburgers_psp

# Use all available anelastic methods
python -m bayesian_fitting_py --anelastic-methods all

# List available anelastic methods
python -m bayesian_fitting_py --list-methods
```

## Location Modes

The package supports three ways to specify locations for inversion:

### 1. Manual Mode (default)

Specify individual locations directly in the configuration:

```yaml
location_mode: manual
locations:
  - [40.7, -117.5]  # [lat, lon]
  - [39.0, -109.8]
names:
  - BasinRange
  - ColoradoPlateau
z_ranges:
  - [75, 105]   # [z_min, z_max] in km
  - [120, 150]
```

Seismic data is loaded from the files specified by `vs_file` and `q_file`.

### 2. Locations File Mode

Load locations from a CSV or text file (seismic data still loaded from
`vs_file`/`q_file`):

```bash
python -m bayesian_fitting_py --location-mode locations_file --location-file locations.csv
```

File format (CSV or whitespace-separated):
```
# lon, lat, name, z_min, z_max
-117.5, 40.7, BasinRange, 75, 105
-109.8, 39.0, ColoradoPlateau, 120, 150
```

The `name` column is optional - if omitted, points are named `point_0`, `point_1`, etc.

### 3. Model Mode

Load both locations AND seismic observations from a single file.
The file format is **auto-detected** from the extension (.csv, .mat, .nc):

```bash
# CSV model
python -m bayesian_fitting_py --location-mode model --vs-file model.csv

# MAT model
python -m bayesian_fitting_py --location-mode model --vs-file model.mat

# NetCDF model (requires xarray)
python -m bayesian_fitting_py --location-mode model --vs-file model.nc

# Separate Vs and Q files
python -m bayesian_fitting_py --location-mode model --vs-file vs_model.csv --q-file q_model.csv
```

CSV format:
```csv
lon,lat,depth,Vs,Q,Vs_error,Q_error
-117.5,40.7,100,4.2,85,0.05,10
-117.5,40.7,150,4.3,90,0.05,10
```

Columns can be named: lon/longitude, lat/latitude, depth/z/z_km, Vs/vs, Q/q,
Vs_error, Q_error. Error columns are optional — defaults are used if missing.

For backward compatibility, the old mode names (`csv_model`, `mat_model`,
`netcdf_model`) and `--seismic-model-file` flag still work but map to `model`
mode internally.

### Options for Model Mode

```bash
# Filter to specific depth range
python -m bayesian_fitting_py --location-mode model --vs-file model.csv --model-z-range 100,200

# Subsample to reduce computation
python -m bayesian_fitting_py --location-mode model --vs-file model.mat --model-subsample 2

# Set default errors if not in file
python -m bayesian_fitting_py --location-mode model --vs-file model.csv --default-vs-error 0.1 --default-q-error 15
```

### Large-Scale Runs

When processing >20 locations:
- Individual plots are automatically suppressed to save time
- Progress is reported every 10 locations
- Results are saved to CSV for easy analysis
- A summary of T and φ ranges is printed instead of full tables

```bash
# Large model with CSV output
python -m bayesian_fitting_py --location-mode csv_model --seismic-model-file model.csv --save-csv --csv-file my_results.csv
```

### Parallel Processing

For large-scale runs (tomography models), use the `--parallel` / `-j` flag to
process locations in parallel using Python multiprocessing:

```bash
# Use 4 worker processes
python -m bayesian_fitting_py --config my_config.yaml --parallel 4

# Auto-detect all available CPU cores
python -m bayesian_fitting_py --config my_config.yaml -j 0

# Or set in the YAML config file:
# parallel_workers: 4
```

Parallel mode uses depth-batched streaming dispatch: locations are grouped by
depth, processed one layer at a time, and results are flushed to split-file
CSVs after each batch.  This bounds peak memory regardless of total model size.

To resume an interrupted run, add `resume: true` to the config.  The code reads
the progress file written by depth-batched streaming to skip completed depths.

| Workers | Time (3168 locs × 4 methods) | Time (23M locs × 1 method) |
|---------|------------------------------|-----------------------------|
| 1 (sequential) | ~22 min | N/A |
| 4 | ~8 min | N/A |
| 16 | ~4 min | N/A |
| 96 (Anvil HPC) | N/A | ~5 hours |

### Using Parameter Files

Instead of specifying all options on the command line, you can use a configuration file:

```bash
# Generate a template configuration file
python -m bayesian_fitting_py --generate-config my_config.yaml

# Run with a configuration file
python -m bayesian_fitting_py --config my_config.yaml

# Override specific settings from the config file
python -m bayesian_fitting_py --config my_config.yaml --anelastic-methods xfit_premelt
```

Example YAML configuration file (`my_config.yaml`):
```yaml
# Location mode: manual, locations_file, model
# (legacy aliases csv_model, mat_model, netcdf_model also accepted)
location_mode: manual

# For locations_file mode:
# location_file: ./locations.csv

# For model mode:
# vs_file: ./model.csv     # or .mat or .nc  (Vs + locations)
#   Can also be a built-in 1D model name: prem, prem_nocrust, stw105, stw105_nocrust
# q_file: ./q_model.csv    # optional separate Q file (or built-in model name)
# model_z_range: [100, 200]        # optional depth filter
# model_subsample: 1               # use every Nth point
# default_vs_error: 0.05           # default if not in file
# default_q_error: 10.0            # default if not in file
# q_error_mode: absolute           # 'absolute' (σ_Q = default_q_error) or
#                                  # 'percent' (σ_Q = default_q_error/100 × Q_obs)

# Locations (for manual mode)
locations:
  - [40.7, -117.5]
  - [39.0, -109.8]
names:
  - BasinRange
  - ColoradoPlateau
z_ranges:
  - [75, 105]
  - [120, 150]
location_colors:
  - [1, 0.6, 0]
  - [0, 0.8, 0]

# Data files
# vs_file / q_file accept file paths (.mat, .csv, .nc) or built-in 1D model
# names: prem, prem_nocrust, stw105, stw105_nocrust
vs_file: ./data/vel_models/Shen_Ritzwoller_2016.mat
q_file: ./data/Q_models/Dalton_Ekstrom_2008.mat
sweep_file: ./data/plate_VBR/sweep_log_gs.mat

# Anelastic methods: andrade_psp, eburgers_psp, xfit_mxw, xfit_premelt
anelastic_methods:
  - xfit_premelt
  - eburgers_psp

# Grain size prior
# gs_prior_type: log_uniform, uniform (alias for log_uniform), or log_normal
# gs_prior_mean_mm: 1.0     # mean grain size in mm (for log_normal)
# gs_prior_std: 1.0         # std dev in log-space (for log_normal)
gs_prior_type: log_uniform

# Melt fraction prior
# phi_prior_type: uniform, zero_melt, or piecewise_depth
# phi_onset_depth_km: 80.0  # for piecewise_depth: depth below which melt is suppressed
phi_prior_type: uniform

# Temperature prior
# t_prior_type: uniform (default, flat) or geotherm (Gaussian centred on a
#   reference geotherm at each depth)
# geotherm_file: sc2006  # built-in name or path to CSV (depth_km, temperature_C)
# geotherm_std_C: 200.0  # Gaussian σ in °C (for geotherm mode)
# t_prior_type: uniform

# Q error mode: how default_q_error is interpreted
# 'absolute' (default): σ_Q = default_q_error
# 'percent': σ_Q = (default_q_error / 100) × Q_obs
# q_error_mode: absolute

# Observations: Vs, Q, or VsQ
obs_types: VsQ

# Output
output_dir: plots/output_plots
save_plots: true
save_ml_csv: false

# Resume: pick up where an interrupted streaming run left off
# resume: true

# Auto-convert ML-estimate CSVs to compressed 3-D NetCDF after streaming
# save_ml_netcdf: false

# Run tagging: controls the inversion output subdirectory name.
#   'none' — plain 'inversion_results' (default, backward compatible)
#   'auto' — auto-generate a compact tag from inversion parameters
#   <str>  — use 'inversion_<str>' as the subdirectory name
# run_tag: none

# Reference Earth model for comparison plots (built-in name or file path)
# reference_model: stw105_nocrust

# Parallelization (for large-scale runs)
# 0 = auto (all cores), 1 = sequential (default), N = N workers
# parallel_workers: 4
```

JSON configuration files are also supported with the same structure.

### Fetching Data

The package requires pre-computed VBR sweep data and seismic observations.
To fetch the required data files (~180 MB):

```bash
# Interactive prompt (asks before downloading)
fetch-vbr-data

# Skip prompt
fetch-vbr-data -y

# Or via python -m
python -m bayesian_fitting_py.fetch_data
```

By default, data is placed in `vbrc_V2Tpy/data/` (alongside the package).
To choose a different location:

```bash
fetch-vbr-data --data-dir /path/to/parent
```

## Data Requirements

The package expects the following data files in `./data/`:

- `vel_models/Shen_Ritzwoller_2016.mat` - Vs model
- `Q_models/Dalton_Ekstrom_2008.mat` - Q model  
- `LAB_models/HopperFischer2018.mat` - LAB depth observations
- `plate_VBR/sweep_log_gs.mat` - Pre-computed VBR parameter sweep

These can be fetched automatically from the vbrPublicData repository.

## Module Structure

```
bayesian_fitting_py/
├── __init__.py          # Package initialization
├── __main__.py          # CLI entry point
├── run_bayes.py         # Main inversion script, CLI, InversionConfig
├── fitting.py           # Fitting functions & ML estimation (incl. viscosity)
├── parallel.py          # Depth-batched streaming parallel dispatch
├── postprocessing.py    # CSV→NetCDF conversion & global map plotting
├── data_processing.py   # Seismic data loading and processing
├── probability.py       # Probability distribution functions
├── prior.py             # Prior probability calculations (incl. geotherm T prior)
├── plotting.py          # Visualization functions
├── orchestration.py     # Reusable sweep/inversion workflow helpers
├── io.py                # Split-file CSV I/O for ML estimates
├── fetch_data.py        # Data fetching utilities
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── vbr/                 # VBR calculator (Python translation)
    ├── __init__.py       # VBR module exports
    ├── core.py           # Main VBR calculation engine (elastic, viscous, anelastic)
    ├── cammarano.py      # Cammarano et al. (2003) finite-strain mineral physics
    ├── params.py         # Parameter classes for all methods
    ├── thermal.py        # Solidus, thermal calcs, Earth model & geotherm I/O
    ├── generate_sweep.py # Parameter sweep generation and I/O (incl. viscosity)
    └── plot_lut.py       # Look-up table plotting & comparison
```

## VBR Calculator (Python Translation)

The package includes a Python translation of the MATLAB VBR (Very Broadband Rheology)
calculator, allowing you to generate parameter sweeps directly in Python without
requiring MATLAB.

### Generating Parameter Sweeps

Parameter sweeps are pre-computed lookup tables used for efficient Bayesian inversion.
The sweep generation can be computationally intensive, so it's typically done once
and saved for repeated use.

#### Using Python API

```python
from bayesian_fitting_py.vbr import VBR, generate_parameter_sweep
from bayesian_fitting_py.vbr.generate_sweep import SweepParams, save_sweep
import numpy as np

# Define sweep parameters
params = SweepParams(
    T=np.arange(1100, 1601, 25),           # Temperature in K
    phi=np.array([0, 0.001, 0.005, 0.01, 0.02, 0.03]),  # Melt fraction
    gs=np.array([0.001, 0.003, 0.01, 0.03]),  # Grain size in m
    depth=np.arange(20, 301, 5),           # Depth in km
    period=np.array([50, 75, 100]),        # Period in seconds
    anelastic_methods=['eburgers_psp', 'xfit_premelt'],
)

# Generate sweep
sweep = generate_parameter_sweep(params, verbose=True)

# Save to .mat file (compatible with existing loading functions)
save_sweep(sweep, 'my_sweep.mat')
```

#### Using Command Line

```bash
# Generate sweep from YAML config file
python -m bayesian_fitting_py.vbr.generate_sweep --config sweep_config.yaml

# Override output file
python -m bayesian_fitting_py.vbr.generate_sweep --config sweep_config.yaml --output my_sweep.mat

# Show progress
python -m bayesian_fitting_py.vbr.generate_sweep --config sweep_config.yaml --verbose
```

#### Example Configuration File (sweep_config.yaml)

```yaml
# Temperature range (K)
T:
  start: 1100
  stop: 1600
  step: 25

# Melt fraction (volume fraction)
phi: [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04]

# Grain size (m)
gs: [0.001, 0.003, 0.01, 0.03]

# Depth range (km)
depth:
  start: 20
  stop: 300
  step: 5

# Period (s)
period: [50, 75, 100]

# Anelastic methods: eburgers_psp, xfit_premelt, andrade_psp, xfit_mxw
anelastic_methods:
  - eburgers_psp
  - xfit_premelt

# Elastic method: 'anharmonic' (linear Taylor expansion, olivine-based) or
# 'cammarano2003' (finite-strain mineral physics, depth-dependent mineralogy)
elastic_method: anharmonic

# Viscous method for stored viscosity:
# 'HK2003' (Hirth & Kohlstedt 2003 composite) or 'xfit_premelt' (Yamauchi & Takei 2016)
viscous_method: HK2003

# Density model: 'constant' (uniform rho), 'prem', 'prem_nocrust',
# 'stw105', 'stw105_nocrust', or 'custom' (requires density_file)
density_model: constant
# density_file: /path/to/custom_model.txt  # for density_model: custom

# Reference Earth model for profile comparison plots
# (built-in name or file path; used by runner scripts, not the core inversion)
# reference_model: stw105_nocrust

# Solidus method: 'hirschmann', 'katz', or 'yk2001'
solidus_method: hirschmann

# xfit_premelt direct melt effect mode:
# 0 = YT2016 (poroelastic via external anh_poro only, default)
# 1 = YT2024 (direct melt effects on anelasticity)
include_direct_melt_effect: 0

# Output file
output_file: sweep_custom.mat
```

### Sweep File Formats

The package supports multiple file formats for parameter sweeps:

| Format | Extension | Description | File Size |
|--------|-----------|-------------|-----------|
| MATLAB | `.mat` | Compatible with MATLAB and original code | Largest |
| NumPy | `.npz` | Compressed NumPy archive, Python-native | ~4x smaller |
| Pickle | `.pkl` | Python pickle format | Similar to .npz |

**Saving in different formats:**
```python
from bayesian_fitting_py.vbr.generate_sweep import save_sweep, load_sweep

# Save to different formats (format determined by extension)
save_sweep(sweep, 'sweep.mat')   # MATLAB format
save_sweep(sweep, 'sweep.npz')   # NumPy format (recommended for Python)
save_sweep(sweep, 'sweep.pkl')   # Pickle format

# Load from any format
sweep = load_sweep('sweep.npz')
```

**Recommended format:** Use `.npz` for Python-only workflows (smaller files, faster loading).
Use `.mat` if you need compatibility with MATLAB or existing code.

### Using the VBR Calculator Directly

You can also use the VBR calculator for single-point calculations:

```python
from bayesian_fitting_py.vbr import VBR
from bayesian_fitting_py.vbr.core import StateVariables
import numpy as np

# Create state variables
sv = StateVariables(
    T_K=1473.0,        # Temperature (K)
    P_GPa=2.0,         # Pressure (GPa)
    rho=3300.0,        # Density (kg/m³)
    dg_um=10000.0,     # Grain size (μm)
    phi=0.01,          # Melt fraction
    sig_MPa=0.1,       # Differential stress (MPa)
    f=np.array([0.01, 0.1]),  # Frequency (Hz)
    Tsolidus_K=1473.0, # Solidus temperature (K)
)

# Create VBR calculator
vbr = VBR(sv)

# Run calculations
results = vbr.calculate(
    elastic_methods=['anharmonic'],
    viscous_methods=['xfit_premelt'],
    anelastic_methods=['eburgers_psp', 'xfit_premelt'],
)

# Access results
print(f"Gu (anharmonic): {results.elastic['anharmonic']['Gu']} GPa")
print(f"Vs (eburgers_psp): {results.anelastic['eburgers_psp']['V']} km/s")
print(f"Q (eburgers_psp): {results.anelastic['eburgers_psp']['Q']}")
print(f"Eta (xfit_premelt): {results.viscous['xfit_premelt']['eta']} Pa·s")
```

### Available Methods

**Elastic Methods:**
- `anharmonic`: Linear Taylor expansion with olivine parameters (Isaak 1992, Cammarano et al., 2003). Suitable for upper mantle.
- `cammarano2003`: Finite-strain mineral physics with depth-dependent mineralogy (Cammarano et al. 2003). Suitable for upper mantle through lower mantle.
- `anh_poro`: Poro-elastic melt effect (poroelastic reduction of shear modulus)

**Anelastic Methods:**
- `eburgers_psp`: Extended Burgers model (Jackson & Faul 2010) — uses FastBurger algorithm
- `andrade_psp`: Andrade model with pseudo-period scaling (Jackson & Faul 2010)
- `xfit_premelt`: Pre-melting scaling (Yamauchi & Takei 2016)
- `xfit_mxw`: Maxwell scaling with relaxation spectrum (McCarthy et al. 2011)

**Viscous Methods:**
- `HK2003`: Hirth & Kohlstedt (2003) composite rheology (diffusion, dislocation, GBS)
- `xfit_premelt`: Near-solidus viscosity (Yamauchi & Takei 2016)

### Solidus Calculations

Configurable via `solidus_method` in the sweep config:
- `hirschmann`: Hirschmann (2000) — default
- `katz`: Katz et al. (2003)
- `yk2001`: Yamazaki & Karato (2001) — uses depth-dependent pressure from earth model

```python
from bayesian_fitting_py.vbr.thermal import solidus, calculate_solidus_K

# Calculate solidus at a given pressure
T_sol = solidus(P_GPa=2.0, method='katz')  # Katz et al. 2003
T_sol = solidus(P_GPa=2.0, method='hirschmann')  # Hirschmann 2000
T_sol = solidus(P_GPa=2.0, method='yk2001')  # Yamazaki & Karato 2001

# With volatile depression
T_sol_wet = calculate_solidus_K(
    P_GPa=2.0,
    solidus_method='katz',
    Cf_H2O=0.01,  # 1 wt% H2O
    Cf_CO2=0.001, # 0.1 wt% CO2
)
```

## Output

The inversion produces:
- Posterior probability plots for each location/method combination
- Regional fits showing T-φ tradeoffs across methods
- Ensemble PDFs combining results across methods
- Pickled results file for further analysis
- **CSV file** with ML estimates for all locations and methods

### CSV Output Columns

When `save_ml_csv: true`, the output CSV contains one row per location × method:

| Column | Description |
|--------|-------------|
| `name`, `lat`, `lon`, `z` | Location identifiers |
| `z_min`, `z_max` | Depth averaging range (km) |
| `anelastic_method` | Method used |
| `T_ml`, `T_std`, `T_mean` | Temperature (K): max-likelihood, std, mean |
| `phi_ml`, `phi_std`, `phi_mean` | Melt fraction: max-likelihood, std, mean |
| `gs_ml_mm`, `gs_std_mm`, `gs_mean_mm` | Grain size (mm): max-likelihood, std, mean |
| `log10_eta_ml`, `log10_eta_std`, `log10_eta_mean` | Viscosity (log₁₀ Pa·s): max-likelihood, std, mean |
| `Vs_obs`, `Vs_pred`, `Vs_misfit`, `Vs_chi2` | Vs fit diagnostics |
| `Q_obs`, `Q_pred`, `Q_misfit`, `Q_chi2` | Q fit diagnostics |
| `chi2_total` | Total χ² misfit |

Viscosity is computed using method-consistent rheology: xfit_premelt uses its
own near-solidus viscosity (Yamauchi & Takei 2016); other methods use HK2003
composite rheology (diffusion + dislocation + GBS). Reported as the full
posterior marginal distribution (not just the ML point estimate).

## Grain Size Priors

Configurable via `gs_prior_type` and related fields:
- `log_uniform`: Uniform probability in log-space (default)
- `uniform`: Alias for `log_uniform` (identical behavior)
- `log_normal`: Log-normal distribution with configurable mean (`gs_prior_mean_mm`
  in mm) and standard deviation (`gs_prior_std` in log-space)

## Melt Fraction Priors

Configurable via `phi_prior_type` and related fields:
- `uniform`: Flat prior over the sweep range (default)
- `zero_melt`: Sharply peaked at φ=0 everywhere (suppresses all melt)
- `piecewise_depth`: Uniform prior above `phi_onset_depth_km`, zero-melt below.
  Use this when melt is physically expected only above a certain depth (e.g.,
  the solidus crossing depth)
- `temperature_dependent`: Placeholder for future T-aware prior (not yet implemented)

Example YAML:
```yaml
phi_prior_type: piecewise_depth
phi_onset_depth_km: 80.0    # melt allowed above 80 km, suppressed below
```

## Temperature Priors

Configurable via `t_prior_type` and related fields:
- `uniform`: Flat prior over the sweep temperature range (default)
- `geotherm`: Gaussian prior centred on a reference geotherm at each depth,
  with standard deviation `geotherm_std_C` (°C).  This constrains the
  inversion toward geophysically reasonable temperatures, particularly
  useful in the lithospheric mantle where Vs deficits can otherwise drive
  unrealistically high temperature estimates.

Built-in geotherms:
- `sc2006`: Steinberger & Calderwood (2006) continental geotherm (0–3000 km)

A custom geotherm can be supplied as a CSV file with columns `depth_km` and
`temperature_C`.  Values are linearly interpolated to the midpoint of each
depth range.

Example YAML:
```yaml
t_prior_type: geotherm
geotherm_file: sc2006        # or path/to/custom_geotherm.csv
geotherm_std_C: 200.0        # Gaussian σ in °C
```

## Original MATLAB Code

This is a Python translation of the MATLAB code in:
`vbr/Projects/bayesian_fitting/`

The original VBR calculator is available at:
https://github.com/vbr-calc/vbr

## Citation

If you use this code, please cite the VBR calculator and relevant papers
describing the anelastic methods used.

## License

Same license as the parent VBR project.
