# Bayesian Fitting for Seismic Inversion (Python)

Python translation of the MATLAB VBR bayesian_fitting project for estimating
temperature, melt fraction, and grain size from seismic Vs and Q observations
using Bayesian inference.

All four anelastic methods (`andrade_psp`, `eburgers_psp`, `xfit_mxw`,
`xfit_premelt`) have been verified against the original MATLAB VBR calculator
to floating-point precision across a full parameter sweep.

## Overview

This package provides tools to:
- Load and process seismic velocity (Vs) and attenuation (Q) observations
- Calculate posterior probability distributions for state variables
- Combine results across multiple anelastic calculation methods
- Generate publication-quality figures
- Scale from single locations to full 3D seismic models

## Installation

### Requirements
- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- PyYAML
- pandas (optional, for CSV model loading)
- xarray (optional, for NetCDF model loading)

### Install the package

From the parent directory of `vbrc_V2Tpy/`, run:

```bash
pip install -e ./vbrc_V2Tpy
```

This installs the package in **editable mode** (`-e`), meaning:
- `python -m bayesian_fitting_py` works from any directory on your system
- Code changes in `vbrc_V2Tpy/bayesian_fitting_py/` take effect immediately (no reinstall needed)
- Required dependencies (NumPy, SciPy, Matplotlib, PyYAML) are installed automatically

To also install optional dependencies for CSV/NetCDF model loading:

```bash
pip install -e "./vbrc_V2Tpy[full]"
```

To uninstall:

```bash
pip uninstall bayesian_fitting_py
```

## Usage

### Basic Usage

```python
from bayesian_fitting_py import run_bayesian_inversion
from bayesian_fitting_py.run_bayes import InversionConfig

# Use default configuration
results = run_bayesian_inversion()

# Or customize
config = InversionConfig(
    gs_prior_case='log_uniform',  # or 'log_normal_1mm', 'log_normal_1cm'
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

The package supports five ways to specify locations for inversion:

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

### 2. Locations File Mode

Load locations from a CSV or text file (seismic data still loaded from separate .mat files):

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

### 3. CSV Model Mode

Load both locations AND seismic observations from a single CSV file:

```bash
python -m bayesian_fitting_py --location-mode csv_model --seismic-model-file model.csv
```

CSV format:
```csv
lon,lat,depth,Vs,Q,Vs_error,Q_error
-117.5,40.7,100,4.2,85,0.05,10
-117.5,40.7,150,4.3,90,0.05,10
```

Columns can be named: lon/longitude, lat/latitude, depth/z/z_km, Vs/vs, Q/q, Vs_error, Q_error. Error columns are optional - defaults are used if missing.

### 4. MAT Model Mode

Load locations and seismic observations from a .mat file, using the exact depths in the file for inversion:

```bash
python -m bayesian_fitting_py --location-mode mat_model --seismic-model-file model.mat
```

The .mat file should contain: Lat, Lon, Depth, and Vs (and optionally Q, Error).

### 5. NetCDF Model Mode

Load locations and seismic observations from a NetCDF file (using xarray):

```bash
python -m bayesian_fitting_py --location-mode netcdf_model --seismic-model-file model.nc
```

The NetCDF should have dimensions for lat, lon, and depth, with Vs (and optionally Q) data variables.

### Options for Model Modes (csv_model, mat_model, netcdf_model)

```bash
# Filter to specific depth range
python -m bayesian_fitting_py --location-mode csv_model --seismic-model-file model.csv --model-z-range 100,200

# Subsample to reduce computation
python -m bayesian_fitting_py --location-mode mat_model --seismic-model-file model.mat --model-subsample 2

# Set default errors if not in file
python -m bayesian_fitting_py --location-mode csv_model --seismic-model-file model.csv --default-vs-error 0.1 --default-q-error 15
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
# Location mode: manual, locations_file, csv_model, mat_model, netcdf_model
location_mode: manual

# For locations_file mode:
# location_file: ./locations.csv

# For csv_model, mat_model, or netcdf_model modes:
# seismic_model_file: ./model.csv  # or .mat or .nc
# model_z_range: [100, 200]        # optional depth filter
# model_subsample: 1               # use every Nth point
# default_vs_error: 0.05           # default if not in file
# default_q_error: 10.0            # default if not in file

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
vs_file: ./data/vel_models/Shen_Ritzwoller_2016.mat
q_file: ./data/Q_models/Dalton_Ekstrom_2008.mat
sweep_file: ./data/plate_VBR/sweep_log_gs.mat

# Anelastic methods: andrade_psp, eburgers_psp, xfit_mxw, xfit_premelt
anelastic_methods:
  - xfit_premelt
  - eburgers_psp

# Grain size prior: log_uniform, log_normal_1mm, log_normal_1cm
gs_prior_case: log_uniform

# Observations: Vs, Q, or VsQ
obs_types: VsQ

# Output
output_dir: plots/output_plots
save_plots: true
save_ml_csv: false
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
├── probability.py       # Probability distribution functions
├── prior.py             # Prior probability calculations
├── data_processing.py   # Seismic data loading and processing
├── fitting.py           # Main fitting functions
├── plotting.py          # Visualization functions
├── run_bayes.py         # Main inversion script
├── fetch_data.py        # Data fetching utilities
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── vbr/                 # VBR calculator (Python translation)
    ├── __init__.py       # VBR module exports
    ├── core.py           # Main VBR calculation engine (elastic, viscous, anelastic)
    ├── params.py         # Parameter classes for all methods
    ├── thermal.py        # Solidus and thermal calculations
    └── generate_sweep.py # Parameter sweep generation and I/O
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
- `anharmonic`: Temperature and pressure scaling (Isaak 1992, Cammarano 2003)
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

```python
from bayesian_fitting_py.vbr.thermal import solidus, calculate_solidus_K

# Calculate solidus at a given pressure
T_sol = solidus(P_GPa=2.0, method='katz')  # Katz et al. 2003
T_sol = solidus(P_GPa=2.0, method='hirschmann')  # Hirschmann 2000

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

## Grain Size Priors

Three options are available:
- `log_uniform`: Uniform probability in log-space (default)
- `log_normal_1mm`: Log-normal centered at 1 mm grain size
- `log_normal_1cm`: Log-normal centered at 1 cm grain size

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
