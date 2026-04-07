# vbrc_V2Tpy — Project State Document

> **Last updated:** 2026-04-06
> **Repo:** https://github.com/ejhgeo/vbrc_V2Tpy.git
> **Latest commit:** `ee91550` — "Add geotherm T prior, run_tag feature, built-in 1D Earth models, viscous_method config, PREM/STW105 validation cases"
> **Uncommitted changes:** Yes — see §6 "Recent Uncommitted Changes"

Use this document to bootstrap a new AI chat session on this project.
Paste it as context and say "Continue working on this project" or ask a specific question.

---

## 1. What This Project Is

A **Python port of the MATLAB VBR Calculator** (Havlin et al., 2021) focused on
Bayesian inversion of seismic Vs (and optionally Q) into mantle state
variables: **temperature (T), melt fraction (φ), grain size (gs), and
viscosity (η)**. Now supports upper mantle through transition zone depths
via Cammarano et al. (2003) finite-strain mineral physics.

The original MATLAB code lives in the same workspace at
`/Users/ehightow/Research/V2T_Inversion/vbr/` (upstream VBRc). The Python
port is a standalone pip-installable package at
`/Users/ehightow/Research/V2T_Inversion/vbrc_V2Tpy/`.

## 2. Project Structure

```
vbrc_V2Tpy/
├── README.md
├── setup.py / pyproject.toml          # pip-installable
├── PROJECT_STATE.md                   # this file
├── config_example_bayesian_fitting.yaml
├── config_example_regenerate_sweep.yaml
├── data/                              # bundled .mat data files
│   ├── vel_models/Shen_Ritzwoller_2016.mat
│   ├── Q_models/Dalton_Ekstrom_2008.mat
│   ├── LAB_models/HopperFischer2018.mat
│   ├── plate_VBR/sweep_log_gs.mat
│   └── reference_models/             # bundled 1D Earth models & geotherms
│       ├── PREM_for_VBRc.txt
│       ├── PREMnoCrust_for_VBRc.txt
│       ├── STW105_for_VBRc.txt
│       ├── STW105noCrust_for_VBRc.txt
│       └── SC2006_geotherm.csv
├── validation/                        # validation & testing framework
│   ├── __init__.py
│   ├── syntheticTest_geotherm/  # geotherm-based validation (SC2006 continental geotherm)
│   │   ├── README.md           # usage, assumptions, output description
│   │   ├── __init__.py / __main__.py
│   │   ├── run_example.py     # 4-step orchestrator with --gs-um and --replot-lut
│   │   ├── config.yaml        # combined sweep + inversion config (single file)
│   │   └── SC2006_geotherm.csv  # prescribed geotherm (120 pts, 0–3000 km)
│   └── benchmarkTest_vsMatlab/  # Python vs MATLAB VBRc benchmark
│       ├── __init__.py / __main__.py
│       ├── run_benchmark.py   # 4-step orchestrator (sweep→compare→LUT plots→inversion)
│       └── config.yaml        # combined sweep + inversion config (single file)
└── bayesian_fitting_py/               # main package
    ├── __init__.py / __main__.py
    ├── run_bayes.py          # CLI entry point + InversionConfig + main loop
    ├── fitting.py            # fit_seismic_observations, fit_preloaded_observations, extract_ml_estimates
    ├── data_processing.py    # Location, SeismicModelData, loaders (CSV/mat/NetCDF)
    ├── prior.py              # prior_model_probs, store_ensemble, confidence_cutoffs, TemperaturePrior
    ├── probability.py        # probability_distributions (likelihood, posterior, combined)
    ├── parallel.py           # multiprocessing support for large-scale runs
    ├── plotting.py           # all figure generation
    ├── orchestration.py      # reusable sweep/inversion workflow helpers
    ├── io.py                 # split-file CSV I/O for ML estimates
    ├── fetch_data.py         # interactive data downloader
    └── vbr/                  # VBR core calculations (Python port of MATLAB VBRc)
        ├── core.py           # anelastic methods: andrade_psp, eburgers_psp, xfit_mxw, xfit_premelt
        ├── cammarano.py      # Cammarano et al. (2003) finite-strain mineral physics
        ├── generate_sweep.py # parameter sweep generation (Box with meanVs, meanQ, meanEta)
        ├── params.py         # default VBR parameters
        ├── thermal.py        # half-space cooling, adiabatic geotherm, Earth model I/O, geotherm I/O
        ├── plot_lut.py       # look-up table plotting (.mat, .npz, .pkl)
        ├── PREM_for_VBRc.csv # PREM density profile for depth-dependent density
        ├── STW105_for_VBRc.txt  # STW105 reference model
        └── STW105noCrust_for_VBRc.txt  # STW105 without crust
```

### Patagonia Test Case (workspace root, not in the package)

```
Patagonia_Test/
├── config.yaml            # full sweep + inversion config
├── run_patagonia.py       # 4-step orchestrator (sweep → inversion → profile plots)
├── plot_map_results.py    # PyGMT-based map-view plotting of inversion results
├── 3_D_model_WashU.nc     # 3D Vs model (NetCDF)
└── output/
    ├── sweep.npz              # shared parameter sweep
    ├── sweep_fingerprint.json # config hash (sweep_generation section only)
    ├── lut_plots/             # look-up table diagnostic plots
    ├── inversion_results/     # default inversion output (run_tag: none)
    └── inversion_<tag>/       # auto/custom-tagged inversion output (run_tag: auto|<str>)
        ├── ml_estimates.csv
        ├── *_ensembles.pkl
        ├── patagonia_profiles_*.png
        └── posteriors/
```

## 3. Python Environment

- **Python:** 3.12 (via conda env `pyGMT2`)
- **venv:** `/Users/ehightow/Research/V2T_Inversion/.venv/` (system Python 3.9.6)
- **conda env (primary):** `pyGMT2` — Python 3.12, includes xarray, PyGMT 0.15, netCDF4
- **Key deps:** numpy, scipy, matplotlib, h5py (for .mat), xarray + netCDF4 (for NetCDF),
  PyGMT (for map plots), pandas
- **Install:** `pip install -e .` from `vbrc_V2Tpy/`

## 4. How to Run

```bash
cd /Users/ehightow/Research/V2T_Inversion

# Sequential (default)
python -m vbrc_V2Tpy.bayesian_fitting_py --config test_config.yaml

# Parallel (4 workers)
python -m vbrc_V2Tpy.bayesian_fitting_py --config test_config.yaml --parallel 4

# Parallel (auto — all 16 cores)
python -m vbrc_V2Tpy.bayesian_fitting_py --config test_config.yaml -j 0
```

### Validation Commands

```bash
cd /Users/ehightow/Research/V2T_Inversion

# Synthetic geotherm test — SC2006 continental geotherm
python -m vbrc_V2Tpy.validation.syntheticTest_geotherm
python -m vbrc_V2Tpy.validation.syntheticTest_geotherm --gs-um 800

# Benchmark: Python vs MATLAB VBRc (sweep comparison + LUT plots + inversion)
python -m vbrc_V2Tpy.validation.benchmarkTest_vsMatlab
```

### Config files in the workspace

| File | Description |
|------|-------------|
| `test_config.yaml` | Full WashU tomography NetCDF, 4 methods, Vs only, model_z_range [90,105] |
| `test_eta_config.yaml` | 2 manual locations, 2 methods, for quick viscosity testing |
| `test_parallel_config.yaml` | NetCDF model subsampled 100×, 1 method, for parallel testing |
| `sweep_config.yaml` | Config for regenerating the parameter sweep |
| `validation/syntheticTest_geotherm/config.yaml` | Combined sweep + inversion config for geotherm validation (single file) |
| `validation/benchmarkTest_vsMatlab/config.yaml` | Combined sweep + inversion config for MATLAB benchmark (single file) |

### Sweep file

The pre-computed parameter sweep is `./sweep.npz` (NumPy format). It contains
the 4D grids of `meanVs`, `meanQ`, `meanEta` for each anelastic method across
the `(T, phi, gs, z)` parameter space. Regenerate with:
```bash
python -m bayesian_fitting_py.vbr.generate_sweep --config sweep_config.yaml
```

## 5. Anelastic Methods

All four methods verified against MATLAB output to floating-point precision:

| Method | Description |
|--------|-------------|
| `andrade_psp` | Andrade pseudo-period scaling |
| `eburgers_psp` | Extended Burgers pseudo-period scaling |
| `xfit_mxw` | Maxwell fit (xfit) |
| `xfit_premelt` | Pre-melt fit (xfit) |

## 6. Key Features Implemented

### Recent Uncommitted Changes (2026-04-06)

#### Run Tag Feature (`run_tag` in InversionConfig)
- **New `run_tag` config field** (default `'none'`): controls the inversion
  output subdirectory name, allowing multiple inversion runs with different
  parameters to share the same sweep data.
- Three modes:
  - `run_tag: none` — output goes to `{output_dir}/inversion_results/` (default,
    preserves backward compatibility)
  - `run_tag: auto` — auto-generates a compact, human-readable tag from inversion
    parameters, e.g. `inversion_eburgers_psp_Tgeo100_gsLU_phiU_VsQ_qe100`
  - `run_tag: my_experiment` — uses `inversion_my_experiment/` as the subdirectory
- **`_auto_run_tag()` method**: builds tag from anelastic_methods, t_prior_type +
  geotherm_std_C, gs_prior_type + params, phi_prior_type, obs_types, default_q_error
- **`resolve_inversion_dir()` method**: returns the appropriate subdirectory path;
  only affects the inversion output, leaving sweep.npz and LUT plots in the parent
  `output_dir`
- **`--run-tag` CLI argument**: overrides config from command line
- All inversion outputs (posteriors, CSV, pickle, summary plots) routed through
  `inversion_dir` from `resolve_inversion_dir()`

#### Geotherm-Based Temperature Prior
- **New `t_prior_type` config option**: `'uniform'` (default, flat prior) or
  `'geotherm'` (Gaussian centered on a reference geotherm at each depth)
- **`geotherm_file`**: built-in name (`'sc2006'` for Steinberger & Calderwood
  2006 continental geotherm) or path to a CSV with `depth_km, temperature_C` columns
- **`geotherm_std_C`**: standard deviation (°C) of the Gaussian prior (default 200.0)
- **`TemperaturePrior` dataclass** in `prior.py`: encapsulates prior configuration;
  `apply_temperature_prior()` computes the prior at each depth
- **Parallel support**: `parallel.py` pre-computes geotherm means at each unique
  depth via `load_geotherm()` and builds per-depth priors for efficient distribution
  across workers

#### Built-in 1D Earth Models
- **New built-in model names**: `prem`, `prem_nocrust`, `stw105`, `stw105_nocrust`
  — usable as `vs_file`, `q_file`, `reference_model`, or `density_model` values
- **`_BUILTIN_MODELS` dict** in `thermal.py` maps names to bundled text files in
  `data/reference_models/`: PREM_for_VBRc.txt, PREMnoCrust_for_VBRc.txt,
  STW105_for_VBRc.txt, STW105noCrust_for_VBRc.txt
- **`load_vs_from_earth_model()` / `load_q_from_earth_model()`**: load Vs/Q
  profiles from builtin or custom Earth model files
- **STW105 discontinuity fix**: epsilon-offset approach handles duplicate depths
  at discontinuities (identical to PREM fix)

#### Configurable Viscous Method (`viscous_method`)
- **`viscous_method` sweep parameter**: `'HK2003'` (Hirth & Kohlstedt 2003
  composite rheology, default) or `'xfit_premelt'` (Yamauchi & Takei 2016)
- Controls which viscosity is stored in `meanEta` during sweep generation;
  previously always used HK2003

#### Patagonia 3D Test Case (`Patagonia_Test/`)
- **`run_patagonia.py`**: Full 4-step orchestrator for a real-data Patagonia
  inversion — sweep generation → Bayesian inversion → profile comparison plots
- **`plot_map_results.py`**: PyGMT-based post-processing script for map-view
  plots of inversion results at multiple depths
  - Uses manual `fig.shift_origin()` layout (PyGMT 0.15 subplot API incompatible)
  - Configurable row/column spacing, depth annotations (rotated 90°), variable
    labels, and colour bars only on the bottom row
  - `_find_default_csv()` auto-discovers `ml_estimates.csv` in `inversion_*`
    subdirectories (run_tag-aware)
- **`config.yaml`**: combined sweep + inversion config for Patagonia
  - Uses `cammarano2003` elastic method, `stw105_nocrust` density model and
    reference model, `yk2001` solidus, `YT2024` direct melt effects
  - Geotherm prior (SC2006) with 100°C std, uniform grain-size and melt priors
  - WashU 3D Vs model (NetCDF) + STW105 Q model
- **Profile comparison plots**: 2×3 figures showing T, Vs, Q, φ, gs, η mean
  profiles with reference model overlay; now save to the run-tagged inversion
  directory (via `save_dir` parameter)

#### Reference Model Support in Profile Plots
- **`reference_model` config option**: specifies a built-in 1D Earth model for
  comparison on profile plots (Vs and Q panels)
- Used in `run_patagonia.py` to overlay STW105 Vs/Q against laterally-averaged
  inversion results

### Previous Uncommitted Changes (2026-03-25)

#### Unified Config Files & Single `output_dir`
- **Config consolidation:** Merged separate `sweep_config.yaml` and
  `inversion_config.yaml` into a single `config.yaml` for both validation cases
  (geotherm and benchmark). Sweep parameters are nested under a
  `sweep_generation:` section; inversion parameters remain at the top level.
- **Single `output_dir`:** One top-level `output_dir` key in the combined YAML
  controls all output paths. The runner scripts (`run_example.py`,
  `run_benchmark.py`) derive sweep, LUT, synthetic obs, and inversion output
  paths automatically from this single directory.
- Old `sweep_config.yaml` and `inversion_config.yaml` still exist but are
  no longer used; can be deleted.

#### `InversionConfig` Unknown-Key Filtering
- `InversionConfig.from_dict()` now filters unknown keys before constructing
  the dataclass, using `fields as _dc_fields` from `dataclasses`. This allows
  the combined config (which includes the `sweep_generation` section) to be
  loaded by `run_bayes.py` without errors.

#### New CLI Arguments for `run_bayes.py`
- **`--sweep-file`**: Override the sweep file path from the command line.
  Used by runner scripts to point inversion at the correct sweep.npz.
- **`--vs-file` / `--q-file`**: Override seismic data file paths from CLI.
  Used by geotherm runner to pass synthetic observation CSV to the inversion.

#### Unified Data Input (`vs_file` / `q_file`)
- Replaced `seismic_model_file` and separate `vel_model`/`q_model` fields with
  two universal fields: `vs_file` and `q_file`. These accept any supported
  format (.mat, .csv, .nc) or built-in model names, auto-detected by extension.
- Backward compatible: old configs with `seismic_model_file` are automatically
  mapped to `vs_file`/`q_file` in `from_dict()`.
- Three old `location_mode` values (`csv_model`, `mat_model`, `netcdf_model`)
  consolidated into a single `model` mode with format auto-detection.

#### PREM Density Discontinuity Fix (`thermal.py`)
- **Fixed:** `_load_earth_model()` now handles duplicate depths at
  discontinuities (e.g., 410/410 km in PREM) using an epsilon-offset approach
  (100 m shift for later duplicates) instead of silently averaging the two
  density values. This preserves the density jump across discontinuities for
  correct lithostatic pressure integration.

#### Runner Script Improvements
- Sweep generation is now called **programmatically** (imported
  `generate_parameter_sweep()` and `save_sweep()`) instead of spawning a
  subprocess. Eliminates shell overhead and simplifies error handling.
- **Config fingerprinting** changed from hashing the entire config file to
  hashing only the `sweep_generation` section (via `json.dumps(section,
  sort_keys=True)` → SHA-256). This means changes to inversion-only
  parameters no longer force unnecessary sweep regeneration.
- Inversion is still run as a subprocess, passing `--config`, `--sweep-file`,
  `--output-dir`, and (for geotherm) `--vs-file`, `--q-file`, `--parallel`.

#### Bug Fixes (2026-03-25)
- **`obs` UnboundLocalError:** In `run_example.py`, `obs = _cfg.get('obs_types',
  'VsQ')` was scoped inside an `else` block; moved to correct indentation so
  it's always defined.
- **`--sweep-file` not applied for benchmark:** The `args.sweep_file` CLI
  override in `run_bayes.py` was incorrectly nested inside `if args.parallel
  is not None:`, so it only took effect when `--parallel` was also passed.
  De-indented to top level so it applies unconditionally.

#### Benchmark Verification (2026-03-25)
- After all fixes, benchmark inversion results verified against the old
  reference (`gsLogUniform_ensembles.pkl`). **All primary inversion variables
  (T, φ, gs, p_joint posteriors) are identical within floating-point
  tolerance (~1e-13).** The only differences are in `xfit_premelt` `log10_eta`
  and `predicted_eta`, which are expected due to the viscosity model consistency
  fix (xfit_premelt now uses its own viscosity instead of HK2003).

### Previous Uncommitted Changes (2026-03-24)

#### Synthetic Geotherm Validation Test (`syntheticTest_geotherm`)
- **New validation case:** `validation/syntheticTest_geotherm/` uses the SC2006
  continental geotherm (Steinberger & Calderwood, 2006) instead of a simple
  adiabat, providing realistic thermal structure with a cold lithospheric lid,
  thermal boundary layer, and convecting interior
- Same 4-step orchestrator pattern as syntheticTest_adiabat
  (sweep → synthetic obs → inversion → comparison)
- **`--gs-um` CLI argument**: configurable true grain size in microns
  (default 1000 = 1 mm), threaded through the synthetic observation generation
- **2×3 comparison figure**: T, Vs, Q (row 1) and φ, gs, η (row 2) with
  both MAP and marginal-mean curves
- **Error statistics in all legends**: RMSE, MAE, and mean %error computed
  for each variable; displayed directly in plot legends
- **Q error threshold**: depths where true Q > 1500 (cold lithosphere) are
  excluded from Q error statistics to avoid blown-up RMSE; exclusion count
  shown in panel title
- **Solidus on temperature panel**: red dashed line showing the configured
  solidus (yk2001) for reference alongside the geotherm and recovered profiles
- **Percentage-based Q error model**: `q_error_mode: percent` with
  `default_q_error: 12%` so shallow Q → ∞ observations don't dominate the
  likelihood
- Color coding: φ = C3 (red), gs = C2 (green) for better visual distinction
- `README.md` included in the test directory documenting usage and assumptions

#### Viscosity Model Consistency Fix (`generate_sweep.py`)
- **Fixed:** Sweep previously always stored HK2003 `eta_total` in `meanEta`,
  even when xfit_premelt was the anelastic method. The forward model used
  xfit_premelt viscosity, causing a systematic divergence growing with depth
  (different activation energies: 462.5 vs 375 kJ/mol, different activation
  volumes: 7.913 vs 10 ×10⁻⁶ m³/mol, and xfit_premelt's A_n near-solidus
  scaling factor)
- Now stores method-consistent viscosity:
  `vbr.output['viscous']['xfit_premelt']['diff']['eta']` when
  xfit_premelt is the selected anelastic method; HK2003 as fallback
- **Requires sweep regeneration** (delete existing sweep.npz) to take effect

#### Q Error Mode Config Option
- New `q_error_mode` config option in inversion YAML: `'absolute'` (default,
  backward compatible) or `'percent'`
- When `percent`, Q error = `default_q_error / 100 * Q_obs`, so error scales
  with the observed Q value
- Implemented in `run_bayes.py`, `data_processing.py`, and `parallel.py`

#### Generalized Grain-Size Prior
- Replaced hardcoded `gs_prior_case` with generalized config fields:
  `gs_prior_type` (`log_uniform` or `log_normal`), `gs_prior_mean_mm`,
  `gs_prior_std`
- Updated all configs, CLI parsers, and prior computation code

### Previous Uncommitted Changes (2026-03-19)

#### `include_direct_melt_effect` Config Option (YT2016 vs YT2024)
- `generate_sweep.py`: New `SweepParams.include_direct_melt_effect` field (0=YT2016, 1=YT2024)
- YT2016 (default): anelastic J1/J2 have no explicit φ dependence; only
  poroelastic elastic correction (`anh_poro`) provides melt sensitivity
- YT2024: adds direct melt effects on anelasticity via `Beta_B`, `Beta_P`,
  `poro_Lambda` terms in xfit_premelt
- Exposed via YAML (`include_direct_melt_effect: 1` in sweep config)
- Logged during sweep generation (`xfit_premelt melt mode: YT2024`)

#### `output_dir` Config for Sweep Generation
- `SweepParams.output_dir`: when set, `output_file` and `plot_lut_dir` are
  automatically derived as `{output_dir}/sweep.npz` and `{output_dir}/lut_plots/`
- Eliminates needing to set three paths manually; only `output_dir` required
- `save_sweep()` now calls `os.makedirs()` to auto-create the output directory

#### Inversion Config Improvements
- `ml_csv_file` defaults to `None` and auto-derives as `{output_dir}/ml_estimates.csv`
- `plot_every_n` option: controls how many posterior plots are generated
  (e.g., `plot_every_n: 13` plots every 13th location)
- Posterior plots now generated correctly in **parallel mode** (previously
  skipped because plot code ran before results were collected)
- `output_dir` in `run_example.py` now read from the inversion YAML so the
  user only sets it once

#### `--replot-lut` CLI Flag for Synthetic Adiabat Test
- `run_example.py --replot-lut` reloads the existing sweep and regenerates
  LUT diagnostic plots without regenerating the sweep or re-running inversion
- Uses `generate_sweep_lut_plots()` from `plot_lut.py`

#### LUT Plot Improvements (`plot_lut.py`)
- `plot_lut_at_depth()`: new `Q_log` parameter for logarithmic Q colour scale
- `Q_clim` parameter for clamping Q colorbar range (default adaptive)
- `generate_sweep_lut_plots()`: convenience function to batch-plot LUT slices
  at multiple depths from a sweep, with `every_n` depth subsampling

#### Comparison Figure — Q Panel and Mean-Predicted Curves
- `_make_comparison_plots()` in `run_example.py` now adds a **Q panel** (panel 3)
  when Q was used in the inversion, showing synthetic Q vs predicted Q
- Both **Vs and Q panels** show dashed "Predicted (Mean)" curves computed by
  looking up Vs/Q in the sweep at the nearest grid point to the marginal mean
  T, φ, gs values (no changes to core `fitting.py` or `run_bayes.py` needed)
- Figure width scales dynamically with number of panels (`4.5 * n_panels`)

### Validation Framework (committed `91e6ef5`)
- **`validation/` directory** with modular test cases and a standalone example workflow
- ~~`validate_roundtrip.py`~~, ~~`validate_prem.py`~~, ~~`run_all.py`~~,
  ~~`MAP_vs_Mean_explanation.md`~~: **moved to workspace root** (outdated tests,
  no longer part of the repo)

### ~~Synthetic Adiabat Test~~ (moved to workspace root)
- Previously `validation/syntheticTest_adiabat/`; **moved to workspace root**
  (`/Users/ehightow/Research/V2T_Inversion/syntheticTest_adiabat/`) — not
  included in the repo. Superseded by the geotherm test.

### MATLAB Benchmark Validation — benchmarkTest_vsMatlab (committed `91e6ef5`, fixed `8a93465`)
- **`validation/benchmarkTest_vsMatlab/`**: Automated comparison of the Python
  VBRc against the original MATLAB VBRc output (`vbr/test.mat`)
- **All 4 methods match MATLAB to machine precision (0.000000% diff)** across
  the full parameter grid (T × gs × z) for both Vs and Q
- 4-step orchestrator (`run_benchmark.py`):
  1. Generate parameter sweep with settings matching original MATLAB defaults
  2. Point-by-point numerical comparison against `vbr/test.mat` (prints
     per-method Vs/Q differences and melt-effect analysis)
  3. LUT comparison plots (`compare_lut_slices_T_gs`, `compare_lut_slices_gs_phi`,
     `compare_lut_slices_T_phi`) at multiple depths for all 4 methods
  4. Bayesian inversion at 3 manual locations (Basin & Range, Colorado
     Plateau, Interior) with all 4 anelastic methods
- Full-grid summary: reports median and max |%diff| for Vs and Q with
  (T, gs, z) coordinates of the worst-case point
- Sweep fingerprint caching (SHA-256 hash of config YAML)
- Output to `validation_results/benchmarkTest_vsMatlab/` (outside git repo)
- Invocation: `python -m vbrc_V2Tpy.validation.benchmarkTest_vsMatlab`

### force_plots Config Option (committed `91e6ef5`)
- `force_plots: true` in inversion YAML (or `--force-plots` CLI flag) forces
  posterior plot generation even for large-scale runs that would normally skip plots
- Non-interactive prompt fix: when running as a subprocess (stdin not a TTY),
  auto-confirms without blocking on `input()` — fixes `EOFError` in pipeline usage

### Posterior Plotting Improvements (committed `91e6ef5`)
- Depth included in posterior plot titles and filenames (`posterior_{z}km_{obs}_{method}.png`)
- Consistent depth labeling across both validation pathways (roundtrip and workflow)

### LUT Auto-Generation During Sweep (committed `91e6ef5`)
- `plot_lut` config section in sweep YAML: `enabled`, `output_dir`, `every_n`
- Automatically generates look-up table diagnostic plots during sweep generation
- Controlled via `SweepParams.plot_lut`, `plot_lut_dir`, `plot_lut_every_n`

### Configurable Solidus Method (committed `91e6ef5`)
- `solidus_method` field in `SweepParams` and sweep YAML config
- Options: `hirschmann` (default), `katz`, `yk2001`
- YK2001 solidus handling fixed for correct pressure/temperature behavior

### Cammarano et al. (2003) Finite-Strain Elastic Method (committed `9008504`)
- **New file `cammarano.py`**: Full mineral physics module implementing Appendix A
  of Cammarano et al. (2003) PEPI
- Mineral database from Table A.1: olivine, wadsleyite, ringwoodite, cpx, opx,
  garnet (Py-Mj-Alm), Ca-garnet, Mg-perovskite, Ca-perovskite, Mg-wüstite
- 3rd-order Birch-Murnaghan finite-strain EOS with Newton solver for Eulerian strain
- Voigt-Reuss-Hill (VRH) averaging for composite mineralogy
- Automatic assemblage switching based on pressure boundaries:
  - Upper mantle (<14 GPa / ~410 km): 60% olivine, 12% opx, 14% cpx, 14% garnet
  - Wadsleyite TZ (14–18 GPa / ~410–520 km): 57% wadsleyite, 28% majorite, 5% cpx, 10% Ca-gt
  - Ringwoodite TZ (18–23 GPa / ~520–660 km): 57% ringwoodite, 28% majorite, 8% Ca-gt, 7% Ca-pv
  - Lower mantle (>23 GPa / ~660 km): 75% Mg-pv, 18% Mg-wüstite, 7% Ca-pv
- **Note:** Mineral assemblage volume fractions are approximate, assembled from
  general knowledge of Ringwood (1975), Ita & Stixrude (1992). Not directly
  copied from a single published table. User should verify against preferred source.
- `core.py` updated: `_el_cammarano2003()` method, `_get_base_elastic_output()` and
  `_get_unrelaxed_Gu()` helpers so all 4 anelastic methods work with either
  `anharmonic` or `cammarano2003` elastic backend
- `params.py` updated: `'cammarano2003'` added to `possible_methods` with X_Fe,
  composition, pressure boundary config; also fixed missing default scaling keys
  for `anharmonic` (temperature_scaling, pressure_scaling, reference_scaling)
- `generate_sweep.py` updated: `elastic_method` field in `SweepParams`, YAML
  parsing (`elastic.method`), CLI (`--elastic-method cammarano2003`)
- Config: `elastic.method: cammarano2003` in YAML triggers this path

### Depth-Dependent Density (committed `9008504`)
- `generate_sweep.py`: Added `load_density_profile()` function with PREM default
  and custom CSV option
- Uses cumulative trapezoid integration for lithostatic pressure instead of
  `P = ρ * g * z` with constant ρ
- PREM gives density 3367–3378 kg/m³ in 50–150 km range, ~2–3% higher pressures
  vs constant 3300
- New `SweepParams` fields: `density_model` ('constant', 'prem', 'custom'),
  `density_file` (path to CSV)
- Bundled `PREM_for_VBRc.csv` in package directory; `pyproject.toml` updated
  for `*.csv` inclusion
- YAML config: `density_model: prem`, CLI: `--density-model prem`

### Depth Range Guard (committed `9008504`)
- `data_processing.py` and `fitting.py`: `extract_calculated_values_in_depth_range`
  now raises `ValueError` with clear message when observation depths don't overlap
  sweep depths (previously returned NaN silently)

### plot_lut.py Multi-Format Support (committed `91e6ef5`)
- `_load_sweep_file()` helper auto-detects `.mat`, `.npz`, `.pkl` formats
- CLI accepts any of these formats as input

### Viscosity Output (commit `21cb561`)
- `generate_sweep.py` stores `meanEta` in the Box — method-consistent viscosity:
  xfit_premelt's own diffusion creep η when that method is selected, HK2003
  `eta_total` as fallback for other methods
- `extract_ml_estimates` computes the full posterior marginal PDF for log₁₀(η)
  using binning, returning `{ml, mean, std}` in log₁₀ Pa·s
- CSV output includes `log10_eta_ml`, `log10_eta_std`, `log10_eta_mean`
- Older sweeps without `meanEta` load gracefully (backward compatible)

### Parallel Processing (commit `21cb561`)
- `parallel.py` module with `multiprocessing.Pool` support
- Pre-computes depth-averaged grids per unique z-range (avoids redundant work)
- Pre-computes prior once (avoids per-location sweep mutation issue)
- Worker function `_process_one_location` is standalone and picklable
- `--parallel N` / `-j N` CLI flag (0=auto, 1=sequential, N=N workers)
- Only activates for preloaded (model-based) location modes
- Sequential path unchanged — fully backward compatible

### Vectorized Depth Averaging (commit `21cb561`)
- `extract_calculated_values_in_depth_range` in `data_processing.py`
  replaced a triple-nested Python loop with `np.mean(data[:,:,:,z_inds], axis=3)`

### Location Modes
- `manual`: locations from YAML, seismic data from `vs_file`/`q_file`
- `locations_file`: locations from CSV, seismic from `vs_file`/`q_file`
- `model`: locations AND seismic data from `vs_file` (format auto-detected:
  .mat, .csv, .nc). Replaces the old `csv_model`, `mat_model`, `netcdf_model`
  modes (those still work as aliases for backward compatibility).

### CSV Output
Every location × method produces a row with:
`name, lat, lon, z, z_min, z_max, anelastic_method, T_ml, T_std, T_mean,
phi_ml, phi_std, phi_mean, gs_ml_mm, gs_std_mm, gs_mean_mm,
log10_eta_ml, log10_eta_std, log10_eta_mean, Vs_obs, Vs_pred, Vs_misfit,
Vs_chi2, Q_obs, Q_pred, Q_misfit, Q_chi2, chi2_total`

## 7. Architecture Notes

### Data Flow
1. **Config** (YAML) → `InversionConfig` dataclass (unknown keys like
   `sweep_generation` are silently filtered)
2. **Locations** prepared via `prepare_locations()` → list of (lat,lon), names,
   z_ranges, SeismicModelData. Data loaded from `vs_file`/`q_file` (format
   auto-detected).
3. **Sweep** loaded once (`.npz` or `.mat`); path set via `sweep_file` config
   field or `--sweep-file` CLI override
4. **Per method × per location:** compute likelihood → posterior → ML estimates
5. **Ensemble** accumulated across methods per location
6. **Output:** CSV + optional plots

### The Mutation Problem (why parallel.py exists)
`fit_preloaded_observations` mutates the shared `sweep` dict:
- Overwrites `sweep['meanVs']`, `sweep['meanQ']` with depth-averaged versions
- Normalizes `sweep['gs']` by `gsref` during prior computation
This prevented naive parallelization. `parallel.py` solves this by
pre-computing everything into read-only copies before dispatching workers.

### Key Functions
- `run_bayesian_inversion(config)` — main entry point (run_bayes.py)
- `fit_preloaded_observations(obs_vs, sigma_vs, ...)` — single location fit (fitting.py)
- `extract_ml_estimates(posterior, sweep, method)` — ML + posterior stats (fitting.py)
- `run_locations_parallel(locations, ..., n_workers)` — parallel dispatcher (parallel.py)
- `probability_distributions(type, ...)` — likelihood/posterior math (probability.py)

## 8. Known Issues / Active Investigations

### RESOLVED — Posterior inconsistency between validation pathways
Two validation approaches previously produced different posterior marginal shapes.
`validate_roundtrip.py` and `syntheticTest_adiabat/` have been **moved to the
workspace root** (no longer in the repo). Root cause was grid augmentation
(`np.union1d`) changing prior mass distribution — MAP estimates were identical.

### RESOLVED — Parallel mode obs_types enforcement
- `parallel.py`'s `_process_one_location` now uses the `use_vs` / `use_q`
  flags from the config to gate likelihood computation, rather than relying
  solely on data presence. This ensures `obs_types: Vs` in the config is
  respected even if Q data is available in the observations.
- The caller already guarded data population via `use_vs`/`use_q`, so the
  behavior was correct in practice; this fix adds defense-in-depth.

### RESOLVED — Cammarano 2003 inversion previously produced bad results
Earlier testing with `cammarano2003` yielded underestimated temperatures and
absurdly high viscosities (>10^26 Pa·s). This was **not a code bug** but a
matter of choosing appropriate physical parameters — in particular, the grain
size was set too large, leading to excessively high viscosity.
- The Cammarano pyrolite model does produce lower unrelaxed Vs than the
  pure-olivine anharmonic model at the same (T, P) due to Fe correction and
  VRH averaging with softer non-olivine phases — this is physically correct
  behaviour, not a bug.
- With appropriate parameter choices (e.g. realistic grain size, geotherm
  prior), the Patagonia test case using cammarano2003 produces plausible
  results.
- The mineral assemblage volume fractions are approximate (assembled from
  general knowledge); users should verify against preferred sources.

### KNOWN ISSUE — PyGMT 0.15 subplot API incompatibility
- `plot_map_results.py` uses manual `fig.shift_origin()` instead of `fig.subplot()`
  because PyGMT 0.15's subplot API does not correctly handle multi-panel layouts
  with per-panel colour maps and selective axis labelling.
- Workaround is stable but less elegant than native subfigure support.

### Other Issues
- **Synthetic adiabat test produces noisier results than roundtrip test** — this
  is expected and by design. The roundtrip test commits an "inverse crime":
  synthetic observations are created by nearest-neighbor grid lookup in the same
  sweep that the inversion uses, so recovery is near-perfect. The synthetic
  adiabat test computes observations independently via the full VBR core at
  exact (non-grid-snapped) T/φ/gs values, creating realistic model mismatch
  due to grid discretization, non-linear interpolation error, and frequency
  averaging differences. The adiabat test reveals the true resolution limits
  of the grid-based Bayesian approach.
- The `xarray` import in `data_processing.py` requires the correct conda env
  (`pyGMT2`) to be active, not just the `.venv`. Running with the wrong env
  gives `ModuleNotFoundError: No module named 'xarray'`.
- Parallel mode only works for preloaded (model-based) location modes, not for
  `manual` or `locations_file` modes (those use `fit_seismic_observations`
  which loads .mat files per-location).
- The `agent_history_ClaudeOpus4.6.json` file in the repo root is untracked
  and can be deleted or gitignored.
- Profile comparison figures from earlier inversion runs (before `run_tag` +
  `save_dir` changes) may still exist in the top-level `output/` directory;
  new runs save them inside the inversion subdirectory.

## 9. Test Results

| Test | Time (sequential) | Time (4 workers) | Locations × Methods |
|------|-------------------|-------------------|---------------------|
| WashU tomo (test_config.yaml) | ~22 min | ~8 min | 3168 × 4 |
| 2-location eta test | ~30s | N/A (manual mode) | 2 × 2 |
| Patagonia 3D (run_patagonia.py) | — | ~hours (16 cores) | ~12,672 × 1 |

Parallel and sequential CSV outputs are byte-for-byte identical (verified with `diff`).

## 10. Patagonia Test Case Details

### Configuration (`Patagonia_Test/config.yaml`)
- **Vs model**: WashU 3D tomography (NetCDF), subsampled ×2
- **Q model**: `stw105_nocrust` (built-in 1D)
- **Reference model**: `stw105_nocrust` (for profile comparison plots)
- **Elastic method**: `cammarano2003` — finite-strain mineral physics
- **Density model**: `stw105_nocrust`
- **Solidus**: `yk2001`
- **Direct melt effect**: YT2024 (`include_direct_melt_effect: 1`)
- **Depth range**: 10–650 km
- **Anelastic method**: `eburgers_psp`
- **T prior**: geotherm (SC2006, σ = 100°C)
- **GS prior**: uniform (log-uniform)
- **φ prior**: uniform
- **Obs types**: VsQ
- **Q error**: 100.0 (absolute)
- **Run tag**: `auto` → `inversion_eburgers_psp_Tgeo100_gsLU_phiU_VsQ_qe100`

### Output Structure (with run_tag: auto)
```
Patagonia_Test/output/
├── sweep.npz                   # shared across all inversions
├── sweep_fingerprint.json
├── lut_plots/
└── inversion_eburgers_psp_Tgeo100_gsLU_phiU_VsQ_qe100/
    ├── ml_estimates.csv        # 28-column results
    ├── *_ensembles.pkl
    ├── patagonia_profiles_*.png
    └── posteriors/
```

### Map Plot Usage (`plot_map_results.py`)
```bash
# Single depth
python Patagonia_Test/plot_map_results.py --depth 100

# Multiple depths (one row per depth)
python Patagonia_Test/plot_map_results.py --depth 50 100 200

# Custom variables and region
python Patagonia_Test/plot_map_results.py \
    --depth 80 150 \
    --vars T_mean log10_eta_mean phi_mean \
    --region -77 -64 -57 -42
```
