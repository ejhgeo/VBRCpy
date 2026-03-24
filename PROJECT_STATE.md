# vbrc_V2Tpy — Project State Document

> **Last updated:** 2026-03-24
> **Repo:** https://github.com/ejhgeo/vbrc_V2Tpy.git
> **Latest commit:** `8a93465` — "Fix MATLAB struct indexing bug, rename compare_lut_slices, cleanup"
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
├── config_example_bayesian_fitting.yaml
├── config_example_regenerate_sweep.yaml
├── data/                              # bundled .mat data files
│   ├── vel_models/Shen_Ritzwoller_2016.mat
│   ├── Q_models/Dalton_Ekstrom_2008.mat
│   ├── LAB_models/HopperFischer2018.mat
│   └── plate_VBR/sweep_log_gs.mat
├── validation/                        # validation & testing framework
│   ├── __init__.py
│   ├── syntheticTest_geotherm/  # geotherm-based validation (SC2006 continental geotherm)
│   │   ├── README.md           # usage, assumptions, output description
│   │   ├── __init__.py / __main__.py
│   │   ├── run_example.py     # 4-step orchestrator with --gs-um and --replot-lut
│   │   ├── sweep_config.yaml  # Cammarano 2003, PREM, YK2001, xfit_premelt, 131 depths
│   │   ├── inversion_config.yaml  # csv_model, VsQ, percent Q error, log-normal gs prior
│   │   └── SC2006_geotherm.csv  # prescribed geotherm (120 pts, 0–3000 km)
│   └── benchmarkTest_vsMatlab/  # Python vs MATLAB VBRc benchmark
│       ├── __init__.py / __main__.py
│       ├── run_benchmark.py   # 4-step orchestrator (sweep→compare→LUT plots→inversion)
│       ├── sweep_config.yaml  # matches original MATLAB sweep params
│       └── inversion_config.yaml  # manual locations, all 4 methods
└── bayesian_fitting_py/               # main package
    ├── __init__.py / __main__.py
    ├── run_bayes.py          # CLI entry point + InversionConfig + main loop
    ├── fitting.py            # fit_seismic_observations, fit_preloaded_observations, extract_ml_estimates
    ├── data_processing.py    # Location, SeismicModelData, loaders (CSV/mat/NetCDF)
    ├── prior.py              # prior_model_probs, store_ensemble, confidence_cutoffs
    ├── probability.py        # probability_distributions (likelihood, posterior, combined)
    ├── parallel.py           # multiprocessing support for large-scale runs
    ├── plotting.py           # all figure generation
    ├── fetch_data.py         # interactive data downloader
    └── vbr/                  # VBR core calculations (Python port of MATLAB VBRc)
        ├── core.py           # anelastic methods: andrade_psp, eburgers_psp, xfit_mxw, xfit_premelt
        ├── cammarano.py      # Cammarano et al. (2003) finite-strain mineral physics
        ├── generate_sweep.py # parameter sweep generation (Box with meanVs, meanQ, meanEta)
        ├── params.py         # default VBR parameters
        ├── thermal.py        # half-space cooling, adiabatic geotherm
        ├── plot_lut.py       # look-up table plotting (.mat, .npz, .pkl)
        └── PREM_for_VBRc.csv # PREM density profile for depth-dependent density
```

## 3. Python Environment

- **Python:** 3.9.6 (system)
- **venv:** `/Users/ehightow/Research/V2T_Inversion/.venv/`
- **conda env (with xarray):** `pyGMT2`
- **Key deps:** numpy, scipy, matplotlib, h5py (for .mat), xarray + netCDF4 (for NetCDF)
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
| `validation/syntheticTest_geotherm/sweep_config.yaml` | Sweep for geotherm validation: 131 depths, Cammarano 2003, PREM density, YK2001 solidus, YT2024 |
| `validation/syntheticTest_geotherm/inversion_config.yaml` | Inversion for geotherm: csv_model, xfit_premelt, VsQ, percent Q error, log-normal gs prior |
| `validation/benchmarkTest_vsMatlab/sweep_config.yaml` | Sweep matching original MATLAB VBRc params (T:1100–1800, 100 depths) |
| `validation/benchmarkTest_vsMatlab/inversion_config.yaml` | Inversion: 3 manual locations, all 4 methods, VsQ |

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

### Recent Uncommitted Changes (2026-03-24)

#### Synthetic Geotherm Validation Test (`syntheticTest_geotherm`)
- **New validation case:** `validation/syntheticTest_geotherm/` uses the SC2006
  continental geotherm (Stixrude & Lithgow-Bertelloni 2006) instead of a simple
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
- `generate_sweep.py` stores `meanEta` in the Box — HK2003 composite `eta_total`
  (same for all anelastic methods; depends only on T, P, φ, gs)
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
- `manual`: locations from YAML, seismic data from .mat files
- `locations_file`: locations from CSV, seismic from .mat files
- `csv_model`: locations + Vs/Q from a single CSV
- `mat_model`: locations + Vs/Q from a .mat file
- `netcdf_model`: locations + Vs/Q from a NetCDF file (xarray)

### CSV Output
Every location × method produces a row with:
`name, lat, lon, z, z_min, z_max, anelastic_method, T_ml, T_std, T_mean,
phi_ml, phi_std, phi_mean, gs_ml_mm, gs_std_mm, gs_mean_mm,
log10_eta_ml, log10_eta_std, log10_eta_mean, Vs_obs, Vs_pred, Vs_misfit,
Vs_chi2, Q_obs, Q_pred, Q_misfit, Q_chi2, chi2_total`

## 7. Architecture Notes

### Data Flow
1. **Config** (YAML) → `InversionConfig` dataclass
2. **Locations** prepared via `prepare_locations()` → list of (lat,lon), names, z_ranges, SeismicModelData
3. **Sweep** loaded once (`.npz` or `.mat`)
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

### KNOWN BUG — Cammarano 2003 inversion produces bad results
When sweep is generated with `method: cammarano2003` and Bayesian inversion is run,
temperatures are underestimated at all depths (even deeper upper mantle) and
viscosities are absurdly high (>10^26 Pa·s, should be ~10^21).
- **Suspected root cause:** The Cammarano pyrolite model produces lower
  unrelaxed Vs than the pure-olivine anharmonic model at the same (T, P):
  - Fe correction reduces olivine G₀ from 81 → 77.9 GPa
  - VRH averaging with softer non-olivine phases (cpx G₀≈66 GPa, opx G₀≈76 GPa)
  - Combined effect: lower Gu → lower unrelaxed Vs → matching observed Vs
    requires cooler T → exponentially higher η
- **Also applies at transition zone depths** (not just upper mantle)
- The mineral assemblage volume fractions are approximate and could compound the issue
- Unit tests (Vs values at individual T, P points) pass correctly; the problem
  manifests only through the full Bayesian inversion loop
- Needs investigation: compare Vs between anharmonic and cammarano2003 at same
  conditions; validate against published Cammarano velocity profiles; possibly
  adjust assemblage fractions or Fe content

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
- Consider adding Q observations to the tomography inversions (currently Vs only
  for the WashU model since it doesn't include Q).

## 9. Test Results

| Test | Time (sequential) | Time (4 workers) | Locations × Methods |
|------|-------------------|-------------------|---------------------|
| WashU tomo (test_config.yaml) | ~22 min | ~8 min | 3168 × 4 |
| 2-location eta test | ~30s | N/A (manual mode) | 2 × 2 |

Parallel and sequential CSV outputs are byte-for-byte identical (verified with `diff`).
