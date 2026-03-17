# vbrc_V2Tpy — Project State Document

> **Last updated:** 2026-03-16
> **Repo:** https://github.com/ejhgeo/vbrc_V2Tpy.git
> **Latest commit:** `4c95424` — "Fix lateral-only subsampling, add parallel progress reporting"
> **Uncommitted changes:** Yes — depth-dependent density, Cammarano 2003, plot_lut updates, depth-range guards, validation framework, example workflow, sweep caching, force_plots, Q synthetic support, posterior plotting improvements, benchmarkTest_vsMatlab

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
├── compare_sweeps.py
├── data/                              # bundled .mat data files
│   ├── vel_models/Shen_Ritzwoller_2016.mat
│   ├── Q_models/Dalton_Ekstrom_2008.mat
│   ├── LAB_models/HopperFischer2018.mat
│   └── plate_VBR/sweep_log_gs.mat
├── validation/                        # validation & testing framework
│   ├── __init__.py
│   ├── run_all.py            # orchestrator for all validation cases
│   ├── validate_prem.py      # Case 1: PREM velocity inversion
│   ├── validate_roundtrip.py # Case 2: self-contained adiabat round-trip
│   ├── MAP_vs_Mean_explanation.md
│   ├── syntheticTest_adiabat/  # realistic config/CLI validation pipeline
│   │   ├── __init__.py / __main__.py
│   │   ├── run_example.py     # 4-step orchestrator (sweep→synth→inversion→compare)
│   │   ├── sweep_config.yaml  # sweep generation config (131 depths, PREM density)
│   │   └── inversion_config.yaml  # inversion config (csv_model, force_plots)
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

# Self-contained round-trip validation (with method selection)
python -m vbrc_V2Tpy.validation.validate_roundtrip \
    --output validation_results/roundtrip_rhoprem_anharmonic \
    --elastic anharmonic --density prem --solidus hirschmann

# Synthetic adiabat test — realistic config/CLI pipeline
python -m vbrc_V2Tpy.validation.syntheticTest_adiabat.run_example

# Benchmark: Python vs MATLAB VBRc (sweep comparison + LUT plots + inversion)
python -m vbrc_V2Tpy.validation.benchmarkTest_vsMatlab

# Run all validation cases
python -m vbrc_V2Tpy.validation.run_all --sweep sweep.npz
```

### Config files in the workspace

| File | Description |
|------|-------------|
| `test_config.yaml` | Full WashU tomography NetCDF, 4 methods, Vs only, model_z_range [90,105] |
| `test_eta_config.yaml` | 2 manual locations, 2 methods, for quick viscosity testing |
| `test_parallel_config.yaml` | NetCDF model subsampled 100×, 1 method, for parallel testing |
| `sweep_config.yaml` | Config for regenerating the parameter sweep |
| `validation/syntheticTest_adiabat/sweep_config.yaml` | Sweep for validation: 131 depths, PREM density, hirschmann solidus |
| `validation/syntheticTest_adiabat/inversion_config.yaml` | Inversion for validation: csv_model, xfit_premelt, force_plots |
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

### Validation Framework (uncommitted)
- **`validation/` directory** with modular test cases and a standalone example workflow
- **`validate_roundtrip.py`**: Self-contained round-trip validation that generates
  a sweep internally, forward-models Vs from a known adiabat + melt + grain-size
  profile, inverts them, and compares recovered vs true parameters
  - CLI flags: `--elastic` (anharmonic | cammarano2003), `--density` (prem | constant),
    `--solidus` (hirschmann | katz | yk2001), `--output`
  - Uses augmented state grids (`np.union1d` with true profile values) for
    best-case grid resolution — this intentionally differs from production grids
- **`validate_prem.py`**: PREM velocity inversion validation
- **`run_all.py`**: Orchestrator to run all validation cases sequentially

### Synthetic Adiabat Test — Realistic Config/CLI Pipeline (uncommitted)
- **`validation/syntheticTest_adiabat/`**: End-to-end validation that exercises the
  exact same code paths a user would run (config files + CLI subprocesses)
- Output written to `validation_results/syntheticTest_adiabat/` (outside the
  git repo, in the workspace root)
- 4-step orchestrator (`run_example.py`):
  1. Generate parameter sweep from `sweep_config.yaml`
  2. Build synthetic Vs+Q observations from a prescribed adiabat
  3. Run Bayesian inversion via `inversion_config.yaml`
  4. Plot recovered vs true profiles + per-depth comparison
- **Sweep fingerprint caching**: SHA-256 hash of `sweep_config.yaml` stored
  alongside sweep file; regeneration only when config changes
- **Q in synthetic observations**: CSV output includes `lon,lat,depth,vs,q`
  columns; can be used with `obs_types: VsQ` in inversion config
- Package entry points (`__init__.py`, `__main__.py`) for `python -m` invocation

### MATLAB Benchmark Validation — benchmarkTest_vsMatlab (uncommitted)
- **`validation/benchmarkTest_vsMatlab/`**: Automated comparison of the Python
  VBRc against the original MATLAB VBRc output (`vbr/test.mat`)
- 4-step orchestrator (`run_benchmark.py`):
  1. Generate parameter sweep with settings matching original MATLAB defaults
  2. Point-by-point numerical comparison against `vbr/test.mat` (prints
     per-method Vs/Q differences and melt-effect analysis, mirroring
     `compare_sweeps.py`)
  3. LUT comparison plots (T vs gs, gs vs phi, T vs phi) at multiple depths
     including the max-diff depth, for all 4 methods
  4. Bayesian inversion at 3 manual locations (Basin & Range, Colorado
     Plateau, Interior) with all 4 anelastic methods
- Full-grid summary: reports median and max |%diff| for Vs and Q, with the
  (T, gs, z) coordinates of the worst-case point, plus a separate summary
  excluding the deepest depth slice
- Sweep fingerprint caching (SHA-256 hash of config YAML)
- Output to `validation_results/benchmarkTest_vsMatlab/` (outside git repo)
- Invocation: `python -m vbrc_V2Tpy.validation.benchmarkTest_vsMatlab`

### force_plots Config Option (uncommitted)
- `force_plots: true` in inversion YAML (or `--force-plots` CLI flag) forces
  posterior plot generation even for large-scale runs that would normally skip plots
- Non-interactive prompt fix: when running as a subprocess (stdin not a TTY),
  auto-confirms without blocking on `input()` — fixes `EOFError` in pipeline usage

### Posterior Plotting Improvements (uncommitted)
- Depth included in posterior plot titles and filenames (`posterior_{z}km_{obs}_{method}.png`)
- Consistent depth labeling across both validation pathways (roundtrip and workflow)

### LUT Auto-Generation During Sweep (uncommitted)
- `plot_lut` config section in sweep YAML: `enabled`, `output_dir`, `every_n`
- Automatically generates look-up table diagnostic plots during sweep generation
- Controlled via `SweepParams.plot_lut`, `plot_lut_dir`, `plot_lut_every_n`

### Configurable Solidus Method (uncommitted)
- `solidus_method` field in `SweepParams` and sweep YAML config
- Options: `hirschmann` (default), `katz`, `yk2001`
- YK2001 solidus handling fixed for correct pressure/temperature behavior

### Cammarano et al. (2003) Finite-Strain Elastic Method (uncommitted)
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

### Depth-Dependent Density (uncommitted)
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

### Depth Range Guard (uncommitted)
- `data_processing.py` and `fitting.py`: `extract_calculated_values_in_depth_range`
  now raises `ValueError` with clear message when observation depths don't overlap
  sweep depths (previously returned NaN silently)

### plot_lut.py Multi-Format Support (uncommitted)
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
Two validation approaches produce different posterior marginal shapes for the
same physical methods (`elastic: anharmonic`, `density: prem`, `solidus: hirschmann`):

1. **`validate_roundtrip.py`** (direct, self-contained) — generates its own sweep,
   augments state grids with true profile values via `np.union1d`, computes
   posteriors directly via `probability_distributions` + `prior_model_probs`.
2. **`syntheticTest_adiabat/run_example.py`** (subprocess/config path) — uses
   pre-generated sweep on a fixed grid, runs inversion through `run_bayes.py`
   and `fit_preloaded_observations`.

- **Root cause confirmed:** Grid augmentation is the **sole** source of
  difference. A diagnostic script (`validation/diagnose_pathway_diff.py`)
  verified that all three code paths (direct/roundtrip-style, fit_preloaded,
  parallel worker) produce **bit-for-bit identical** posterior arrays when
  given the same sweep grid, at every tested depth.
- **Why the plots look different:** When `validate_roundtrip.py` inserts true
  profile values into the T/φ/gs axes via `np.union1d`, it changes the 3D
  posterior array dimensions and the discrete prior mass distribution. `imshow`
  renders these non-uniformly-spaced cells as uniform pixels, distorting the
  visual appearance. Marginal PDFs sum over different numbers of bins.
- **MAP estimates agree** — both grids recover the same parameter values.
  Only the posterior shape (widths, tails) changes due to prior redistribution.

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
