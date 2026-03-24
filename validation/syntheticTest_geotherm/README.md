# Synthetic Geotherm Round-Trip Validation

End-to-end validation test that uses a prescribed continental geotherm
(SC2006 — Steinberger and Calderwood, 2006) instead of a simple mantle
adiabat as the "true" temperature profile.  This exercises the full
inversion pipeline under more realistic thermal conditions, including
a cold lithospheric lid, a thermal boundary layer, and a convecting
interior.

## Purpose

Verify that the Bayesian inversion can recover known thermodynamic
state variables (T, φ, grain size, viscosity) from synthetic seismic
observations generated with the same VBR forward model but computed
independently (point-by-point VBR core evaluation, not grid lookup).
This avoids the "inverse crime" of inverting on the same grid used to
generate the data.

## What It Tests

| Variable | True profile | Recovery metric |
|---|---|---|
| Temperature | SC2006 continental geotherm (0–660 km) | RMSE, MAE, %error |
| Melt fraction | Linear ramp above 80 km (max 1%); zero below | MAE, %error |
| Grain size | Constant (default 1 mm, configurable via `--gs-um`) | RMSE, %error |
| Viscosity | xfit_premelt diffusion creep (YT2016) | RMSE in log₁₀(η) |
| Vs, Q | Forward-modelled by the VBR core at each depth | RMSE, %error |

## Physical Assumptions

- **Elastic method:** Cammarano et al. (2003) finite-strain mineral physics
  with depth-dependent pyrolite mineralogy (olivine → wadsleyite →
  ringwoodite assemblage transitions at ~410 and ~520 km).
- **Anelastic method:** `xfit_premelt` (Yamauchi & Takei 2016) with
  direct melt effects enabled (YT2024 mode:
  `include_direct_melt_effect: 1`).
- **Viscosity:** xfit_premelt diffusion creep with near-solidus A_n
  scaling.  The sweep stores method-consistent viscosity (xfit_premelt
  η when that method is selected, otherwise HK2003).
- **Solidus:** Yamazaki & Karato (2001) parameterization (`yk2001`).
- **Density:** PREM depth-dependent profile for lithostatic pressure.
- **Grain-size prior:** Log-normal centered at 0.8 mm (σ = 0.25 in
  log-space).
- **Q error model:** Percentage-based (`q_error_mode: percent`,
  `default_q_error: 12%`), so shallow depths with Q → ∞ (cold
  lithosphere) do not dominate the likelihood.

## Directory Contents

| File | Description |
|---|---|
| `run_example.py` | 4-step orchestrator: sweep → synthetic obs → inversion → comparison |
| `sweep_config.yaml` | Parameter sweep settings (T, φ, gs, depth grids; elastic/anelastic methods) |
| `inversion_config.yaml` | Bayesian inversion settings (priors, error model, output paths) |
| `SC2006_geotherm.csv` | Prescribed geotherm: 120 depth points (0–3000 km), truncated to 660 km at runtime |

## Usage

From the **workspace root** (`V2T_Inversion/`):

```bash
# Activate the virtual environment
source .venv/bin/activate 
or 
conda activate myENV

# Run with default true grain size (1 mm = 1000 µm)
python -m vbrc_V2Tpy.validation.syntheticTest_geotherm

# Specify a different true grain size (e.g. 800 µm) (and update your prior in the config file if you want)
python -m vbrc_V2Tpy.validation.syntheticTest_geotherm --gs-um 800

# Regenerate LUT diagnostic plots from existing sweep (no recomputation)
python -m vbrc_V2Tpy.validation.syntheticTest_geotherm --replot-lut
```

### Pipeline Steps

1. **Generate parameter sweep** — builds a 4-D look-up table
   (T × φ × gs × z) of mean Vs, Q, and η.  Cached via SHA-256
   fingerprint of the sweep config; only regenerated when the config
   changes to save time.
2. **Forward-model synthetic observations** — evaluates the full VBR
   core (elastic → viscous → anelastic) at the exact true T, φ, gs
   for each depth.  Writes a CSV with columns `lon, lat, depth, vs, q`.
3. **Run Bayesian inversion** — calls the standard
   `bayesian_fitting_py` CLI on the synthetic CSV, producing posterior
   distributions and ML estimates at each depth.
4. **Comparison plots** — generates a 2×3 figure (T, Vs, Q, φ, gs, η)
   comparing MAP and marginal-mean estimates against the known truth,
   with RMSE / MAE / %error statistics in every legend.  The solidus
   curve is overlaid on the temperature panel.

### Output

All output is written to
`validation_tests/syntheticTest_geotherm_cammarano2003_xfitpremelt/`
(relative to the workspace root, outside the git-tracked package)
Change to whatever directory you want to save output to:

```
validation_tests/syntheticTest_geotherm_cammarano2003_xfitpremelt/
├── sweep.npz                  # Parameter sweep look-up table
├── sweep_fingerprint.json     # SHA-256 hash for change detection
├── synthetic_observations.csv # Synthetic Vs/Q profile
├── lut_plots/                 # Lookup Table diagnostic slices (every 5th depth)
└── inversion_results/         # Bayesian inversion output
    ├── ml_estimates.csv       # MAP/mean/std for all variables
    ├── posteriors/             # Posterior PDF plots (every 13th depth)
    └── inversion_results_xfit_premelt_*.png  # Comparison figure
```

## Customisation

- **True grain size:** `--gs-um <value>` (default 1000 µm = 1 mm)
- **Melt profile:** edit `melt_fraction()` in `run_example.py`
  (onset depth, max φ)
- **Geotherm:** replace `SC2006_geotherm.csv` with any two-column
  `depth_km, temperature_C` CSV
- **Sweep resolution:** adjust grid bounds/steps in `sweep_config.yaml`
- **Inversion priors:** edit `gs_prior_type`, `gs_prior_mean_mm`,
  `gs_prior_std`, `q_error_mode`, etc. in `inversion_config.yaml`

## Known Behaviours

- **Q → ∞ at shallow depths:** Expected for cold lithosphere where
  homologous temperature Tn ≪ 0.7.  The xfit_premelt model predicts
  negligible dissipation, yielding Q values of 10⁵–10⁸.  Error
  statistics for Q exclude depths where true Q > 1500 to avoid
  dominated RMSE.
- **Solidus on the temperature panel:** Plotted as a red dashed line
  using the configured solidus method (yk2001 by default).
  Temperatures approaching the solidus trigger the A_n near-solidus
  scaling in the xfit_premelt viscosity model.
