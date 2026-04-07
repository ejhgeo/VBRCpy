#!/usr/bin/env python3
"""
STW105 validation case.

Inverts STW105 Vs and Q profiles through a pre-computed VBR sweep to recover
temperature, melt fraction, grain size, and viscosity as a function of depth.

Unlike the geotherm round-trip test there is no known "true" answer — this
case checks that the inversion produces geophysically reasonable profiles
for a well-known 1-D reference Earth model.

Workflow
--------
  1. Generate a parameter sweep (look-up table) from config.yaml.
  2. Load STW105 Vs and Q reference profiles.
  3. Run the Bayesian inversion on the STW105 observations.
  4. Plot recovered profiles (T, Vs, Q, φ, gs, η) vs depth.

Usage
-----
From the repository root::

    python -m vbrc_V2Tpy.validation.validationCase_STW105.run_stw105
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml

# ---------------------------------------------------------------------------
# Paths (relative to repository root)
# ---------------------------------------------------------------------------
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXAMPLE_DIR, '..', '..', '..'))

CONFIG_FILE = os.path.join(EXAMPLE_DIR, 'config.yaml')

# All output paths derive from the single output_dir in config.yaml.
with open(CONFIG_FILE, 'r') as _f:
    _cfg = yaml.safe_load(_f)
_output_dir_rel = _cfg.get('output_dir', 'validation_tests/validationCase_STW105')
OUTPUT_DIR = os.path.join(REPO_ROOT, _output_dir_rel)

SWEEP_FILE        = os.path.join(OUTPUT_DIR, 'sweep.npz')
SWEEP_FINGERPRINT = os.path.join(OUTPUT_DIR, 'sweep_fingerprint.json')
INVERSION_DIR     = os.path.join(OUTPUT_DIR, 'inversion_results')

sys.path.insert(0, REPO_ROOT)
from vbrc_V2Tpy.bayesian_fitting_py.orchestration import (
    run_sweep_step, replot_lut, run_inversion_step,
)



# ===================================================================
# Step 4: profile plots from inversion output
# ===================================================================
def _make_profile_plots(depth_km, Vs_obs, Q_obs, ml_csv, method='xfit_premelt'):
    """Plot recovered profiles alongside reference model observations."""
    import pandas as pd

    ref_model = _cfg.get('reference_model', 'stw105').upper()

    sys.path.insert(0, REPO_ROOT)
    from vbrc_V2Tpy.bayesian_fitting_py.fitting import load_sweep_data
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.thermal import calculate_solidus_K
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.params import C2K
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.generate_sweep import load_sweep_params_from_yaml

    df = pd.read_csv(ml_csv)
    if 'anelastic_method' in df.columns:
        df = df[df['anelastic_method'] == method].copy()

    z_inv = df['z'].values if 'z' in df.columns else df['z_min'].values
    T_MAP  = df['T_ml'].values
    T_mean = df['T_mean'].values
    T_std  = df['T_std'].values
    phi_MAP  = df['phi_ml'].values
    phi_mean = df['phi_mean'].values if 'phi_mean' in df.columns else phi_MAP
    gs_MAP_mm  = df['gs_ml_mm'].values
    gs_mean_mm = df['gs_mean_mm'].values if 'gs_mean_mm' in df.columns else gs_MAP_mm
    Vs_pred = df['Vs_pred'].values if 'Vs_pred' in df.columns else None
    Q_pred  = df['Q_pred'].values if 'Q_pred' in df.columns else None
    has_Q   = Q_pred is not None

    log10_eta_MAP  = df['log10_eta_ml'].values if 'log10_eta_ml' in df.columns else None
    log10_eta_mean = df['log10_eta_mean'].values if 'log10_eta_mean' in df.columns else None
    log10_eta_std  = df['log10_eta_std'].values if 'log10_eta_std' in df.columns else None
    has_eta = log10_eta_MAP is not None

    # Compute mean-predicted Vs/Q from sweep at marginal mean parameters
    sweep = load_sweep_data(SWEEP_FILE)
    sweep_T   = sweep['T']
    sweep_phi = sweep['phi']
    sweep_gs  = sweep['gs']

    # Solidus profile
    sweep_params = load_sweep_params_from_yaml(CONFIG_FILE)
    sweep_params.output_file = SWEEP_FILE
    sweep_z_km = sweep['z'] / 1e3
    solidus_K = calculate_solidus_K(
        sweep['P_GPa'],
        method=sweep_params.solidus_method,
        depth_km=sweep_z_km,
        density_model=sweep_params.density_model,
        density_rho=sweep_params.rho,
        density_file=sweep_params.density_file,
    )
    solidus_C = solidus_K - C2K

    # Mean-predicted Vs/Q at marginal-mean state variables
    Vs_pred_mean = np.full(len(z_inv), np.nan)
    Q_pred_mean  = np.full(len(z_inv), np.nan) if has_Q else None
    for i, z in enumerate(z_inv):
        iz = np.argmin(np.abs(sweep['z'] / 1e3 - z))
        meanVs = sweep['Box'][method]['meanVs'][:, :, :, iz]
        i_T   = np.argmin(np.abs(sweep_T - T_mean[i]))
        i_phi = np.argmin(np.abs(sweep_phi - phi_mean[i]))
        i_gs  = np.argmin(np.abs(sweep_gs - gs_mean_mm[i] * 1000))
        Vs_pred_mean[i] = meanVs[i_T, i_phi, i_gs]
        if has_Q:
            meanQ = sweep['Box'][method]['meanQ'][:, :, :, iz]
            Q_pred_mean[i] = meanQ[i_T, i_phi, i_gs]

    # ---- Figure layout: 2×3 when Q and η are available ----
    use_2x3 = has_Q and has_eta
    if use_2x3:
        fig, axes = plt.subplots(2, 3, figsize=(14, 12), sharey=True)
        axf = axes.flatten()
    else:
        n_panels = 4 + int(has_Q) + int(has_eta)
        fig, axf = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 8), sharey=True)

    # Panel 1: Temperature
    ax = axf[0]
    ax.fill_betweenx(z_inv, T_mean - T_std, T_mean + T_std,
                     alpha=0.2, color='C0')
    ax.plot(solidus_C, sweep_z_km, 'r--', lw=1.5, alpha=0.7,
            label=f'Solidus ({sweep_params.solidus_method})')
    ax.plot(T_MAP, z_inv, 'C0-', lw=1.5, label='T (MAP)')
    ax.plot(T_mean, z_inv, 'C0--', lw=1.5, label='T (mean)')
    ax.set_xlabel('Temperature (°C)')
    ax.set_xlim([0, 2700])
    ax.set_ylabel('Depth (km)')
    ax.invert_yaxis()
    ax.legend(fontsize=7)
    ax.set_title('Temperature')

    # Panel 2: Vs
    ax = axf[1]
    ax.plot(Vs_obs, depth_km, 'k-', lw=2, label=f'{ref_model} Vs')
    if Vs_pred is not None:
        ax.plot(Vs_pred, z_inv, 'C1-', lw=1.5, label='Vs (MAP)')
    if Vs_pred_mean is not None:
        ax.plot(Vs_pred_mean, z_inv, 'C1--', lw=1.5, label='Vs (mean)')
    ax.set_xlabel('Vs (km/s)')
    ax.legend(fontsize=7)
    ax.set_title('Shear Velocity')

    # Panel 3: Q
    panel_idx = 2
    if has_Q:
        ax = axf[panel_idx]
        ax.plot(Q_obs, depth_km, 'k-', lw=2, label=f'{ref_model} Q')
        ax.plot(Q_pred, z_inv, 'C4-', lw=1.5, label='Q (MAP)')
        if Q_pred_mean is not None:
            ax.plot(Q_pred_mean, z_inv, 'C4--', lw=1.5, label='Q (mean)')
        ax.set_xlabel('Q')
        ax.set_xlim([50, 800])
        ax.legend(fontsize=7)
        ax.set_title('Quality Factor')
        panel_idx += 1

    # Panel: Melt fraction
    ax = axf[panel_idx]
    ax.plot(phi_MAP * 100, z_inv, 'C3-', lw=1.5, label='φ (MAP)')
    ax.plot(phi_mean * 100, z_inv, 'C3--', lw=1.5, label='φ (mean)')
    ax.set_xlabel('Melt Fraction (%)')
    ax.legend(fontsize=7)
    ax.set_title('Melt Fraction')
    if use_2x3:
        ax.set_ylabel('Depth (km)')
    panel_idx += 1

    # Panel: Grain size
    ax = axf[panel_idx]
    ax.plot(gs_MAP_mm, z_inv, 'C2-', lw=1.5, label='d (MAP)')
    ax.plot(gs_mean_mm, z_inv, 'C2--', lw=1.5, label='d (mean)')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_xscale('log')
    ax.set_xlim([0.1, 10])
    ax.legend(fontsize=7)
    ax.set_title('Grain Size')
    panel_idx += 1

    # Panel: Viscosity
    if has_eta:
        ax = axf[panel_idx]
        ax.plot(log10_eta_MAP, z_inv, 'C5-', lw=1.5, label='η (MAP)')
        ax.plot(log10_eta_mean, z_inv, 'C5--', lw=1.5, label='η (mean)')
        if log10_eta_std is not None:
            ax.fill_betweenx(z_inv, log10_eta_mean - log10_eta_std,
                             log10_eta_mean + log10_eta_std,
                             alpha=0.2, color='C5')
        ax.set_xlabel('log₁₀(η) [Pa·s]')
        ax.set_xlim([18, 25])
        ax.legend(fontsize=7)
        ax.set_title('Viscosity')

    # Build descriptive prior label for filename
    gs_type = _cfg.get('gs_prior_type', 'log_uniform')
    if gs_type == 'log_normal':
        mean_mm = _cfg.get('gs_prior_mean_mm', '?')
        std = _cfg.get('gs_prior_std', 0.25)
        prior_label = f'gsLN_{mean_mm}mm_std{std}'
    else:
        prior_label = 'gsLogUniform'
    obs = _cfg.get('obs_types', 'VsQ')

    fig.suptitle(f'{ref_model} Inversion – {method}  ({prior_label}, {obs})', fontsize=13)
    fig.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR,
                            f'stw105_profiles_{method}_{prior_label}_{obs}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Profile figure saved to {fig_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description='STW105 validation case: invert STW105 Vs/Q profiles',
        add_help=True,
    )
    parser.add_argument('--replot-lut', action='store_true',
                        help='Regenerate LUT plots from existing sweep and exit')
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.replot_lut:
        replot_lut(CONFIG_FILE, SWEEP_FILE, OUTPUT_DIR)
        return

    # ------------------------------------------------------------------
    # Step 1: Generate the parameter sweep
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Generate parameter sweep (look-up table)")
    print("=" * 70, flush=True)
    run_sweep_step(CONFIG_FILE, SWEEP_FILE, OUTPUT_DIR, SWEEP_FINGERPRINT)

    # ------------------------------------------------------------------
    # Step 2: Load reference model observations (for plotting)
    # ------------------------------------------------------------------
    ref_model = _cfg.get('reference_model', 'stw105')
    print("\n" + "=" * 70)
    print(f"STEP 2: Load {ref_model} Vs/Q reference profiles")
    print("=" * 70, flush=True)
    sys.path.insert(0, REPO_ROOT)
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.thermal import _load_earth_model
    depth_m, _density, vs_m_s, qmu = _load_earth_model(
        ref_model, fields=['Vs', 'Qmu'],
    )
    depth_km = depth_m / 1e3
    mask = (depth_km <= 660.0) & (depth_km >= 0) & (vs_m_s > 0)
    depth_km, Vs_obs, Q_obs = depth_km[mask], vs_m_s[mask] / 1e3, qmu[mask]
    print(f"  {len(depth_km)} depths, Vs: {Vs_obs.min():.3f}–{Vs_obs.max():.3f} km/s, "
          f"Q: {Q_obs.min():.1f}–{Q_obs.max():.1f}")

    # ------------------------------------------------------------------
    # Step 3: Run Bayesian inversion
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"STEP 3: Run Bayesian inversion on {ref_model} observations")
    print("=" * 70)
    run_inversion_step(
        CONFIG_FILE, SWEEP_FILE, INVERSION_DIR,
        vs_file=ref_model, q_file=ref_model, parallel=16,
    )

    # ------------------------------------------------------------------
    # Step 4: Plot recovered profiles
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Plot recovered profiles")
    print("=" * 70, flush=True)
    ml_csv = os.path.join(INVERSION_DIR, 'ml_estimates.csv')
    methods = _cfg.get('anelastic_methods', ['xfit_premelt'])
    if os.path.isfile(ml_csv):
        for method in methods:
            _make_profile_plots(depth_km, Vs_obs, Q_obs, ml_csv, method=method)
    else:
        print(f"  WARNING: ML estimates CSV not found at {ml_csv}")
        print("  Skipping profile plots.")

    print("\n" + "=" * 70)
    print("Done.  Outputs are in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
