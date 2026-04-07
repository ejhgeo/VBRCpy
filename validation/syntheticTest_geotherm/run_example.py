#!/usr/bin/env python3
"""
Geotherm round-trip validation example.

Same workflow as syntheticTest_adiabat, but uses a prescribed geotherm
(SC2006_geotherm.csv — Steinberger & Calderwood, 2006)
instead of a simple mantle adiabat for the true temperature profile.

  1. Generate a parameter sweep (look-up table) from a YAML config.
  2. Build synthetic Vs/Q observations from the geotherm profile.
  3. Run the Bayesian inversion on those synthetics using a YAML config.
  4. Compare recovered parameters against the known true values.

Usage
-----
From the repository root::

    python -m vbrc_V2Tpy.validation.syntheticTest_geotherm.run_example
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
GEOTHERM_CSV = os.path.join(EXAMPLE_DIR, 'SC2006_geotherm.csv')

# All output paths derive from the single output_dir in config.yaml.
with open(CONFIG_FILE, 'r') as _f:
    _cfg = yaml.safe_load(_f)
_output_dir_rel = _cfg.get('output_dir', 'validation_tests/syntheticTest_geotherm')
OUTPUT_DIR = os.path.join(REPO_ROOT, _output_dir_rel)

SWEEP_FILE       = os.path.join(OUTPUT_DIR, 'sweep.npz')
SWEEP_FINGERPRINT = os.path.join(OUTPUT_DIR, 'sweep_fingerprint.json')
SYNTH_CSV        = os.path.join(OUTPUT_DIR, 'synthetic_observations.csv')
INVERSION_DIR    = os.path.join(OUTPUT_DIR, 'inversion_results')

sys.path.insert(0, REPO_ROOT)
from vbrc_V2Tpy.bayesian_fitting_py.orchestration import (
    run_sweep_step, replot_lut, run_inversion_step,
)


# ===================================================================
# Geotherm loading
# ===================================================================
def load_geotherm(csv_path, max_depth_km=660.0):
    """Load the geotherm CSV and return (depth_km, temperature_C) arrays.

    The profile is truncated at *max_depth_km*.
    """
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    depth_km = data[:, 0]
    temperature_C = data[:, 1]
    mask = depth_km <= max_depth_km
    return depth_km[mask], temperature_C[mask]


# ===================================================================
# Synthetic profiles
# ===================================================================
def geotherm_T(depth_km_query, geotherm_z, geotherm_T_vals):
    """Interpolate geotherm temperature at arbitrary depths."""
    return np.interp(depth_km_query, geotherm_z, geotherm_T_vals)


def melt_fraction(depth_km, onset_km=80.0, max_phi=0.01):
    """Linear melt above *onset_km*; zero below."""
    phi = np.where(depth_km < onset_km,
                   max_phi * (1.0 - depth_km / onset_km), 0.0)
    return np.clip(phi, 0, max_phi)


def grain_size_um(depth_km, gs_um=1000.0):
    """Constant grain size (default 1.0 mm)."""
    return np.full_like(depth_km, gs_um, dtype=float)


# ===================================================================
# Step 2 helper: forward-model Vs directly from VBR core
# ===================================================================
def _generate_synthetic_csv(sweep_file, csv_out, geotherm_z, geotherm_T_vals,
                            method='xfit_premelt', gs_um=1000.0):
    """Compute exact synthetic Vs and Q via VBR core for the true profile.

    Instead of nearest-neighbour lookup in the sweep grid, this runs the
    full VBR calculation (elastic -> viscous -> anelastic) at the exact
    true T, phi, and gs values for each depth.
    """
    sys.path.insert(0, REPO_ROOT)
    from vbrc_V2Tpy.bayesian_fitting_py.fitting import load_sweep_data
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.core import VBR, StateVariables
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.thermal import calculate_solidus_K
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.params import C2K
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.generate_sweep import load_sweep_params_from_yaml

    print(f"Loading sweep from {sweep_file} ...")
    sweep = load_sweep_data(sweep_file)

    z_m = np.atleast_1d(sweep['z'])
    z_km = z_m / 1e3
    n_z = len(z_km)
    P_GPa = sweep['P_GPa']
    rho = sweep['rho']
    density_model = sweep.get('density_model', 'constant')

    sweep_params = load_sweep_params_from_yaml(CONFIG_FILE)
    sweep_params.output_file = SWEEP_FILE
    f = np.logspace(sweep_params.freq_log_min, sweep_params.freq_log_max,
                    sweep_params.n_freq)

    T_true = geotherm_T(z_km, geotherm_z, geotherm_T_vals)
    phi_true = melt_fraction(z_km)
    gs_true = grain_size_um(z_km, gs_um=gs_um)

    Vs_syn = np.zeros(n_z)
    Q_syn = np.zeros(n_z)
    eta_syn = np.zeros(n_z)

    print(f"Computing exact Vs/Q/eta via VBR core at {n_z} depths ...")
    for iz in range(n_z):
        T_K = np.atleast_1d(T_true[iz] + C2K)
        P = np.atleast_1d(P_GPa[iz])
        rho_iz = np.atleast_1d(rho[iz])
        dg_um = np.atleast_1d(gs_true[iz])
        phi_iz = np.atleast_1d(phi_true[iz])
        sig = np.atleast_1d(sweep_params.sig_MPa)

        Tsolidus_K = calculate_solidus_K(
            P_GPa[iz],
            method=sweep_params.solidus_method,
            depth_km=z_km[iz],
            density_model=density_model,
            density_rho=sweep_params.rho,
            density_file=sweep_params.density_file,
        )

        sv = StateVariables(
            T_K=T_K, P_GPa=P, rho=rho_iz, dg_um=dg_um,
            phi=phi_iz, sig_MPa=sig, f=f,
            Tsolidus_K=np.atleast_1d(Tsolidus_K),
        )

        base_elastic = sweep_params.elastic_method
        elastic_methods = [base_elastic]
        if phi_true[iz] > 0:
            elastic_methods.append('anh_poro')

        viscous_methods = ['HK2003', 'xfit_premelt'] if 'xfit_premelt' in [method] else ['HK2003']

        vbr = VBR(
            sv,
            elastic_methods=elastic_methods,
            anelastic_methods=[method],
            viscous_methods=viscous_methods,
        )

        if 'anharmonic' in elastic_methods:
            vbr.input['elastic']['anharmonic']['temperature_scaling'] = sweep_params.temperature_scaling
            vbr.input['elastic']['anharmonic']['pressure_scaling'] = sweep_params.pressure_scaling
            vbr.input['elastic']['anharmonic']['reference_scaling'] = sweep_params.reference_scaling

        vbr.run()

        result = vbr.output['anelastic'][method]
        Vs_syn[iz] = float(np.mean(result['V']) / 1e3)
        Q_syn[iz] = float(np.mean(result['Q']))

        # Extract viscosity from the viscous output
        if 'xfit_premelt' in vbr.output.get('viscous', {}):
            eta_syn[iz] = float(vbr.output['viscous']['xfit_premelt']['diff']['eta'])
        elif 'HK2003' in vbr.output.get('viscous', {}):
            eta_syn[iz] = float(vbr.output['viscous']['HK2003']['eta_total'])

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, 'w') as f:
        f.write('lon,lat,depth,vs,q\n')
        for iz in range(n_z):
            f.write(f'0.0,0.0,{z_km[iz]:.4f},{Vs_syn[iz]:.6f},{Q_syn[iz]:.2f}\n')

    print(f"Synthetic CSV ({n_z} depths) written to {csv_out}")
    print(f"  Vs range: {Vs_syn.min():.3f} – {Vs_syn.max():.3f} km/s")
    print(f"  Q  range: {Q_syn.min():.1f} – {Q_syn.max():.1f}")
    print(f"  η  range: {eta_syn.min():.2e} – {eta_syn.max():.2e} Pa·s")

    return z_km, T_true, phi_true, gs_true, Vs_syn, Q_syn, eta_syn


# ===================================================================
# Step 4 helper: comparison plots
# ===================================================================
def _make_comparison_plots(z_km, T_true, phi_true, gs_true, Vs_syn, Q_syn,
                           ml_csv, method='xfit_premelt', eta_true=None):
    """Plot recovered vs true profiles."""
    import pandas as pd
    sys.path.insert(0, REPO_ROOT)
    from vbrc_V2Tpy.bayesian_fitting_py.fitting import load_sweep_data

    df = pd.read_csv(ml_csv)
    if 'anelastic_method' in df.columns:
        df = df[df['anelastic_method'] == method].copy()

    z_inv = df['z'].values if 'z' in df.columns else df['z_min'].values
    T_MAP = df['T_ml'].values
    T_mean = df['T_mean'].values
    T_std = df['T_std'].values
    phi_MAP = df['phi_ml'].values
    phi_mean = df['phi_mean'].values if 'phi_mean' in df.columns else phi_MAP
    gs_MAP_mm = df['gs_ml_mm'].values
    gs_mean_mm = df['gs_mean_mm'].values if 'gs_mean_mm' in df.columns else gs_MAP_mm
    Vs_pred = df['Vs_pred'].values if 'Vs_pred' in df.columns else None
    Q_pred = df['Q_pred'].values if 'Q_pred' in df.columns else None
    has_Q = Q_pred is not None

    # Viscosity columns from inversion output
    log10_eta_MAP = df['log10_eta_ml'].values if 'log10_eta_ml' in df.columns else None
    log10_eta_mean = df['log10_eta_mean'].values if 'log10_eta_mean' in df.columns else None
    log10_eta_std = df['log10_eta_std'].values if 'log10_eta_std' in df.columns else None
    has_eta = log10_eta_MAP is not None and eta_true is not None

    # Compute mean-predicted Vs/Q from sweep at marginal mean parameters
    sweep = load_sweep_data(SWEEP_FILE)
    sweep_T = sweep['T']
    sweep_phi = sweep['phi']
    sweep_gs = sweep['gs']

    # Compute solidus profile for the temperature panel
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.thermal import calculate_solidus_K
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.params import C2K
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.generate_sweep import load_sweep_params_from_yaml
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

    Vs_pred_mean = np.full(len(z_inv), np.nan)
    Q_pred_mean = np.full(len(z_inv), np.nan) if has_Q else None
    for i, z in enumerate(z_inv):
        iz = np.argmin(np.abs(sweep['z'] / 1e3 - z))
        meanVs = sweep['Box'][method]['meanVs'][:, :, :, iz]
        i_T = np.argmin(np.abs(sweep_T - T_mean[i]))
        i_phi = np.argmin(np.abs(sweep_phi - phi_mean[i]))
        i_gs = np.argmin(np.abs(sweep_gs - gs_mean_mm[i] * 1000))
        Vs_pred_mean[i] = meanVs[i_T, i_phi, i_gs]
        if has_Q:
            meanQ = sweep['Box'][method]['meanQ'][:, :, :, iz]
            Q_pred_mean[i] = meanQ[i_T, i_phi, i_gs]

    # ---- Helper: compute error stats for a variable ----
    def _err_stats(predicted, true_matched):
        """Return (RMSE, MAE, mean percent error) for predicted vs true."""
        diff = predicted - true_matched
        rmse = float(np.sqrt(np.nanmean(diff ** 2)))
        mae = float(np.nanmean(np.abs(diff)))
        denom = np.where(np.abs(true_matched) > 0, np.abs(true_matched), np.nan)
        pct = float(np.nanmean(np.abs(diff) / denom) * 100)
        return rmse, mae, pct

    # Interpolate true profiles to inversion depths for error computation
    T_true_matched = np.interp(z_inv, z_km, T_true)
    phi_true_matched = np.interp(z_inv, z_km, phi_true)
    gs_true_matched_mm = np.interp(z_inv, z_km, gs_true) / 1000.0
    Vs_true_matched = np.interp(z_inv, z_km, Vs_syn)
    Q_true_matched = np.interp(z_inv, z_km, Q_syn) if has_Q else None

    # Temperature stats
    rmse_T, mae_T, pct_T = _err_stats(T_MAP, T_true_matched)
    print(f"\nRound-trip temperature recovery:")
    print(f"  RMSE = {rmse_T:.1f} °C    MAE = {mae_T:.1f} °C    %err = {pct_T:.1f}%")

    # Figure layout: 2×3 grid when Q and η are available, otherwise 1-row
    use_2x3 = has_Q and has_eta
    if use_2x3:
        fig, axes = plt.subplots(2, 3, figsize=(14, 12), sharey=True)
        axf = axes.flatten()  # [T, Vs, Q, φ, gs, η]
    else:
        n_panels = 4 + int(has_Q) + int(has_eta)
        fig, axf = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 8), sharey=True)

    # Panel 1: Temperature
    ax = axf[0]
    ax.fill_betweenx(z_inv, T_mean - T_std, T_mean + T_std,
                     alpha=0.2, color='C0')
    ax.plot(solidus_C, sweep_z_km, 'r--', lw=1.5, alpha=0.7,
            label=f'Solidus ({sweep_params.solidus_method})')
    ax.plot(T_true, z_km, 'k-', lw=2, label='True T')
    ax.plot(T_MAP, z_inv, 'C0-', lw=1.5,
            label=f'MAP  RMSE={rmse_T:.1f}°C  %err={pct_T:.1f}%')
    rmse_Tm, mae_Tm, pct_Tm = _err_stats(T_mean, T_true_matched)
    ax.plot(T_mean, z_inv, 'C0--', lw=1.5,
            label=f'Mean  RMSE={rmse_Tm:.1f}°C  %err={pct_Tm:.1f}%')
    ax.set_xlabel('Temperature (°C)')
    ax.set_xlim([0, 2700])
    ax.set_ylabel('Depth (km)')
    ax.invert_yaxis()
    ax.legend(fontsize=7)
    ax.set_title('Temperature')

    # Panel 2: Vs
    ax = axf[1]
    ax.plot(Vs_syn, z_km, 'k-', lw=2, label='Synthetic Vs')
    if Vs_pred is not None:
        rmse_Vs, mae_Vs, pct_Vs = _err_stats(Vs_pred, Vs_true_matched)
        ax.plot(Vs_pred, z_inv, 'C1-', lw=1.5,
                label=f'MAP  RMSE={rmse_Vs:.3f}  %err={pct_Vs:.2f}%')
    if Vs_pred_mean is not None:
        rmse_Vsm, mae_Vsm, pct_Vsm = _err_stats(Vs_pred_mean, Vs_true_matched)
        ax.plot(Vs_pred_mean, z_inv, 'C1--', lw=1.5,
                label=f'Mean  RMSE={rmse_Vsm:.3f}  %err={pct_Vsm:.2f}%')
    ax.set_xlabel('Vs (km/s)')
    ax.legend(fontsize=7)
    ax.set_title('Shear Velocity')

    # Panel 3: Q (if used)
    # Exclude depths where true Q exceeds threshold (cold lithosphere
    # drives Q → ∞, which would dominate RMSE).
    Q_ERR_THRESHOLD = 1500
    panel_idx = 2
    if has_Q:
        print(f"  Q range: {Q_syn.min():.1f} – {Q_syn.max():.1f}")
        q_valid = Q_true_matched < Q_ERR_THRESHOLD
        n_q_excl = int(np.sum(~q_valid))
        ax = axf[panel_idx]
        ax.plot(Q_syn, z_km, 'k-', lw=2, label='Synthetic Q')
        rmse_Q, mae_Q, pct_Q = _err_stats(Q_pred[q_valid], Q_true_matched[q_valid])
        ax.plot(Q_pred, z_inv, 'C4-', lw=1.5,
                label=f'MAP  RMSE={rmse_Q:.1f}  %err={pct_Q:.1f}%')
        if Q_pred_mean is not None:
            rmse_Qm, mae_Qm, pct_Qm = _err_stats(Q_pred_mean[q_valid], Q_true_matched[q_valid])
            ax.plot(Q_pred_mean, z_inv, 'C4--', lw=1.5,
                    label=f'Mean  RMSE={rmse_Qm:.1f}  %err={pct_Qm:.1f}%')
        ax.set_xlabel('Q')
        ax.set_xlim([50, 1600])
        ax.legend(fontsize=7)
        excl_str = f' (excl. {n_q_excl} pts Q>{Q_ERR_THRESHOLD})' if n_q_excl else ''
        ax.set_title(f'Quality Factor{excl_str}')
        panel_idx += 1

    # Panel: Melt fraction (red)
    ax = axf[panel_idx]
    ax.plot(phi_true * 100, z_km, 'k-', lw=2, label='True φ')
    rmse_phi, mae_phi, pct_phi = _err_stats(phi_MAP, phi_true_matched)
    ax.plot(phi_MAP * 100, z_inv, 'C3-', lw=1.5,
            label=f'MAP  MAE={mae_phi:.4f}  %err={pct_phi:.1f}%')
    rmse_phim, mae_phim, pct_phim = _err_stats(phi_mean, phi_true_matched)
    ax.plot(phi_mean * 100, z_inv, 'C3--', lw=1.5,
            label=f'Mean  MAE={mae_phim:.4f}  %err={pct_phim:.1f}%')
    ax.set_xlabel('Melt Fraction (%)')
    ax.legend(fontsize=7)
    ax.set_title('Melt Fraction')
    if use_2x3:
        ax.set_ylabel('Depth (km)')  # y-label for second row
    panel_idx += 1

    # Panel: Grain size (green)
    ax = axf[panel_idx]
    rmse_gs, mae_gs, pct_gs = _err_stats(gs_MAP_mm, gs_true_matched_mm)
    rmse_gsm, mae_gsm, pct_gsm = _err_stats(gs_mean_mm, gs_true_matched_mm)
    ax.plot(gs_true / 1000, z_km, 'k-', lw=2, label='True d')
    ax.plot(gs_MAP_mm, z_inv, 'C2-', lw=1.5,
            label=f'MAP  RMSE={rmse_gs:.3f}  %err={pct_gs:.1f}%')
    ax.plot(gs_mean_mm, z_inv, 'C2--', lw=1.5,
            label=f'Mean  RMSE={rmse_gsm:.3f}  %err={pct_gsm:.1f}%')
    ax.set_xlabel('Grain Size (mm)')
    ax.legend(fontsize=7)
    ax.set_title('Grain Size')
    panel_idx += 1

    # Panel: Viscosity (if available)
    if has_eta:
        ax = axf[panel_idx]
        log10_eta_true_matched = np.interp(z_inv, z_km, np.log10(eta_true))
        rmse_eta, mae_eta, pct_eta = _err_stats(log10_eta_MAP, log10_eta_true_matched)
        rmse_etam, mae_etam, pct_etam = _err_stats(log10_eta_mean, log10_eta_true_matched)
        ax.plot(np.log10(eta_true), z_km, 'k-', lw=2, label='True η')
        ax.plot(log10_eta_MAP, z_inv, 'C5-', lw=1.5,
                label=f'MAP  RMSE={rmse_eta:.2f}  %err={pct_eta:.1f}%')
        ax.plot(log10_eta_mean, z_inv, 'C5--', lw=1.5,
                label=f'Mean  RMSE={rmse_etam:.2f}  %err={pct_etam:.1f}%')
        if log10_eta_std is not None:
            ax.fill_betweenx(z_inv, log10_eta_mean - log10_eta_std,
                             log10_eta_mean + log10_eta_std,
                             alpha=0.2, color='C5')
        ax.set_xlabel('log₁₀(η) [Pa·s]')
        ax.set_xlim([18, 25]) # Adjust as needed for better visualization
        ax.legend(fontsize=7)
        ax.set_title('Viscosity')

    fig.suptitle(f'Round-Trip Validation (SC2006 Geotherm) – {method}\n'
                 f'RMSE(T) = {rmse_T:.1f} °C   MAE(T) = {mae_T:.1f} °C',
                 fontsize=13)
    fig.tight_layout()

    # Build a descriptive prior label for the filename
    gs_type = _cfg.get('gs_prior_type', 'log_uniform')
    if gs_type == 'log_normal':
        mean_mm = _cfg.get('gs_prior_mean_mm', '?')
        std = _cfg.get('gs_prior_std', 0.25)
        prior_label = f'gsLN_{mean_mm}mm_std{std}'
    else:
        prior_label = 'gsLogUniform'
    obs = _cfg.get('obs_types', 'VsQ')
    fig_path = os.path.join(OUTPUT_DIR, f'inversion_results_{method}_{prior_label}_{obs}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison figure saved to {fig_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Geotherm round-trip validation example',
        add_help=True,
    )
    parser.add_argument('--replot-lut', action='store_true',
                        help='Regenerate LUT plots from existing sweep and exit')
    parser.add_argument('--gs-um', type=float, default=1000.0,
                        help='True grain size in microns (default: 1000 = 1 mm)')
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.replot_lut:
        replot_lut(CONFIG_FILE, SWEEP_FILE, OUTPUT_DIR)
        return

    # Load the geotherm profile (truncated at 660 km)
    geotherm_z, geotherm_T_vals = load_geotherm(GEOTHERM_CSV, max_depth_km=660.0)
    print(f"Loaded geotherm: {len(geotherm_z)} points, "
          f"{geotherm_z.min():.0f}–{geotherm_z.max():.0f} km, "
          f"{geotherm_T_vals.min():.0f}–{geotherm_T_vals.max():.0f} °C")

    # ------------------------------------------------------------------
    # Step 1: Generate the parameter sweep
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Generate parameter sweep (look-up table)")
    print("=" * 70, flush=True)
    run_sweep_step(CONFIG_FILE, SWEEP_FILE, OUTPUT_DIR, SWEEP_FINGERPRINT)

    # ------------------------------------------------------------------
    # Step 2: Create synthetic observations from the geotherm
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Forward-model synthetic Vs/Q from SC2006 geotherm")
    print("=" * 70, flush=True)
    z_km, T_true, phi_true, gs_true, Vs_syn, Q_syn, eta_true = \
        _generate_synthetic_csv(SWEEP_FILE, SYNTH_CSV,
                                geotherm_z, geotherm_T_vals,
                                gs_um=args.gs_um)

    # ------------------------------------------------------------------
    # Step 3: Run Bayesian inversion via the standard CLI
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Run Bayesian inversion on synthetic observations")
    print("=" * 70)
    run_inversion_step(
        CONFIG_FILE, SWEEP_FILE, INVERSION_DIR,
        vs_file=SYNTH_CSV, q_file=SYNTH_CSV, parallel=16,
    )

    # ------------------------------------------------------------------
    # Step 4: Compare recovered parameters against truth
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Compare recovered parameters against known truth")
    print("=" * 70, flush=True)
    ml_csv = os.path.join(INVERSION_DIR, 'ml_estimates.csv')
    if os.path.isfile(ml_csv):
        _make_comparison_plots(z_km, T_true, phi_true, gs_true, Vs_syn, Q_syn,
                               ml_csv, eta_true=eta_true)
    else:
        print(f"  WARNING: ML estimates CSV not found at {ml_csv}")
        print("  Skipping comparison plots.")

    print("\n" + "=" * 70)
    print("Done.  Outputs are in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
