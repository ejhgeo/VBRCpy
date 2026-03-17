#!/usr/bin/env python3
"""
Standalone round-trip validation example.

Demonstrates the full user workflow using config files and CLI commands,
exactly as a user would run the code:

  1. Generate a parameter sweep (look-up table) from a YAML config.
  2. Build synthetic Vs observations from a prescribed adiabat.
  3. Run the Bayesian inversion on those synthetics using a YAML config.
  4. Compare recovered parameters against the known true values.

Usage
-----
From the repository root::

    python vbrc_V2Tpy/validation/syntheticTest_adiabat/run_example.py

Or equivalently::

    python -m vbrc_V2Tpy.validation.syntheticTest_adiabat.run_example
"""

import os
import sys
import json
import hashlib
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths (relative to repository root)
# ---------------------------------------------------------------------------
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXAMPLE_DIR, '..', '..', '..'))
OUTPUT_DIR = os.path.join(REPO_ROOT, 'validation_results', 'syntheticTest_adiabat')

SWEEP_CONFIG = os.path.join(EXAMPLE_DIR, 'sweep_config.yaml')
INVERSION_CONFIG = os.path.join(EXAMPLE_DIR, 'inversion_config.yaml')
SWEEP_FILE = os.path.join(OUTPUT_DIR, 'sweep.npz')
SWEEP_FINGERPRINT = os.path.join(OUTPUT_DIR, 'sweep_fingerprint.json')
SYNTH_CSV = os.path.join(OUTPUT_DIR, 'synthetic_observations.csv')
INVERSION_DIR = os.path.join(OUTPUT_DIR, 'inversion_results')


# ===================================================================
# Sweep config fingerprinting
# ===================================================================
def _config_fingerprint(config_path):
    """SHA-256 hash of the sweep config file contents."""
    with open(config_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _sweep_needs_regeneration():
    """Return True if the sweep must be (re)generated."""
    if not os.path.isfile(SWEEP_FILE):
        return True
    if not os.path.isfile(SWEEP_FINGERPRINT):
        return True
    current = _config_fingerprint(SWEEP_CONFIG)
    with open(SWEEP_FINGERPRINT, 'r') as f:
        saved = json.load(f).get('hash')
    return current != saved


def _save_sweep_fingerprint():
    """Write the current config hash next to the sweep file."""
    fp = _config_fingerprint(SWEEP_CONFIG)
    with open(SWEEP_FINGERPRINT, 'w') as f:
        json.dump({'hash': fp, 'config': SWEEP_CONFIG}, f)


# ===================================================================
# Synthetic profiles
# ===================================================================
def adiabat_C(depth_km, pot_temp=1300.0, grad=0.4):
    """Mantle adiabat: T = T_pot + grad * z."""
    return pot_temp + grad * depth_km


def melt_fraction(depth_km, onset_km=80.0, max_phi=0.01):
    """Linear melt above *onset_km*; zero below."""
    phi = np.where(depth_km < onset_km,
                   max_phi * (1.0 - depth_km / onset_km), 0.0)
    return np.clip(phi, 0, max_phi)


def grain_size_um(depth_km, gs_um=10000.0):
    """Constant grain size (default 1 cm)."""
    return np.full_like(depth_km, gs_um, dtype=float)


# ===================================================================
# Step 2 helper: forward-model Vs directly from VBR core
# ===================================================================
def _generate_synthetic_csv(sweep_file, csv_out, method='xfit_premelt'):
    """Compute exact synthetic Vs and Q via VBR core for the true profile.

    Instead of nearest-neighbour lookup in the sweep grid, this runs the
    full VBR calculation (elastic → viscous → anelastic) at the exact
    true T, phi, and gs values for each depth.  The sweep file is still
    loaded to obtain the depth/pressure/density/frequency grids (which
    must be consistent with the sweep that the inversion will use).
    """
    sys.path.insert(0, REPO_ROOT)
    from vbrc_V2Tpy.bayesian_fitting_py.fitting import load_sweep_data
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.core import VBR, StateVariables
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.thermal import calculate_solidus_K
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.params import C2K
    from vbrc_V2Tpy.bayesian_fitting_py.vbr.generate_sweep import load_sweep_params_from_yaml

    # Load sweep for its depth / pressure / density / frequency grids
    print(f"Loading sweep from {sweep_file} ...")
    sweep = load_sweep_data(sweep_file)

    z_m = np.atleast_1d(sweep['z'])
    z_km = z_m / 1e3
    n_z = len(z_km)
    P_GPa = sweep['P_GPa']
    rho = sweep['rho']  # density at each depth (kg/m³)
    density_model = sweep.get('density_model', 'constant')

    # Load sweep config to get frequency, solidus, and elastic settings
    sweep_params = load_sweep_params_from_yaml(SWEEP_CONFIG)
    f = np.logspace(sweep_params.freq_log_min, sweep_params.freq_log_max,
                    sweep_params.n_freq)

    T_true = adiabat_C(z_km)
    phi_true = melt_fraction(z_km)
    gs_true = grain_size_um(z_km)

    Vs_syn = np.zeros(n_z)
    Q_syn = np.zeros(n_z)

    print(f"Computing exact Vs/Q via VBR core at {n_z} depths ...")
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
        # V shape: (1, nfreq) — mean over frequency, convert m/s → km/s
        Vs_syn[iz] = float(np.mean(result['V']) / 1e3)
        Q_syn[iz] = float(np.mean(result['Q']))

    # Write CSV in the format expected by csv_model location_mode:
    #   lon, lat, depth, vs, q
    # We use dummy lon/lat = 0 since this is a 1-D validation.
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, 'w') as f:
        f.write('lon,lat,depth,vs,q\n')
        for iz in range(n_z):
            f.write(f'0.0,0.0,{z_km[iz]:.4f},{Vs_syn[iz]:.6f},{Q_syn[iz]:.2f}\n')

    print(f"Synthetic CSV ({n_z} depths) written to {csv_out}")
    print(f"  Vs range: {Vs_syn.min():.3f} – {Vs_syn.max():.3f} km/s")
    print(f"  Q  range: {Q_syn.min():.1f} – {Q_syn.max():.1f}")

    return z_km, T_true, phi_true, gs_true, Vs_syn, Q_syn


# ===================================================================
# Step 4 helper: comparison plots
# ===================================================================
def _make_comparison_plots(z_km, T_true, phi_true, gs_true, Vs_syn, Q_syn,
                           ml_csv, method='xfit_premelt'):
    """Plot recovered vs true profiles, plus per-depth posteriors."""
    import pandas as pd

    df = pd.read_csv(ml_csv)
    # The CSV has columns like: name, lat, lon, z, z_min, z_max,
    #   T_ml, T_mean, T_std, phi_ml, phi_mean, gs_ml, gs_mean, ...
    # Filter to the method we care about
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

    # --- Summary statistics ---
    # Match depths between true and recovered
    T_true_matched = np.interp(z_inv, z_km, T_true)
    dT = T_MAP - T_true_matched
    rmse_T = float(np.sqrt(np.nanmean(dT ** 2)))
    mae_T = float(np.nanmean(np.abs(dT)))
    print(f"\nRound-trip temperature recovery:")
    print(f"  RMSE = {rmse_T:.1f} °C    MAE = {mae_T:.1f} °C")

    # --- Figure ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 8), sharey=True)

    # Panel 1: Temperature
    ax = axes[0]
    ax.fill_betweenx(z_inv, T_mean - T_std, T_mean + T_std,
                     alpha=0.2, color='C0')
    ax.plot(T_true, z_km, 'k-', lw=2, label='True T (adiabat)')
    ax.plot(T_MAP, z_inv, 'C0-', lw=1.5, label='MAP')
    ax.plot(T_mean, z_inv, 'C0--', lw=1.5, label='Mean')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Depth (km)')
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_title('Temperature')

    # Panel 2: Vs
    ax = axes[1]
    ax.plot(Vs_syn, z_km, 'k-', lw=2, label='Synthetic Vs')
    if Vs_pred is not None:
        ax.plot(Vs_pred, z_inv, 'C1-', lw=1.5, label='Predicted (MAP)')
    ax.set_xlabel('Vs (km/s)')
    ax.legend(fontsize=8)
    ax.set_title('Shear Velocity')

    # Panel 3: Melt fraction
    ax = axes[2]
    phi_true_matched = np.interp(z_inv, z_km, phi_true)
    ax.plot(phi_true * 100, z_km, 'k-', lw=2, label='True φ')
    ax.plot(phi_MAP * 100, z_inv, 'C2-', lw=1.5, label='MAP')
    ax.plot(phi_mean * 100, z_inv, 'C2--', lw=1.5, label='Mean')
    ax.set_xlabel('Melt Fraction (%)')
    ax.legend(fontsize=8)
    ax.set_title('Melt Fraction')

    # Panel 4: Grain size
    ax = axes[3]
    ax.plot(gs_true / 1000, z_km, 'k-', lw=2, label='True d')
    ax.plot(gs_MAP_mm, z_inv, 'C3-', lw=1.5, label='MAP')
    ax.plot(gs_mean_mm, z_inv, 'C3--', lw=1.5, label='Mean')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.set_title('Grain Size')

    fig.suptitle(f'Round-Trip Validation (standalone workflow) – {method}\n'
                 f'RMSE(T) = {rmse_T:.1f} °C   MAE(T) = {mae_T:.1f} °C',
                 fontsize=13)
    fig.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR, f'roundtrip_comparison_{method}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison figure saved to {fig_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    os.chdir(REPO_ROOT)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    python = sys.executable

    # ------------------------------------------------------------------
    # Step 1: Generate the parameter sweep
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Generate parameter sweep (look-up table)")
    print("=" * 70, flush=True)
    if _sweep_needs_regeneration():
        if os.path.isfile(SWEEP_FILE):
            print("  Sweep config changed — regenerating ...")
        cmd = [
            python, '-m',
            'vbrc_V2Tpy.bayesian_fitting_py.vbr.generate_sweep',
            '--config', SWEEP_CONFIG,
        ]
        print(f"  Running: {' '.join(cmd)}\n", flush=True)
        subprocess.run(cmd, check=True)
        _save_sweep_fingerprint()
    else:
        print(f"  Sweep up-to-date at {SWEEP_FILE} — skipping generation.")

    # ------------------------------------------------------------------
    # Step 2: Create synthetic observations from a known adiabat
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Forward-model synthetic Vs from known adiabat")
    print("=" * 70, flush=True)
    z_km, T_true, phi_true, gs_true, Vs_syn, Q_syn = \
        _generate_synthetic_csv(SWEEP_FILE, SYNTH_CSV)

    # ------------------------------------------------------------------
    # Step 3: Run Bayesian inversion via the standard CLI
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Run Bayesian inversion on synthetic observations")
    print("=" * 70)
    cmd = [
        python, '-m',
        'vbrc_V2Tpy.bayesian_fitting_py',
        '--config', INVERSION_CONFIG, '--parallel', '16',
    ]
    print(f"  Running: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------
    # Step 4: Compare recovered parameters against truth
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Compare recovered parameters against known truth")
    print("=" * 70, flush=True)
    ml_csv = os.path.join(INVERSION_DIR, 'ml_estimates.csv')
    if not os.path.isfile(ml_csv):
        # Try the default name pattern
        ml_csv = os.path.join(INVERSION_DIR, 'ml_estimates.csv')
    if os.path.isfile(ml_csv):
        _make_comparison_plots(z_km, T_true, phi_true, gs_true, Vs_syn, Q_syn,
                               ml_csv)
    else:
        print(f"  WARNING: ML estimates CSV not found at {ml_csv}")
        print("  Skipping comparison plots.")

    print("\n" + "=" * 70)
    print("Done.  Outputs are in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
