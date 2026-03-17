#!/usr/bin/env python3
"""
Validation Case 2 – Known-adiabat round-trip test (self-contained).

Starts from a prescribed temperature adiabat, melt-fraction profile, and
grain-size profile.  Generates a sweep internally for the relevant depth
range, forward-models Vs through the sweep, then inverts those synthetic
velocities back through the Bayesian machinery to recover the original
parameters.  Compares the recovered values against the known inputs and
plots the results.

Usage
-----
CLI::

    python -m vbrc_V2Tpy.validation.validate_roundtrip

Python::

    from vbrc_V2Tpy.validation.validate_roundtrip import validate_roundtrip
    results = validate_roundtrip()
"""

import os
import json
import hashlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Callable, List

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------
from ..bayesian_fitting_py.fitting import (
    load_sweep_data,
    extract_ml_estimates,
    GrainSizePrior,
)
from ..bayesian_fitting_py.prior import (
    make_param_grid,
    prep_gs_lognormal,
    prior_model_probs,
)
from ..bayesian_fitting_py.probability import probability_distributions
from ..bayesian_fitting_py.vbr.thermal import solidus
from ..bayesian_fitting_py.vbr.params import C2K
from ..bayesian_fitting_py.vbr.generate_sweep import (
    SweepParams,
    generate_parameter_sweep,
    save_sweep,
)
from ..bayesian_fitting_py.plotting import plot_tradeoffs_posterior


# ---------------------------------------------------------------------------
# Default synthetic profiles
# ---------------------------------------------------------------------------
def default_adiabat_C(depth_km: np.ndarray) -> np.ndarray:
    """
    Simple mantle adiabat: T(z) = T_pot + (dT/dz)*z.

    Uses potential temperature 1350 °C and gradient 0.4 °C/km.
    """
    return 1350.0 + 0.4 * depth_km


def default_melt_fraction(depth_km: np.ndarray,
                          onset_km: float = 80.0,
                          max_phi: float = 0.01) -> np.ndarray:
    """
    Melt fraction that increases linearly above *onset_km* depth.

    Below *onset_km* → 0.  At the surface → *max_phi*.
    For deeper regions (> onset_km), φ = 0.
    """
    phi = np.where(depth_km < onset_km,
                   max_phi * (1.0 - depth_km / onset_km), 0.0)
    return np.clip(phi, 0, max_phi)


def default_grain_size_um(depth_km: np.ndarray,
                          gs_um: float = 10000.0) -> np.ndarray:
    """Constant grain size (default 1 cm = 10 000 µm)."""
    return np.full_like(depth_km, gs_um, dtype=float)


# ---------------------------------------------------------------------------
# Forward model: look up synthetic Vs/Q from the sweep
# ---------------------------------------------------------------------------
def _lookup_from_sweep(sweep, method, T_C, phi, gs_um, iz):
    """
    Return (Vs, Q) at a single depth index by nearest-neighbour lookup
    in the pre-computed sweep.
    """
    T_arr = sweep['T']
    phi_arr = sweep['phi']
    gs_arr = sweep['gs']

    i_T = int(np.argmin(np.abs(T_arr - T_C)))
    i_phi = int(np.argmin(np.abs(phi_arr - phi)))
    i_gs = int(np.argmin(np.abs(gs_arr - gs_um)))

    Vs = float(sweep['Box'][method]['meanVs'][i_T, i_phi, i_gs, iz])
    Q = float(sweep['Box'][method]['meanQ'][i_T, i_phi, i_gs, iz])
    return Vs, Q


# ---------------------------------------------------------------------------
# Sweep fingerprinting for caching
# ---------------------------------------------------------------------------
def _sweep_fingerprint(params: SweepParams) -> str:
    """Return a deterministic hash of the sweep parameters."""
    key_parts = [
        params.T.tobytes(),
        params.phi.tobytes(),
        params.gs.tobytes(),
        str(params.z_min).encode(),
        str(params.z_max).encode(),
        str(params.n_z).encode(),
        params.density_model.encode(),
        params.elastic_method.encode(),
        params.solidus_method.encode(),
        ','.join(params.anelastic_methods).encode(),
    ]
    h = hashlib.sha256()
    for part in key_parts:
        h.update(part)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------
def validate_roundtrip(
    anelastic_method: str = 'xfit_premelt',
    elastic_method: str = 'cammarano2003',
    density_model: str = 'prem',
    solidus_method: str = 'yk2001',
    gs_prior_case: str = 'log_uniform',
    sigma_vs: float = 0.05,
    T_profile: Optional[Callable] = None,
    phi_profile: Optional[Callable] = None,
    gs_profile: Optional[Callable] = None,
    z_min_km: float = 10.0,
    z_max_km: float = 660.0,
    n_z: int = 131,
    T_min_C: float = 800.0,
    T_max_C: float = 2500.0,
    T_step_C: float = 20.0,
    phi_max_grid: float = 0.05,
    n_phi: int = 21,
    gs_min_um: float = 100.0,
    gs_max_um: float = 30000.0,
    n_gs: int = 25,
    anelastic_methods: Optional[List[str]] = None,
    output_dir: str = 'validation_roundtrip',
    show: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Self-contained round-trip validation: generate sweep → forward Vs → invert → compare.

    Parameters
    ----------
    anelastic_method : str
        Anelastic method used for inversion.
    elastic_method : str
        Elastic method for sweep generation ('anharmonic' or 'cammarano2003').
    density_model : str
        Density model for sweep ('constant', 'prem', 'custom').
    solidus_method : str
        Solidus parameterization ('hirschmann', 'katz', 'yk2001').
    gs_prior_case : str
        Grain-size prior case.
    sigma_vs : float
        Assumed Vs uncertainty (km/s).
    T_profile : callable(depth_km) → T_C, optional
        Temperature profile in °C. Default: 1350 °C potential + 0.4 °C/km.
    phi_profile : callable(depth_km) → φ, optional
        Melt-fraction profile. Default: zero everywhere below 80 km.
    gs_profile : callable(depth_km) → gs_µm, optional
        Grain-size profile in µm. Default: constant 10 000 µm (1 cm).
    z_min_km, z_max_km : float
        Depth range for the sweep (km).
    n_z : int
        Number of depth points.
    T_min_C, T_max_C, T_step_C : float
        Temperature grid bounds and step (°C).
    phi_max_grid : float
        Maximum melt fraction in sweep grid.
    n_phi : int
        Number of melt-fraction grid points.
    gs_min_um, gs_max_um : float
        Grain-size grid bounds (µm).
    n_gs : int
        Number of grain-size grid points.
    anelastic_methods : list of str, optional
        Methods for sweep generation. Defaults to [anelastic_method].
    output_dir : str
        Directory for outputs.
    show : bool
        Show figure interactively.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Arrays keyed by 'depth_km', 'T_true_C', 'T_MAP_C', 'T_mean_C',
        'T_std_C', 'phi_true', 'phi_MAP', 'gs_true_um', 'gs_MAP_um',
        'Vs_synthetic', 'Vs_pred', 'Q_synthetic'.
    """
    if T_profile is None:
        T_profile = default_adiabat_C
    if phi_profile is None:
        phi_profile = default_melt_fraction
    if gs_profile is None:
        gs_profile = default_grain_size_um
    if anelastic_methods is None:
        anelastic_methods = [anelastic_method]
    elif anelastic_method not in anelastic_methods:
        anelastic_methods = [anelastic_method] + anelastic_methods

    # --- Build sweep parameter grids ---
    T_grid_base = np.arange(T_min_C, T_max_C + T_step_C / 2, T_step_C)
    phi_grid_base = np.linspace(0, phi_max_grid, n_phi)
    gs_grid_base = np.logspace(np.log10(gs_min_um), np.log10(gs_max_um), n_gs)

    # Ensure true profile values sit exactly on the grid so that the
    # forward lookup and the inversion use identical nodes.
    z_km_tmp = np.linspace(z_min_km, z_max_km, n_z)
    T_true_tmp = T_profile(z_km_tmp)
    phi_true_tmp = phi_profile(z_km_tmp)
    gs_true_tmp = gs_profile(z_km_tmp)

    # Erin changes for testing 
    T_grid = T_grid_base #np.union1d(T_grid_base, np.unique(np.round(T_true_tmp / T_step_C) * T_step_C))
    phi_grid = phi_grid_base #np.union1d(phi_grid_base, np.unique(phi_true_tmp))
    gs_grid = gs_grid_base #np.sort(np.unique(np.concatenate([gs_grid_base, np.unique(gs_true_tmp)])))

    sweep_params = SweepParams(
        T=T_grid,
        phi=phi_grid,
        gs=gs_grid,
        z_min=z_min_km,
        z_max=z_max_km,
        n_z=n_z,
        density_model=density_model,
        elastic_method=elastic_method,
        solidus_method=solidus_method,
        anelastic_methods=anelastic_methods,
    )

    # --- Generate or load cached sweep ---
    os.makedirs(output_dir, exist_ok=True)
    sweep_file = os.path.join(output_dir, 'roundtrip_sweep.npz')
    fingerprint = _sweep_fingerprint(sweep_params)
    fp_file = os.path.join(output_dir, 'roundtrip_sweep_fingerprint.json')

    if os.path.isfile(sweep_file) and os.path.isfile(fp_file):
        with open(fp_file, 'r') as f:
            saved_fp = json.load(f)
        if saved_fp.get('hash') == fingerprint:
            if verbose:
                print(f"Loading cached sweep from {sweep_file} ...")
            sweep = load_sweep_data(sweep_file)
        else:
            if verbose:
                print("Sweep parameters changed — regenerating ...")
            sweep = generate_parameter_sweep(sweep_params, verbose=verbose)
            save_sweep(sweep, sweep_file, verbose=verbose)
            with open(fp_file, 'w') as f:
                json.dump({'hash': fingerprint}, f)
    else:
        if verbose:
            print("Generating sweep for round-trip validation ...")
        sweep = generate_parameter_sweep(sweep_params, verbose=verbose)
        save_sweep(sweep, sweep_file, verbose=verbose)
        with open(fp_file, 'w') as f:
            json.dump({'hash': fingerprint}, f)
    # --- LUT plots ---
    lut_dir = os.path.join(output_dir, 'lut_plots')
    from ..bayesian_fitting_py.vbr.plot_lut import generate_sweep_lut_plots
    if verbose:
        print(f"Generating LUT plots ...")
    generate_sweep_lut_plots(sweep, lut_dir, every_n=max(1, n_z // 10),
                             verbose=verbose)

    z_m = np.atleast_1d(sweep['z'])
    z_km = z_m / 1e3
    P_GPa = np.atleast_1d(sweep['P_GPa'])
    n_z = len(z_km)

    # --- Build known input profiles ---
    T_true = T_profile(z_km)
    phi_true = phi_profile(z_km)
    gs_true = gs_profile(z_km)

    # --- Forward model: look up synthetic Vs/Q from sweep ---
    Vs_syn = np.zeros(n_z)
    Q_syn = np.zeros(n_z)
    for iz in range(n_z):
        Vs_syn[iz], Q_syn[iz] = _lookup_from_sweep(
            sweep, anelastic_method, T_true[iz], phi_true[iz], gs_true[iz], iz
        )
    if verbose:
        print(f"Synthetic Vs range: {Vs_syn.min():.3f} – {Vs_syn.max():.3f} km/s")
        print(f"Synthetic Q range: {Q_syn.min():.1f} – {Q_syn.max():.1f}")

    # --- Grain-size prior ---
    gs_prior = GrainSizePrior()
    if gs_prior_case == 'log_uniform':
        gs_prior.gs_pdf_type = 'uniform_log'
    elif gs_prior_case == 'log_normal_1mm':
        gs_prior.gs_pdf_type = 'lognormal'
        gs_prior.gs_mean = 1000.0
        gs_prior.gs_std = 0.25
    elif gs_prior_case == 'log_normal_1cm':
        gs_prior.gs_pdf_type = 'lognormal'
        gs_prior.gs_mean = 10000.0
        gs_prior.gs_std = 0.25
    else:
        gs_prior.gs_pdf_type = 'uniform_log'

    # --- Invert each depth ---
    T_MAP = np.full(n_z, np.nan)
    T_mean = np.full(n_z, np.nan)
    T_std = np.full(n_z, np.nan)
    phi_MAP = np.full(n_z, np.nan)
    phi_mean = np.full(n_z, np.nan)
    gs_MAP = np.full(n_z, np.nan)
    gs_mean = np.full(n_z, np.nan)
    Vs_pred = np.full(n_z, np.nan)
    Vs_pred_mean = np.full(n_z, np.nan)
    solidus_C = np.full(n_z, np.nan)

    if verbose:
        print(f"Inverting {n_z} depth levels ...")

    for iz in range(n_z):
        obs_vs = Vs_syn[iz]

        meanVs = sweep['Box'][anelastic_method]['meanVs'][:, :, :, iz]

        local_sweep = {
            'T': sweep['T'], 'phi': sweep['phi'], 'gs': sweep['gs'],
            'state_names': ['T', 'phi', 'gs'],
            'gs_params': sweep['gs_params'],
            'meanVs': meanVs,
        }
        if 'meanEta' in sweep['Box'][anelastic_method]:
            local_sweep['meanEta'] = sweep['Box'][anelastic_method]['meanEta'][:, :, :, iz]

        params = make_param_grid(local_sweep['state_names'], local_sweep)
        if gs_prior.gs_mean is not None:
            params['gs_mean'] = gs_prior.gs_mean
        if gs_prior.gs_std is not None:
            params['gs_std'] = gs_prior.gs_std
        if gs_prior.gs_pdf_type is not None:
            params['gs_pdf_type'] = gs_prior.gs_pdf_type

        gs_lognormal = False
        if params.get('gs_pdf_type') in ['lognormal', 'uniform_log']:
            gs_lognormal = True
            if params.get('gs_pdf_type') == 'lognormal':
                params = prep_gs_lognormal(params, local_sweep)
            local_sweep['gs'] = local_sweep['gs'] / local_sweep['gs_params']['gsref']
            params['gs'] = params['gs'] / local_sweep['gs_params']['gsref']

        prior_sv, _ = prior_model_probs(params, local_sweep['state_names'])

        if gs_lognormal:
            local_sweep['gs'] = local_sweep['gs'] * local_sweep['gs_params']['gsref']
            params['gs'] = params['gs'] * local_sweep['gs_params']['gsref']

        likelihood = probability_distributions(
            'likelihood from residuals', obs_vs, sigma_vs, meanVs
        )
        pS = probability_distributions('A|B', likelihood, prior_sv, 1.0)

        posterior = {
            'pS': pS,
            'state_names': ['T', 'phi', 'gs'],
            'T': sweep['T'], 'phi': sweep['phi'], 'gs': sweep['gs'],
        }
        ml = extract_ml_estimates(posterior, local_sweep, anelastic_method)

        T_MAP[iz] = ml['T']['ml']
        T_mean[iz] = ml['T']['mean']
        T_std[iz] = ml['T']['std']
        phi_MAP[iz] = ml['phi']['ml']
        phi_mean[iz] = ml['phi']['mean']
        gs_MAP[iz] = ml['gs']['ml']
        gs_mean[iz] = ml['gs']['mean']
        if 'predicted_Vs' in ml:
            Vs_pred[iz] = ml['predicted_Vs']

        # Vs at marginal-mean parameters (nearest-neighbour lookup)
        i_T_m = int(np.argmin(np.abs(sweep['T'] - T_mean[iz])))
        i_phi_m = int(np.argmin(np.abs(sweep['phi'] - phi_mean[iz])))
        i_gs_m = int(np.argmin(np.abs(sweep['gs'] - gs_mean[iz])))
        Vs_pred_mean[iz] = float(meanVs[i_T_m, i_phi_m, i_gs_m])

        sol = solidus(
            P_GPa[iz] * 1e9, method=solidus_method,
            depth_km=z_km[iz], density_model=density_model,
        )
        solidus_C[iz] = float(sol['Tsol'])

        # Posterior distribution plot
        post_dir = os.path.join(output_dir, 'posteriors')
        os.makedirs(post_dir, exist_ok=True)
        post_path = os.path.join(
            post_dir, f'posterior_{z_km[iz]:.0f}km.png'
        )
        fig_post = plot_tradeoffs_posterior(
            pS, local_sweep,
            f'Vs={obs_vs:.3f} km/s, z={z_km[iz]:.1f} km',
            anelastic_method, save_path=post_path,
        )
        plt.close(fig_post)

        if verbose and (iz % 10 == 0 or iz == n_z - 1):
            dT = T_MAP[iz] - T_true[iz]
            print(f"  z={z_km[iz]:6.1f} km  T_true={T_true[iz]:7.1f}  "
                  f"T_MAP={T_MAP[iz]:7.1f}  ΔT={dT:+6.1f} °C  "
                  f"Vs_syn={obs_vs:.3f}")

    # --- Summary statistics ---
    dT = T_MAP - T_true
    rmse_T = float(np.sqrt(np.nanmean(dT ** 2)))
    mae_T = float(np.nanmean(np.abs(dT)))
    if verbose:
        print(f"\nRound-trip temperature recovery:")
        print(f"  RMSE = {rmse_T:.1f} °C    MAE = {mae_T:.1f} °C")

    # --- Figure ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 8), sharey=True)

    # Panel 1: Temperature
    ax = axes[0]
    ax.fill_betweenx(z_km, T_mean - T_std, T_mean + T_std,
                     alpha=0.2, color='C0')
    ax.plot(T_true, z_km, 'k-', lw=2, label='True T (adiabat)')
    ax.plot(T_MAP, z_km, 'C0-', lw=1.5, label='MAP')
    ax.plot(T_mean, z_km, 'C0--', lw=1.5, label='Mean')
    ax.plot(solidus_C, z_km, 'r:', lw=1, label='Solidus')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Depth (km)')
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_title('Temperature')

    # Panel 2: Vs
    ax = axes[1]
    ax.plot(Vs_syn, z_km, 'k-', lw=2, label='Synthetic Vs')
    ax.plot(Vs_pred, z_km, 'C1-', lw=1.5, label='MAP')
    ax.plot(Vs_pred_mean, z_km, 'C1--', lw=1.5, label='Mean')
    ax.set_xlabel('Vs (km/s)')
    ax.legend(fontsize=8)
    ax.set_title('Shear Velocity')

    # Panel 3: Melt fraction
    ax = axes[2]
    ax.plot(phi_true * 100, z_km, 'k-', lw=2, label='True φ')
    ax.plot(phi_MAP * 100, z_km, 'C2-', lw=1.5, label='MAP')
    ax.plot(phi_mean * 100, z_km, 'C2--', lw=1.5, label='Mean')
    ax.set_xlabel('Melt Fraction (%)')
    ax.legend(fontsize=8)
    ax.set_title('Melt Fraction')

    # Panel 4: Grain size
    ax = axes[3]
    ax.plot(gs_true / 1000, z_km, 'k-', lw=2, label='True d')
    ax.plot(gs_MAP / 1000, z_km, 'C3-', lw=1.5, label='MAP')
    ax.plot(gs_mean / 1000, z_km, 'C3--', lw=1.5, label='Mean')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.set_title('Grain Size')

    fig.suptitle(f'Round-Trip Validation – {anelastic_method}\n'
                 f'RMSE(T) = {rmse_T:.1f} °C   MAE(T) = {mae_T:.1f} °C',
                 fontsize=13)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f'roundtrip_{anelastic_method}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"Figure saved to {fig_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # --- CSV ---
    csv_path = os.path.join(output_dir, f'roundtrip_{anelastic_method}.csv')
    header = ('depth_km,T_true_C,T_MAP_C,T_mean_C,T_std_C,'
              'phi_true,phi_MAP,phi_mean,gs_true_um,gs_MAP_um,gs_mean_um,'
              'Vs_synthetic,Vs_pred_MAP,Vs_pred_mean,Q_synthetic,solidus_C')
    np.savetxt(csv_path,
               np.column_stack([z_km, T_true, T_MAP, T_mean, T_std,
                                phi_true, phi_MAP, phi_mean,
                                gs_true, gs_MAP, gs_mean,
                                Vs_syn, Vs_pred, Vs_pred_mean,
                                Q_syn, solidus_C]),
               header=header, delimiter=',', fmt='%.4f', comments='')
    if verbose:
        print(f"CSV saved to {csv_path}")

    return {
        'depth_km': z_km,
        'T_true_C': T_true,
        'T_MAP_C': T_MAP,
        'T_mean_C': T_mean,
        'T_std_C': T_std,
        'phi_true': phi_true,
        'phi_MAP': phi_MAP,
        'phi_mean': phi_mean,
        'gs_true_um': gs_true,
        'gs_MAP_um': gs_MAP,
        'gs_mean_um': gs_mean,
        'Vs_synthetic': Vs_syn,
        'Vs_pred': Vs_pred,
        'Vs_pred_mean': Vs_pred_mean,
        'Q_synthetic': Q_syn,
        'solidus_C': solidus_C,
        'rmse_T': rmse_T,
        'mae_T': mae_T,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Validation Case 2: Self-contained round-trip test with known adiabat',
    )
    # Sweep generation parameters
    parser.add_argument('--z-min', type=float, default=10.0,
                        help='Min depth (km, default: 10)')
    parser.add_argument('--z-max', type=float, default=660.0,
                        help='Max depth (km, default: 660)')
    parser.add_argument('--n-z', type=int, default=131,
                        help='Number of depth points (default: 131)')
    parser.add_argument('--T-min', type=float, default=800.0,
                        help='Min temperature grid (°C, default: 800)')
    parser.add_argument('--T-max', type=float, default=2500.0,
                        help='Max temperature grid (°C, default: 2500)')
    parser.add_argument('--T-step', type=float, default=20.0,
                        help='Temperature grid step (°C, default: 20)')
    parser.add_argument('--elastic', default='cammarano2003',
                        choices=['anharmonic', 'cammarano2003'],
                        help='Elastic method (default: cammarano2003)')
    parser.add_argument('--density', default='prem',
                        choices=['constant', 'prem'],
                        help='Density model (default: prem)')
    parser.add_argument('--solidus', default='yk2001',
                        choices=['hirschmann', 'katz', 'yk2001'],
                        help='Solidus method (default: yk2001)')
    # Inversion parameters
    parser.add_argument('--method', '-m', default='xfit_premelt',
                        help='Anelastic method (default: xfit_premelt)')
    parser.add_argument('--gs-prior', default='log_uniform',
                        choices=['log_uniform', 'log_normal_1mm', 'log_normal_1cm'])
    parser.add_argument('--sigma-vs', type=float, default=0.05,
                        help='Vs uncertainty (km/s)')
    # Adiabat / profile parameters
    parser.add_argument('--pot-temp', type=float, default=1300.0,
                        help='Potential temperature for adiabat (°C)')
    parser.add_argument('--grad', type=float, default=0.4,
                        help='Adiabatic gradient (°C/km)')
    parser.add_argument('--phi-max', type=float, default=0.01,
                        help='Maximum melt fraction (default: 0.01)')
    parser.add_argument('--phi-onset', type=float, default=80.0,
                        help='Depth of melt onset (km)')
    parser.add_argument('--gs-um', type=float, default=10000.0,
                        help='Constant grain size (µm)')
    parser.add_argument('--output', '-o', default='validation_roundtrip')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    # Build profile callables from CLI args
    def T_func(z): return args.pot_temp + args.grad * z
    def phi_func(z): return default_melt_fraction(z, onset_km=args.phi_onset,
                                                  max_phi=args.phi_max)
    def gs_func(z): return default_grain_size_um(z, gs_um=args.gs_um)

    validate_roundtrip(
        anelastic_method=args.method,
        elastic_method=args.elastic,
        density_model=args.density,
        solidus_method=args.solidus,
        gs_prior_case=args.gs_prior,
        sigma_vs=args.sigma_vs,
        T_profile=T_func,
        phi_profile=phi_func,
        gs_profile=gs_func,
        z_min_km=args.z_min,
        z_max_km=args.z_max,
        n_z=args.n_z,
        T_min_C=args.T_min,
        T_max_C=args.T_max,
        T_step_C=args.T_step,
        output_dir=args.output,
        show=args.show,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
