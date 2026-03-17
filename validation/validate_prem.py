#!/usr/bin/env python3
"""
Validation Case 1 – PREM velocity inversion.

Inverts the shear-wave velocities of a reference 1-D Earth model (e.g. PREM)
through a pre-computed sweep to obtain a depth-dependent temperature profile.
Produces a figure showing recovered temperature vs depth alongside the
solidus.

Usage
-----
From the command line::

    python -m vbrc_V2Tpy.validation.validate_prem --sweep sweep.npz

Or from Python::

    from vbrc_V2Tpy.validation.validate_prem import validate_prem
    results = validate_prem('sweep.npz')
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Imports from the main package
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
from ..bayesian_fitting_py.vbr.thermal import solidus, calculate_solidus_K
from ..bayesian_fitting_py.vbr.generate_sweep import load_density_profile
from ..bayesian_fitting_py.plotting import plot_tradeoffs_posterior


# ---------------------------------------------------------------------------
# PREM Vs loader
# ---------------------------------------------------------------------------
def load_prem_vs() -> Dict[str, np.ndarray]:
    """
    Load PREM shear-wave velocity profile.

    Returns
    -------
    dict
        'depth_km', 'Vs_km_s' (sorted ascending in depth).
    """
    prem_csv = Path(__file__).parent / '..' / 'bayesian_fitting_py' / 'vbr' / 'PREM_for_VBRc.csv'
    prem_csv = prem_csv.resolve()
    data = np.genfromtxt(prem_csv, delimiter=',', skip_header=1)
    radius_m = data[:, 0]
    vs_m_s = data[:, 3]
    depth_m = 6_371_000.0 - radius_m

    idx = np.argsort(depth_m)
    depth_m = depth_m[idx]
    vs_m_s = vs_m_s[idx]

    # Average duplicate depths (PREM discontinuities)
    ud, inv = np.unique(depth_m, return_inverse=True)
    uvs = np.zeros(len(ud))
    for i in range(len(ud)):
        uvs[i] = np.mean(vs_m_s[inv == i])

    return {'depth_km': ud / 1e3, 'Vs_km_s': uvs / 1e3}


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------
def validate_prem(
    sweep_file: str,
    anelastic_method: str = 'xfit_premelt',
    gs_prior_case: str = 'log_normal_1cm',
    sigma_vs: float = 0.05,
    reference_model: str = 'prem',
    output_dir: str = 'validation_prem',
    show: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Invert PREM Vs through a sweep to recover temperature vs depth.

    Parameters
    ----------
    sweep_file : str
        Path to a pre-computed sweep (.npz / .mat / .pkl).
    anelastic_method : str
        Anelastic method to use for inversion.
    gs_prior_case : str
        Grain-size prior: 'log_uniform', 'log_normal_1mm', 'log_normal_1cm'.
    sigma_vs : float
        Assumed Vs uncertainty (km/s).
    reference_model : str
        Reference model ('prem').  Reserved for future expansion.
    output_dir : str
        Directory for output figure and CSV.
    show : bool
        Show the figure interactively.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        'depth_km', 'T_MAP_C', 'T_mean_C', 'T_std_C', 'Vs_obs', 'Vs_pred',
        'solidus_C', 'phi_MAP', 'gs_MAP_um'.
    """
    # --- Load sweep ---
    if verbose:
        print(f"Loading sweep from {sweep_file} ...")
    sweep = load_sweep_data(sweep_file)
    z_m = np.atleast_1d(sweep['z'])
    z_km = z_m / 1e3
    P_GPa = np.atleast_1d(sweep['P_GPa'])
    n_z = len(z_km)

    # --- Load PREM Vs and interpolate to sweep depths ---
    prem = load_prem_vs()
    Vs_at_z = np.interp(z_km, prem['depth_km'], prem['Vs_km_s'])

    # --- Set up grain-size prior ---
    gs_prior = GrainSizePrior()
    if gs_prior_case == 'log_uniform':
        gs_prior.gs_pdf_type = 'uniform_log'
    elif gs_prior_case == 'log_normal_1mm':
        gs_prior.gs_pdf_type = 'lognormal'
        gs_prior.gs_mean = 1000.0    # 1 mm in µm
        gs_prior.gs_std = 0.25
    elif gs_prior_case == 'log_normal_1cm':
        gs_prior.gs_pdf_type = 'lognormal'
        gs_prior.gs_mean = 10000.0   # 1 cm in µm
        gs_prior.gs_std = 0.25
    else:
        gs_prior.gs_pdf_type = 'uniform_log'

    # Determine solidus method from sweep metadata
    density_model = sweep.get('density_model', 'prem')
    # Inspect sweep config if available
    solidus_method = sweep.get('solidus_method', 'yk2001')

    # --- Results arrays ---
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
        print(f"Inverting {n_z} depth levels ({z_km[0]:.0f}–{z_km[-1]:.0f} km) ...")

    for iz in range(n_z):
        depth = z_km[iz]
        obs_vs = Vs_at_z[iz]

        # Extract Vs at this single depth
        meanVs = sweep['Box'][anelastic_method]['meanVs'][:, :, :, iz]
        meanQ = sweep['Box'][anelastic_method]['meanQ'][:, :, :, iz]

        # Build local sweep-like dict for prior machinery
        local_sweep = {
            'T': sweep['T'], 'phi': sweep['phi'], 'gs': sweep['gs'],
            'state_names': ['T', 'phi', 'gs'],
            'gs_params': sweep['gs_params'],
            'meanVs': meanVs,
        }
        if 'meanEta' in sweep['Box'][anelastic_method]:
            local_sweep['meanEta'] = sweep['Box'][anelastic_method]['meanEta'][:, :, :, iz]

        # Parameter grid & prior
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

        # Likelihood
        likelihood = probability_distributions(
            'likelihood from residuals', obs_vs, sigma_vs, meanVs
        )

        # Posterior
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

        # Predicted Vs at marginal-mean (T, phi, gs)
        i_T_mean = int(np.argmin(np.abs(sweep['T'] - ml['T']['mean'])))
        i_phi_mean = int(np.argmin(np.abs(sweep['phi'] - ml['phi']['mean'])))
        i_gs_mean = int(np.argmin(np.abs(sweep['gs'] - ml['gs']['mean'])))
        Vs_pred_mean[iz] = float(meanVs[i_T_mean, i_phi_mean, i_gs_mean])

        # Solidus
        sol = solidus(
            P_GPa[iz] * 1e9, method=solidus_method,
            depth_km=depth, density_model=density_model,
        )
        solidus_C[iz] = float(sol['Tsol'])

        # Posterior distribution plot
        post_dir = os.path.join(output_dir, 'posteriors')
        os.makedirs(post_dir, exist_ok=True)
        post_path = os.path.join(
            post_dir, f'posterior_{depth:.0f}km.png'
        )
        fig_post = plot_tradeoffs_posterior(
            pS, local_sweep,
            f'Vs={obs_vs:.3f} km/s, z={depth:.1f} km',
            anelastic_method, save_path=post_path,
        )
        plt.close(fig_post)

        if verbose and (iz % 10 == 0 or iz == n_z - 1):
            print(f"  depth={depth:6.1f} km  T_MAP={T_MAP[iz]:7.1f} °C  "
                  f"Vs_obs={obs_vs:.3f}  Vs_pred={Vs_pred[iz]:.3f}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 8), sharey=True)

    # Panel 1: Temperature
    ax = axes[0]
    ax.fill_betweenx(z_km, T_mean - T_std, T_mean + T_std,
                     alpha=0.2, color='C0')
    ax.plot(T_MAP, z_km, 'C0-', lw=2, label='T (joint MAP)')
    ax.plot(T_mean, z_km, 'C0--', lw=1.5, label='T (marginal mean)')
    ax.plot(solidus_C, z_km, 'r--', lw=1.5, label=f'Solidus ({solidus_method})')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Depth (km)')
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_title('Temperature')

    # Panel 2: Vs comparison
    ax = axes[1]
    ax.plot(Vs_at_z, z_km, 'k-', lw=1.5, label='PREM Vs')
    ax.plot(Vs_pred, z_km, 'C1-', lw=1.5, label='Vs from MAP')
    ax.plot(Vs_pred_mean, z_km, 'C1--', lw=1.5, label='Vs from mean')
    ax.set_xlabel('Vs (km/s)')
    ax.legend(fontsize=8)
    ax.set_title('Shear Velocity')

    # Panel 3: Melt fraction
    ax = axes[2]
    ax.plot(phi_MAP * 100, z_km, 'C2-', lw=1.5, label='φ MAP')
    ax.plot(phi_mean * 100, z_km, 'C2--', lw=1.5, label='φ mean')
    ax.set_xlabel('Melt Fraction (%)')
    ax.legend(fontsize=8)
    ax.set_title('Melt Fraction')

    # Panel 4: Grain size
    ax = axes[3]
    ax.plot(gs_MAP / 1000, z_km, 'C3-', lw=1.5, label='d MAP')
    ax.plot(gs_mean / 1000, z_km, 'C3--', lw=1.5, label='d mean')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.set_title('Grain Size')

    fig.suptitle(f'PREM Validation – {anelastic_method}  (σ_Vs = {sigma_vs} km/s)',
                 fontsize=13)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f'prem_validation_{anelastic_method}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    if verbose:
        print(f"\nFigure saved to {fig_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    # --- CSV ---
    csv_path = os.path.join(output_dir, f'prem_validation_{anelastic_method}.csv')
    header = ('depth_km,T_MAP_C,T_mean_C,T_std_C,phi_MAP,phi_mean,'
              'gs_MAP_um,gs_mean_um,Vs_obs,Vs_pred_MAP,Vs_pred_mean,solidus_C')
    np.savetxt(csv_path,
               np.column_stack([z_km, T_MAP, T_mean, T_std, phi_MAP, phi_mean,
                                gs_MAP, gs_mean, Vs_at_z, Vs_pred,
                                Vs_pred_mean, solidus_C]),
               header=header, delimiter=',', fmt='%.4f', comments='')
    if verbose:
        print(f"CSV saved to {csv_path}")

    return {
        'depth_km': z_km,
        'T_MAP_C': T_MAP,
        'T_mean_C': T_mean,
        'T_std_C': T_std,
        'phi_MAP': phi_MAP,
        'phi_mean': phi_mean,
        'gs_MAP_um': gs_MAP,
        'gs_mean_um': gs_mean,
        'Vs_obs': Vs_at_z,
        'Vs_pred_MAP': Vs_pred,
        'Vs_pred_mean': Vs_pred_mean,
        'solidus_C': solidus_C,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Validation Case 1: Invert PREM Vs to recover temperature profile',
    )
    parser.add_argument('--sweep', '-s', required=True,
                        help='Path to pre-computed sweep file (.npz / .mat)')
    parser.add_argument('--method', '-m', default='xfit_premelt',
                        help='Anelastic method (default: xfit_premelt)')
    parser.add_argument('--gs-prior', default='log_normal_1cm',
                        choices=['log_uniform', 'log_normal_1mm', 'log_normal_1cm'],
                        help='Grain-size prior (default: log_normal_1cm)')
    parser.add_argument('--sigma-vs', type=float, default=0.05,
                        help='Vs uncertainty in km/s (default: 0.05)')
    parser.add_argument('--output', '-o', default='validation_prem',
                        help='Output directory')
    parser.add_argument('--show', action='store_true',
                        help='Show figure interactively')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()
    validate_prem(
        args.sweep,
        anelastic_method=args.method,
        gs_prior_case=args.gs_prior,
        sigma_vs=args.sigma_vs,
        output_dir=args.output,
        show=args.show,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
