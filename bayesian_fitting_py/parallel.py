"""
Parallel processing support for large-scale Bayesian inversion.

Provides a worker function and dispatch logic for multiprocessing.
Each location is processed independently, so the workload is
embarrassingly parallel.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from .probability import probability_distributions
from .prior import make_param_grid, prep_gs_lognormal, prior_model_probs
from .fitting import extract_ml_estimates


# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------

def precompute_depth_averaged_sweep(
    sweep: Dict[str, Any],
    anelastic_method: str,
    z_range: Tuple[float, float],
) -> Dict[str, Any]:
    """
    Pre-compute depth-averaged Vs/Q/Eta grids for a given z-range.

    Returns a lightweight dict with only the arrays needed for fitting
    (no mutation of the original sweep).
    """
    z = sweep['z']
    depth_range_m = np.array(z_range) * 1e3
    z_inds = np.where((z >= depth_range_m[0]) & (z <= depth_range_m[1]))[0]

    box = sweep['Box'][anelastic_method]
    result = {
        'meanVs': np.mean(box['meanVs'][:, :, :, z_inds], axis=3),
        'meanQ': np.mean(box['meanQ'][:, :, :, z_inds], axis=3),
    }
    if 'meanEta' in box:
        result['meanEta'] = np.mean(box['meanEta'][:, :, :, z_inds], axis=3)
    return result


def precompute_prior(
    sweep: Dict[str, Any],
    grain_size_prior,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build the prior probability grid (same for every location).

    Returns (prior_statevars, params) without mutating sweep.
    """
    # Work on copies so the original sweep is not mutated
    sweep_copy = {
        k: (v.copy() if isinstance(v, np.ndarray) else v)
        for k, v in sweep.items()
        if k != 'Box'
    }
    sweep_copy['Box'] = sweep['Box']  # read-only reference is fine

    params = make_param_grid(sweep_copy['state_names'], sweep_copy)

    if grain_size_prior.gs_mean is not None:
        params['gs_mean'] = grain_size_prior.gs_mean
    if grain_size_prior.gs_std is not None:
        params['gs_std'] = grain_size_prior.gs_std
    if grain_size_prior.gs_pdf_type is not None:
        params['gs_pdf_type'] = grain_size_prior.gs_pdf_type

    gs_lognormal = False
    if params.get('gs_pdf_type') in ['lognormal', 'uniform_log']:
        gs_lognormal = True
        if params.get('gs_pdf_type') == 'lognormal':
            params = prep_gs_lognormal(params, sweep_copy)
        gsref = sweep_copy['gs_params']['gsref']
        sweep_copy['gs'] = sweep_copy['gs'] / gsref
        params['gs'] = params['gs'] / gsref

    prior_statevars, _ = prior_model_probs(params, sweep_copy['state_names'])

    if gs_lognormal:
        gsref = sweep_copy['gs_params']['gsref']
        sweep_copy['gs'] = sweep_copy['gs'] * gsref
        params['gs'] = params['gs'] * gsref

    return prior_statevars, params


# ---------------------------------------------------------------------------
# Single-location worker  (called from Pool or sequentially)
# ---------------------------------------------------------------------------

def _process_one_location(args: Tuple) -> Optional[Dict[str, Any]]:
    """
    Process a single location for one anelastic method.

    Designed to be picklable for multiprocessing — takes a single
    tuple argument and returns a results dict (or None on error).
    """
    (
        il,               # location index
        locname,
        lat, lon,
        z_min, z_max,
        use_vs, use_q,
        obs_vs, sigma_vs,
        obs_q, sigma_q,
        depth_key,        # (z_min, z_max) tuple used to look up pre-averaged grids
        precomputed,      # dict with 'meanVs', 'meanQ', 'meanEta' for this depth_key
        prior_statevars,  # 3-D prior array
        sweep_vectors,    # {'T': ..., 'phi': ..., 'gs': ..., 'state_names': ..., 'gs_params': ...}
        anelastic_method,
        save_ml_csv,
        has_depth_col,    # bool — whether to include 'z' in CSV record
        depth_val,        # actual depth value (or None)
        default_vs_error,
        default_q_error,
    ) = args

    try:
        # Honour the config-level obs_types flags (use_vs / use_q) so that
        # data presence alone does not override the user's intent.
        vs_exists = use_vs and obs_vs is not None and sigma_vs is not None
        q_exists = use_q and obs_q is not None and sigma_q is not None
        if not vs_exists and not q_exists:
            return None

        mean_vs = precomputed['meanVs'] if vs_exists else None
        mean_q = precomputed['meanQ'] if q_exists else None

        # --- likelihoods ---
        likelihood_vs = None
        likelihood_q = None
        if vs_exists:
            likelihood_vs = probability_distributions(
                'likelihood from residuals', obs_vs, sigma_vs, mean_vs
            )
        if q_exists:
            likelihood_q = probability_distributions(
                'likelihood from residuals', obs_q, sigma_q, mean_q
            )

        # --- posteriors ---
        if vs_exists:
            pS = probability_distributions('A|B', likelihood_vs, prior_statevars, 1.0)
        if q_exists:
            pS_q = probability_distributions('A|B', likelihood_q, prior_statevars, 1.0)
            if vs_exists:
                # Combined
                pS = probability_distributions(
                    'C|A,B conditionally independent',
                    likelihood_vs, likelihood_q, prior_statevars, 1.0,
                )
            else:
                pS = pS_q

        # Build a minimal "sweep" dict for extract_ml_estimates
        mini_sweep = dict(sweep_vectors)  # T, phi, gs, state_names, gs_params
        if mean_vs is not None:
            mini_sweep['meanVs'] = mean_vs
        if mean_q is not None:
            mini_sweep['meanQ'] = mean_q
        if 'meanEta' in precomputed:
            mini_sweep['meanEta'] = precomputed['meanEta']

        posterior = {
            'pS': pS,
            'state_names': sweep_vectors['state_names'],
        }
        for name in sweep_vectors['state_names']:
            posterior[name] = sweep_vectors[name]
        if vs_exists:
            posterior['obs_Vs'] = obs_vs
            posterior['sigma_Vs'] = sigma_vs
        if q_exists:
            posterior['obs_Q'] = obs_q
            posterior['sigma_Q'] = sigma_q

        ml_est = extract_ml_estimates(posterior, mini_sweep, anelastic_method)
        ml_est['lat'] = lat
        ml_est['lon'] = lon
        ml_est['z_min'] = z_min
        ml_est['z_max'] = z_max
        if vs_exists:
            ml_est['obs_Vs'] = obs_vs
            ml_est['sigma_Vs'] = sigma_vs
        if q_exists:
            ml_est['obs_Q'] = obs_q
            ml_est['sigma_Q'] = sigma_q

        # Build CSV record
        record = None
        if save_ml_csv:
            record = {
                'name': locname,
                'lat': lat,
                'lon': lon,
            }
            if has_depth_col and depth_val is not None:
                record['z'] = depth_val
            record['z_min'] = z_min
            record['z_max'] = z_max
            record['anelastic_method'] = anelastic_method
            record['T_ml'] = ml_est['T']['ml']
            record['T_std'] = ml_est['T']['std']
            record['T_mean'] = ml_est['T']['mean']
            record['phi_ml'] = ml_est['phi']['ml']
            record['phi_std'] = ml_est['phi']['std']
            record['phi_mean'] = ml_est['phi']['mean']
            record['gs_ml_mm'] = ml_est['gs']['ml_mm']
            record['gs_std_mm'] = ml_est['gs']['std_mm']
            record['gs_mean_mm'] = ml_est['gs']['mean_mm']
            if 'log10_eta' in ml_est:
                record['log10_eta_ml'] = ml_est['log10_eta']['ml']
                record['log10_eta_std'] = ml_est['log10_eta']['std']
                record['log10_eta_mean'] = ml_est['log10_eta']['mean']
            chi2_total = 0.0
            n_obs = 0
            if 'obs_Vs' in ml_est:
                record['Vs_obs'] = ml_est['obs_Vs']
                vs_pred = ml_est.get('predicted_Vs', np.nan)
                sig_vs = ml_est['sigma_Vs']
                record['Vs_pred'] = vs_pred
                record['Vs_misfit'] = (ml_est['obs_Vs'] - vs_pred) if not np.isnan(vs_pred) else np.nan
                if not np.isnan(vs_pred) and sig_vs > 0:
                    vs_chi2 = ((ml_est['obs_Vs'] - vs_pred) / sig_vs) ** 2
                    record['Vs_chi2'] = vs_chi2
                    chi2_total += vs_chi2
                    n_obs += 1
                else:
                    record['Vs_chi2'] = np.nan
            if 'obs_Q' in ml_est:
                record['Q_obs'] = ml_est['obs_Q']
                q_pred = ml_est.get('predicted_Q', np.nan)
                sig_q = ml_est['sigma_Q']
                record['Q_pred'] = q_pred
                record['Q_misfit'] = (ml_est['obs_Q'] - q_pred) if not np.isnan(q_pred) else np.nan
                if not np.isnan(q_pred) and sig_q > 0:
                    q_chi2 = ((ml_est['obs_Q'] - q_pred) / sig_q) ** 2
                    record['Q_chi2'] = q_chi2
                    chi2_total += q_chi2
                    n_obs += 1
                else:
                    record['Q_chi2'] = np.nan
            record['chi2_total'] = chi2_total if n_obs > 0 else np.nan

        # Marginal P(phi, T | S) for ensemble
        pS_norm = pS / np.sum(pS)
        p_joint = np.sum(pS_norm, axis=2)  # marginalise over grain size

        return {
            'il': il,
            'locname': locname,
            'ml_est': ml_est,
            'record': record,
            'p_joint': p_joint,
            'posterior_phi': posterior['phi'],
            'posterior_T': posterior['T'],
            'posterior': posterior,  # full dict for optional plot generation
        }

    except Exception as e:
        return None


def _process_one_location_indexed(indexed_args):
    """Wrapper that preserves index for imap_unordered ordering."""
    idx, args = indexed_args
    return idx, _process_one_location(args)


# ---------------------------------------------------------------------------
# Batch dispatcher
# ---------------------------------------------------------------------------

def run_locations_parallel(
    locations: List[Tuple[float, float]],
    names: List[str],
    z_ranges: List[Tuple[float, float]],
    seismic_model_data: Any,
    sweep: Dict[str, Any],
    anelastic_method: str,
    grain_size_prior,
    config,
    n_workers: int = 1,
    use_vs: bool = True,
    use_q: bool = True,
) -> List[Optional[Dict[str, Any]]]:
    """
    Process all locations for one anelastic method, optionally in parallel.

    Parameters
    ----------
    n_workers : int
        Number of worker processes (1 = sequential, >1 = multiprocessing).
    use_vs : bool
        Whether Vs observations should be used.
    use_q : bool
        Whether Q observations should be used.
    """
    if not use_vs and not use_q:
        raise ValueError("At least one of use_vs or use_q must be True")

    n_locations = len(locations)
    use_preloaded = seismic_model_data is not None
    has_depth_col = (use_preloaded and seismic_model_data.depths is not None)

    # ------------------------------------------------------------------
    # 1. Pre-compute depth-averaged grids for each unique z-range
    # ------------------------------------------------------------------
    unique_z = {}
    for z_min, z_max in z_ranges:
        key = (z_min, z_max)
        if key not in unique_z:
            unique_z[key] = precompute_depth_averaged_sweep(
                sweep, anelastic_method, key,
            )

    # ------------------------------------------------------------------
    # 2. Pre-compute the prior (same for all locations)
    # ------------------------------------------------------------------
    prior_statevars, params = precompute_prior(sweep, grain_size_prior)

    sweep_vectors = {
        'T': sweep['T'].copy(),
        'phi': sweep['phi'].copy(),
        'gs': sweep['gs'].copy(),
        'state_names': list(sweep['state_names']),
        'gs_params': dict(sweep['gs_params']),
    }

    # ------------------------------------------------------------------
    # 3. Build argument tuples for each location
    # ------------------------------------------------------------------
    job_args = []
    for il, (lat, lon) in enumerate(locations):
        locname = names[il]
        z_min, z_max = z_ranges[il]
        depth_key = (z_min, z_max)
        precomputed = unique_z[depth_key]

        obs_vs = sigma_vs = obs_q = sigma_q = None
        depth_val = None

        if use_preloaded:
            if use_vs and seismic_model_data.Vs is not None:
                obs_vs = float(seismic_model_data.Vs[il])
                sigma_vs = (
                    float(seismic_model_data.Vs_error[il])
                    if seismic_model_data.Vs_error is not None
                    else config.default_vs_error
                )
            if use_q and seismic_model_data.Q is not None:
                obs_q = float(seismic_model_data.Q[il])
                sigma_q = (
                    float(seismic_model_data.Q_error[il])
                    if seismic_model_data.Q_error is not None
                    else config.default_q_error
                )
                # Apply percent mode if Q_error wasn't already converted during loading
                if seismic_model_data.Q_error is None and config.q_error_mode == 'percent':
                    sigma_q = obs_q * config.default_q_error / 100.0
            if has_depth_col:
                depth_val = seismic_model_data.depths[il]

        job_args.append((
            il, locname, lat, lon, z_min, z_max,
            use_vs, use_q,
            obs_vs, sigma_vs, obs_q, sigma_q,
            depth_key, precomputed,
            prior_statevars, sweep_vectors,
            anelastic_method, config.save_ml_csv,
            has_depth_col, depth_val,
            config.default_vs_error, config.default_q_error,
        ))

    # ------------------------------------------------------------------
    # 4. Execute — sequential or parallel
    # ------------------------------------------------------------------
    if n_workers <= 1:
        # Sequential (no overhead, good for small runs)
        results = []
        for i, args in enumerate(job_args):
            if i % max(1, n_locations // 20) == 0 or i == n_locations - 1:
                print(f"     Processing location {i+1}/{n_locations}: {names[i]}")
            results.append(_process_one_location(args))
    else:
        import multiprocessing as mp
        import time as _time

        print(f"     Dispatching {n_locations} locations across {n_workers} workers...")
        chunksize = max(1, n_locations // (n_workers * 4))
        results = [None] * n_locations
        n_done = 0
        report_interval = max(1, n_locations // 20)  # ~5% increments
        t_start = _time.time()

        with mp.Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(
                _process_one_location_indexed, enumerate(job_args), chunksize=chunksize
            ):
                idx, result = res
                results[idx] = result
                n_done += 1
                if n_done % report_interval == 0 or n_done == n_locations:
                    elapsed = _time.time() - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (n_locations - n_done) / rate if rate > 0 else 0
                    print(
                        f"     [{n_done}/{n_locations}] "
                        f"{100*n_done/n_locations:.0f}% done — "
                        f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining "
                        f"({rate:.1f} loc/s)"
                    )

        n_ok = sum(1 for r in results if r is not None)
        print(f"     Completed {n_ok}/{n_locations} locations successfully")

    return results
