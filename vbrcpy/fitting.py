"""
Main fitting function for seismic observations.

Fits asthenospheric Vs and Q with the most likely state variables
(temperature, melt fraction, grain size) using Bayesian inference.

Translated from MATLAB Projects/bayesian_fitting/fit_seismic_observations.m
"""

import os
import numpy as np
from scipy.io import loadmat
from typing import Dict, Any, Tuple, Optional

from .prior import GrainSizePrior, MeltFractionPrior, TemperaturePrior
from .prior import apply_melt_fraction_prior, apply_temperature_prior
from .data_processing import (
    Location,
    process_seismic_models,
    check_file_exists,
)
from .prior import (
    make_param_grid,
    prep_gs_lognormal,
    prior_model_probs,
)
from .probability import probability_distributions


def load_sweep_data(sweep_file: str) -> Dict[str, Any]:
    """
    Load pre-calculated parameter sweep from file.

    Supports multiple formats:
    - .mat: MATLAB format (original and Python-generated)
    - .npz: NumPy compressed archive
    - .pkl/.pickle: Python pickle format

    Parameters
    ----------
    sweep_file : str
        Path to the sweep file

    Returns
    -------
    dict
        Sweep dictionary with VBR calculation results
    """
    ext = os.path.splitext(sweep_file)[1].lower()
    
    if ext in ['.npz', '.pkl', '.pickle']:
        # Use the new Python-native loader
        from .vbr.generate_sweep import load_sweep
        return load_sweep(sweep_file)
    elif ext == '.mat':
        return _load_sweep_mat(sweep_file)
    else:
        # Try .mat format as default
        return _load_sweep_mat(sweep_file)


def _load_sweep_mat(sweep_file: str) -> Dict[str, Any]:
    """
    Load sweep from MATLAB .mat file.
    
    Handles both original MATLAB-generated files and Python-generated files.
    """
    mat_data = loadmat(sweep_file, squeeze_me=True, struct_as_record=False)
    sweep_obj = mat_data['sweep']
    
    sweep = {}
    
    # Extract basic fields
    for attr in ['T', 'phi', 'gs', 'z', 'P_GPa', 'per_bw_max', 'per_bw_min']:
        if hasattr(sweep_obj, attr):
            value = getattr(sweep_obj, attr)
            sweep[attr] = np.atleast_1d(value)
    
    sweep['state_names'] = ['T', 'phi', 'gs']
    
    # Extract gs_params if present
    if hasattr(sweep_obj, 'gs_params'):
        gs_params_obj = sweep_obj.gs_params
        sweep['gs_params'] = {
            'type': str(getattr(gs_params_obj, 'type', 'linear')),
            'gsmin': float(getattr(gs_params_obj, 'gsmin', sweep['gs'].min())),
            'gsmax': float(getattr(gs_params_obj, 'gsmax', sweep['gs'].max())),
            'gsref': float(getattr(gs_params_obj, 'gsref', 1e3)),
        }
    else:
        sweep['gs_params'] = {
            'type': 'linear',
            'gsmin': sweep['gs'].min(),
            'gsmax': sweep['gs'].max(),
            'gsref': 1e3,
        }
    
    # Extract Box structure
    nT = len(sweep['T'])
    nphi = len(sweep['phi'])
    ngs = len(sweep['gs'])
    nz = len(sweep['z'])
    
    methods = ['andrade_psp', 'eburgers_psp', 'xfit_mxw', 'xfit_premelt']
    
    box = {}
    for method in methods:
        box[method] = {
            'meanVs': np.zeros((nT, nphi, ngs, nz)),
            'meanQ': np.zeros((nT, nphi, ngs, nz)),
        }
    
    # Check if this is original MATLAB format (has Box attribute with struct array)
    if hasattr(sweep_obj, 'Box'):
        box_obj = sweep_obj.Box
        if hasattr(box_obj, '__iter__') and not isinstance(box_obj, str):
            # Original MATLAB format: Box is (nT, nphi, ngs) array of structs
            for idx, box_elem in enumerate(box_obj.flat):
                i_T = idx // (nphi * ngs)
                i_phi = (idx // ngs) % nphi
                i_gs = idx % ngs
                
                for method in methods:
                    if hasattr(box_elem, method):
                        method_obj = getattr(box_elem, method)
                        if hasattr(method_obj, 'meanVs'):
                            vs_vals = np.atleast_1d(method_obj.meanVs)
                            q_vals = np.atleast_1d(method_obj.meanQ)
                            box[method]['meanVs'][i_T, i_phi, i_gs, :] = vs_vals
                            box[method]['meanQ'][i_T, i_phi, i_gs, :] = q_vals
                    elif isinstance(box_elem, dict) and method in box_elem:
                        # Python-generated format with dict structure
                        method_data = box_elem[method]
                        if isinstance(method_data, dict):
                            vs_vals = np.atleast_1d(method_data.get('meanVs', method_data.get('meanVs', [])))
                            q_vals = np.atleast_1d(method_data.get('meanQ', method_data.get('meanQ', [])))
                            box[method]['meanVs'][i_T, i_phi, i_gs, :] = vs_vals
                            box[method]['meanQ'][i_T, i_phi, i_gs, :] = q_vals
    
    # Also check for flat format (old Python-generated format with Box_method_meanVs keys)
    for method in methods:
        vs_key = f'Box_{method}_meanVs'
        q_key = f'Box_{method}_meanQ'
        if hasattr(sweep_obj, vs_key):
            box[method]['meanVs'] = np.atleast_1d(getattr(sweep_obj, vs_key))
            box[method]['meanQ'] = np.atleast_1d(getattr(sweep_obj, q_key))
    
    sweep['Box'] = box
    return sweep


def extract_calculated_values_in_depth_range(
    sweep: Dict[str, Any],
    obs_name: str,
    anelastic_method: str,
    depth_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mean calculated values within a depth range.

    Parameters
    ----------
    sweep : dict
        Parameter sweep dictionary
    obs_name : str
        'Vs' or 'Q'
    anelastic_method : str
        Anelastic method name
    depth_range : tuple
        (min_depth_km, max_depth_km)

    Returns
    -------
    tuple
        (mean_values, z_indices)
    """
    z = sweep['z']
    depth_range_m = np.array(depth_range) * 1e3
    z_inds = np.where((z >= depth_range_m[0]) & (z <= depth_range_m[1]))[0]
    
    if len(z_inds) == 0:
        sweep_zmin_km = z.min() / 1e3
        sweep_zmax_km = z.max() / 1e3
        raise ValueError(
            f"No sweep depths fall within the observation depth range "
            f"({depth_range[0]:.0f}–{depth_range[1]:.0f} km). "
            f"The sweep covers {sweep_zmin_km:.0f}–{sweep_zmax_km:.0f} km. "
            f"Regenerate the sweep with a depth range that covers your observations."
        )
    
    field_name = f'mean{obs_name}'
    data = sweep['Box'][anelastic_method][field_name]
    
    # Average over the depth indices
    mean_val = np.mean(data[:, :, :, z_inds], axis=3)
    
    return mean_val, z_inds


def fit_seismic_observations(
    filenames: Dict[str, str],
    location: Location,
    anelastic_method: str,
    grain_size_prior: GrainSizePrior,
    sweep: Optional[Dict[str, Any]] = None,
    sweep_file: str = 'data/plate_VBR/sweep_log_gs.mat',
    melt_fraction_prior: Optional[MeltFractionPrior] = None,
    temperature_prior: Optional[TemperaturePrior] = None,
    obs_vs_override: Optional[float] = None,
    sigma_vs_override: Optional[float] = None,
    obs_q_override: Optional[float] = None,
    sigma_q_override: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Fit input shear velocity and/or Q to state variables using VBR.

    Parameters
    ----------
    filenames : dict
        Dictionary with paths to observational data:
        - 'Vs': path to velocity model .mat file
        - 'Q': path to Q model .mat file
        - 'LAB': path to LAB depth model .mat file
    location : Location
        Location specification with lat, lon, z_min, z_max
    anelastic_method : str
        Anelastic method for calculations:
        'eburgers_psp', 'xfit_mxw', 'xfit_premelt', 'andrade_psp'
    grain_size_prior : GrainSizePrior
        Configuration for grain size prior distribution
    sweep : dict, optional
        Pre-loaded parameter sweep. If None, will be loaded from sweep_file
    sweep_file : str
        Path to sweep .mat file (used if sweep is None)

    Returns
    -------
    tuple
        (posterior, sweep):
        - posterior: dict with pS (posterior probability), state_names,
          and vectors for each state variable
        - sweep: the parameter sweep (passed back for reuse)
    """
    # Check which observation files exist
    vs_exists = check_file_exists(filenames, 'Vs')
    q_exists = check_file_exists(filenames, 'Q')
    # Overrides from 1D Earth model take precedence over .mat files
    vs_override = obs_vs_override is not None and sigma_vs_override is not None
    if vs_override:
        vs_exists = True
    q_override = obs_q_override is not None and sigma_q_override is not None
    if q_override:
        q_exists = True
    
    if not vs_exists and not q_exists:
        raise ValueError("At least one of Vs or Q files must exist")
    
    # Load or get sweep data
    if sweep is None:
        if not os.path.exists(sweep_file):
            raise FileNotFoundError(
                f"Sweep file {sweep_file} not found. "
                "Please generate it using the VBR MATLAB code first, or provide pre-computed sweep data."
            )
        print(f"Loading sweep from {sweep_file}...")
        sweep = load_sweep_data(sweep_file)
    
    # Get observed values
    if vs_exists:
        if vs_override:
            print("        using Vs from 1D Earth model")
            obs_vs = obs_vs_override
            sigma_vs = sigma_vs_override
        else:
            print("        extracting Vs")
            obs_vs, sigma_vs = process_seismic_models(
                'Vs', location, filenames['Vs'], ifplot=False
            )
        mean_vs, z_inds = extract_calculated_values_in_depth_range(
            sweep, 'Vs', anelastic_method, (location.z_min, location.z_max)
        )
        sweep['meanVs'] = mean_vs
        sweep['z_inds'] = z_inds
    
    if q_exists:
        if q_override:
            print("        using Q from 1D Earth model")
            obs_q = obs_q_override
            sigma_q = sigma_q_override
        else:
            print("        extracting Q")
            obs_q, sigma_q = process_seismic_models(
                'Q', location, filenames['Q'], ifplot=False
            )
        mean_q, _ = extract_calculated_values_in_depth_range(
            sweep, 'Q', anelastic_method, (location.z_min, location.z_max)
        )
        sweep['meanQ'] = mean_q
    
    # Extract viscosity if available in the sweep
    if 'meanEta' in sweep.get('Box', {}).get(anelastic_method, {}):
        mean_eta, _ = extract_calculated_values_in_depth_range(
            sweep, 'Eta', anelastic_method, (location.z_min, location.z_max)
        )
        sweep['meanEta'] = mean_eta
    
    # Build parameter grid and set up priors
    params = make_param_grid(sweep['state_names'], sweep)
    
    # Apply grain size prior settings
    if grain_size_prior.gs_mean is not None:
        params['gs_mean'] = grain_size_prior.gs_mean
    if grain_size_prior.gs_std is not None:
        params['gs_std'] = grain_size_prior.gs_std
    if grain_size_prior.gs_pdf_type is not None:
        params['gs_pdf_type'] = grain_size_prior.gs_pdf_type
    
    # Apply melt fraction prior
    depth_km = (location.z_min + location.z_max) / 2.0
    if melt_fraction_prior is not None:
        apply_melt_fraction_prior(params, melt_fraction_prior, depth_km=depth_km)

    # Apply temperature prior (depth-dependent geotherm)
    if temperature_prior is not None:
        apply_temperature_prior(params, temperature_prior, depth_km=depth_km)

    # Handle lognormal grain size prior
    gs_lognormal = False
    if params.get('gs_pdf_type') in ['lognormal', 'uniform_log']:
        gs_lognormal = True
        if params.get('gs_pdf_type') == 'lognormal':
            params = prep_gs_lognormal(params, sweep)
        # Normalize grain size for prior calculation
        sweep['gs'] = sweep['gs'] / sweep['gs_params']['gsref']
        params['gs'] = params['gs'] / sweep['gs_params']['gsref']
    
    # Calculate prior probabilities
    prior_statevars, _ = prior_model_probs(params, sweep['state_names'])
    
    # Restore grain size units
    if gs_lognormal:
        sweep['gs'] = sweep['gs'] * sweep['gs_params']['gsref']
        params['gs'] = params['gs'] * sweep['gs_params']['gsref']
    
    sweep['prior_model_params'] = params
    
    # Calculate likelihoods
    if vs_exists:
        likelihood_vs = probability_distributions(
            'likelihood from residuals', obs_vs, sigma_vs, mean_vs
        )
    
    if q_exists:
        likelihood_q = probability_distributions(
            'likelihood from residuals', obs_q, sigma_q, mean_q
        )
    
    # Calculate posteriors
    posterior = {}
    
    if vs_exists:
        print("        building p(S|Vs)")
        posterior_s_given_vs = probability_distributions(
            'A|B', likelihood_vs, prior_statevars, 1.0
        )
        posterior['pS'] = posterior_s_given_vs
        posterior['obs_Vs'] = obs_vs
        posterior['sigma_Vs'] = sigma_vs
    
    if q_exists:
        print("        building p(S|Q)")
        posterior_s_given_q = probability_distributions(
            'A|B', likelihood_q, prior_statevars, 1.0
        )
        posterior['pS'] = posterior_s_given_q
        posterior['obs_Q'] = obs_q
        posterior['sigma_Q'] = sigma_q
    
    # Combined Vs and Q posterior
    if vs_exists and q_exists:
        print("        building p(S|Vs,Q)")
        posterior_s_given_vs_and_q = probability_distributions(
            'C|A,B conditionally independent',
            likelihood_vs, likelihood_q, prior_statevars, 1.0
        )
        posterior['pS'] = posterior_s_given_vs_and_q
    
    # Store state variable info
    posterior['state_names'] = sweep['state_names']
    for name in sweep['state_names']:
        posterior[name] = sweep[name]
    
    return posterior, sweep


def fit_preloaded_observations(
    obs_vs: Optional[float],
    sigma_vs: Optional[float],
    obs_q: Optional[float],
    sigma_q: Optional[float],
    z_range: Tuple[float, float],
    anelastic_method: str,
    grain_size_prior: GrainSizePrior,
    sweep: Optional[Dict[str, Any]] = None,
    sweep_file: str = 'data/plate_VBR/sweep_log_gs.mat',
    melt_fraction_prior: Optional[MeltFractionPrior] = None,
    temperature_prior: Optional[TemperaturePrior] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Fit pre-loaded seismic observations to state variables using VBR.

    This function is similar to fit_seismic_observations but accepts
    already-extracted Vs and Q values rather than loading from files.
    This is used for model modes where locations and observations come
    from a single data file (CSV, MAT, or NetCDF).

    Parameters
    ----------
    obs_vs : float or None
        Observed shear velocity (km/s). None if not using Vs.
    sigma_vs : float or None
        Uncertainty in Vs observation. None if not using Vs.
    obs_q : float or None
        Observed quality factor Q. None if not using Q.
    sigma_q : float or None
        Uncertainty in Q observation. None if not using Q.
    z_range : tuple
        (z_min, z_max) depth range in km for averaging calculated values
    anelastic_method : str
        Anelastic method for calculations:
        'eburgers_psp', 'xfit_mxw', 'xfit_premelt', 'andrade_psp'
    grain_size_prior : GrainSizePrior
        Configuration for grain size prior distribution
    sweep : dict, optional
        Pre-loaded parameter sweep. If None, will be loaded from sweep_file
    sweep_file : str
        Path to sweep .mat file (used if sweep is None)

    Returns
    -------
    tuple
        (posterior, sweep):
        - posterior: dict with pS (posterior probability), state_names,
          and vectors for each state variable
        - sweep: the parameter sweep (passed back for reuse)
    """
    vs_exists = obs_vs is not None and sigma_vs is not None
    q_exists = obs_q is not None and sigma_q is not None
    
    if not vs_exists and not q_exists:
        raise ValueError("At least one of Vs or Q observations must be provided")
    
    # Load or get sweep data
    if sweep is None:
        if not os.path.exists(sweep_file):
            raise FileNotFoundError(
                f"Sweep file {sweep_file} not found. "
                "Please generate it using the VBR MATLAB code first, or provide pre-computed sweep data."
            )
        print(f"Loading sweep from {sweep_file}...")
        sweep = load_sweep_data(sweep_file)
    
    z_min, z_max = z_range
    
    # Get calculated values from sweep in depth range
    if vs_exists:
        print("        using pre-loaded Vs")
        from .data_processing import extract_calculated_values_in_depth_range
        mean_vs, z_inds = extract_calculated_values_in_depth_range(
            sweep, 'Vs', anelastic_method, (z_min, z_max)
        )
        sweep['meanVs'] = mean_vs
        sweep['z_inds'] = z_inds
    
    if q_exists:
        print("        using pre-loaded Q")
        from .data_processing import extract_calculated_values_in_depth_range
        mean_q, _ = extract_calculated_values_in_depth_range(
            sweep, 'Q', anelastic_method, (z_min, z_max)
        )
        sweep['meanQ'] = mean_q
    
    # Extract viscosity if available in the sweep
    if 'meanEta' in sweep.get('Box', {}).get(anelastic_method, {}):
        from .data_processing import extract_calculated_values_in_depth_range
        mean_eta, _ = extract_calculated_values_in_depth_range(
            sweep, 'Eta', anelastic_method, (z_min, z_max)
        )
        sweep['meanEta'] = mean_eta
    
    # Build parameter grid and set up priors
    params = make_param_grid(sweep['state_names'], sweep)
    
    # Apply grain size prior settings
    if grain_size_prior.gs_mean is not None:
        params['gs_mean'] = grain_size_prior.gs_mean
    if grain_size_prior.gs_std is not None:
        params['gs_std'] = grain_size_prior.gs_std
    if grain_size_prior.gs_pdf_type is not None:
        params['gs_pdf_type'] = grain_size_prior.gs_pdf_type
    
    # Apply melt fraction prior
    depth_km = (z_min + z_max) / 2.0
    if melt_fraction_prior is not None:
        apply_melt_fraction_prior(params, melt_fraction_prior, depth_km=depth_km)

    # Apply temperature prior (depth-dependent geotherm)
    if temperature_prior is not None:
        apply_temperature_prior(params, temperature_prior, depth_km=depth_km)

    # Handle lognormal grain size prior
    gs_lognormal = False
    if params.get('gs_pdf_type') in ['lognormal', 'uniform_log']:
        gs_lognormal = True
        if params.get('gs_pdf_type') == 'lognormal':
            params = prep_gs_lognormal(params, sweep)
        # Normalize grain size for prior calculation
        sweep['gs'] = sweep['gs'] / sweep['gs_params']['gsref']
        params['gs'] = params['gs'] / sweep['gs_params']['gsref']
    
    # Calculate prior probabilities
    prior_statevars, _ = prior_model_probs(params, sweep['state_names'])
    
    # Restore grain size units
    if gs_lognormal:
        sweep['gs'] = sweep['gs'] * sweep['gs_params']['gsref']
        params['gs'] = params['gs'] * sweep['gs_params']['gsref']
    
    sweep['prior_model_params'] = params
    
    # Calculate likelihoods
    if vs_exists:
        likelihood_vs = probability_distributions(
            'likelihood from residuals', obs_vs, sigma_vs, mean_vs
        )
    
    if q_exists:
        likelihood_q = probability_distributions(
            'likelihood from residuals', obs_q, sigma_q, mean_q
        )
    
    # Calculate posteriors
    posterior = {}
    
    if vs_exists:
        print("        building p(S|Vs)")
        posterior_s_given_vs = probability_distributions(
            'A|B', likelihood_vs, prior_statevars, 1.0
        )
        posterior['pS'] = posterior_s_given_vs
        posterior['obs_Vs'] = obs_vs
        posterior['sigma_Vs'] = sigma_vs
    
    if q_exists:
        print("        building p(S|Q)")
        posterior_s_given_q = probability_distributions(
            'A|B', likelihood_q, prior_statevars, 1.0
        )
        posterior['pS'] = posterior_s_given_q
        posterior['obs_Q'] = obs_q
        posterior['sigma_Q'] = sigma_q
    
    # Combined Vs and Q posterior
    if vs_exists and q_exists:
        print("        building p(S|Vs,Q)")
        posterior_s_given_vs_and_q = probability_distributions(
            'C|A,B conditionally independent',
            likelihood_vs, likelihood_q, prior_statevars, 1.0
        )
        posterior['pS'] = posterior_s_given_vs_and_q
    
    # Store state variable info
    posterior['state_names'] = sweep['state_names']
    for name in sweep['state_names']:
        posterior[name] = sweep[name]
    
    return posterior, sweep


def extract_ml_estimates(
    posterior: Dict[str, Any],
    sweep: Dict[str, Any],
    anelastic_method: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Extract maximum likelihood estimates and uncertainties from posterior.

    The ML (MAP) estimates are taken from the joint maximum of the full
    posterior distribution — the single (T, φ, gs) combination with the
    highest probability.  Mean and standard deviation are still computed
    from the per-variable marginal distributions.

    Parameters
    ----------
    posterior : dict
        Posterior dictionary containing 'pS' and state variable arrays
    sweep : dict
        Parameter sweep dictionary
    anelastic_method : str, optional
        Anelastic method name for extracting predicted Vs/Q

    Returns
    -------
    dict
        Dictionary with ML estimates for each state variable:
        {
            'T': {'ml': value, 'std': value, 'mean': value, 'ml_idx': idx},
            'phi': {'ml': value, 'std': value, 'mean': value, 'ml_idx': idx},
            'gs': {'ml': value, 'std': value, 'mean': value, 'ml_idx': idx},
            'predicted_Vs': value (if available),
            'predicted_Q': value (if available),
        }
    """
    pS = posterior['pS'].copy()
    pS = pS / np.sum(pS)  # Normalize
    
    state_names = posterior['state_names']
    estimates = {}
    ml_indices = {}
    
    # Joint MAP: find the single (T, phi, gs) with highest posterior
    joint_map_idx = np.unravel_index(np.argmax(pS), pS.shape)
    
    for i, name in enumerate(state_names):
        values = posterior[name]
        
        # ML estimate from the joint MAP
        ml_idx = joint_map_idx[i]
        ml_value = values[ml_idx]
        ml_indices[name] = ml_idx
        
        # Compute marginal for mean and std (still useful for uncertainty)
        axes_to_sum = tuple(j for j in range(len(state_names)) if j != i)
        marginal = np.sum(pS, axis=axes_to_sum)
        marginal = marginal / np.sum(marginal)
        
        # Mean (expected value)
        mean_value = np.sum(marginal * values)
        
        # Variance and standard deviation
        variance = np.sum(marginal * (values - mean_value)**2)
        std_value = np.sqrt(variance)
        
        estimates[name] = {
            'ml': float(ml_value),
            'mean': float(mean_value),
            'std': float(std_value),
            'ml_idx': int(ml_idx),
        }
    
    # Convert grain size from micrometers to mm for display
    estimates['gs']['ml_mm'] = estimates['gs']['ml'] / 1000.0
    estimates['gs']['mean_mm'] = estimates['gs']['mean'] / 1000.0
    estimates['gs']['std_mm'] = estimates['gs']['std'] / 1000.0
    
    # Extract predicted Vs and Q at the ML state variable values
    if anelastic_method is not None:
        # Get indices for ML estimates
        i_T = ml_indices['T']
        i_phi = ml_indices['phi']
        i_gs = ml_indices['gs']
        
        # meanVs and meanQ are already averaged over the depth range
        if 'meanVs' in sweep:
            estimates['predicted_Vs'] = float(sweep['meanVs'][i_T, i_phi, i_gs])
        if 'meanQ' in sweep:
            estimates['predicted_Q'] = float(sweep['meanQ'][i_T, i_phi, i_gs])
        
        # Compute full viscosity posterior from the joint posterior
        # Each (T, phi, gs) point maps to a unique log10(eta), so we
        # can compute the posterior PDF of log10(eta) by binning.
        if 'meanEta' in sweep:
            eta_grid = sweep['meanEta']  # (nT, nphi, ngs)
            # Predicted eta at the MAP estimate
            eta_at_map = float(eta_grid[i_T, i_phi, i_gs])
            estimates['predicted_eta'] = eta_at_map
            
            # Compute log10(eta) posterior using the full joint posterior
            log10_eta_grid = np.log10(np.clip(eta_grid, 1e10, None))  # floor at 1e10 Pa·s
            log10_eta_flat = log10_eta_grid.ravel()
            pS_flat = pS.ravel()
            
            # Create bins spanning the range of log10(eta) values
            eta_min = np.floor(log10_eta_flat.min() * 2) / 2  # round down to 0.5
            eta_max = np.ceil(log10_eta_flat.max() * 2) / 2   # round up to 0.5
            n_bins = max(int((eta_max - eta_min) / 0.05), 50)
            bin_edges = np.linspace(eta_min, eta_max, n_bins + 1)
            bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            # Accumulate posterior probability into bins
            marginal_eta = np.zeros(n_bins)
            bin_idx = np.digitize(log10_eta_flat, bin_edges) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            for bi in range(n_bins):
                mask = bin_idx == bi
                marginal_eta[bi] = pS_flat[mask].sum()
            
            if marginal_eta.sum() > 0:
                marginal_eta /= marginal_eta.sum()
                
                ml_idx_eta = np.argmax(marginal_eta)
                ml_log10_eta = float(bin_centres[ml_idx_eta])
                mean_log10_eta = float(np.sum(marginal_eta * bin_centres))
                std_log10_eta = float(np.sqrt(np.sum(marginal_eta * (bin_centres - mean_log10_eta)**2)))
                
                estimates['log10_eta'] = {
                    'ml': ml_log10_eta,
                    'mean': mean_log10_eta,
                    'std': std_log10_eta,
                }
            else:
                estimates['log10_eta'] = {
                    'ml': np.log10(eta_at_map) if eta_at_map > 0 else np.nan,
                    'mean': np.nan,
                    'std': np.nan,
                }
    
    return estimates


def extract_ml_from_joint(
    p_joint: np.ndarray,
    T_values: np.ndarray,
    phi_values: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Extract ML estimates from a 2D joint distribution (T, phi).

    Used for ensemble results where grain size has been marginalized out.

    Parameters
    ----------
    p_joint : np.ndarray
        2D joint probability array (T x phi)
    T_values : np.ndarray
        Temperature values
    phi_values : np.ndarray
        Melt fraction values

    Returns
    -------
    dict
        Dictionary with ML estimates for T and phi
    """
    p_joint = p_joint / np.sum(p_joint)
    
    estimates = {}
    
    # Joint MAP: single (T, phi) with highest probability
    joint_idx = np.unravel_index(np.argmax(p_joint), p_joint.shape)
    
    # Temperature (axis 0)
    marginal_T = np.sum(p_joint, axis=1)
    marginal_T = marginal_T / np.sum(marginal_T)
    mean_T = np.sum(marginal_T * T_values)
    std_T = np.sqrt(np.sum(marginal_T * (T_values - mean_T)**2))
    estimates['T'] = {
        'ml': float(T_values[joint_idx[0]]),
        'mean': float(mean_T),
        'std': float(std_T),
    }
    
    # Melt fraction (axis 1)
    marginal_phi = np.sum(p_joint, axis=0)
    marginal_phi = marginal_phi / np.sum(marginal_phi)
    mean_phi = np.sum(marginal_phi * phi_values)
    std_phi = np.sqrt(np.sum(marginal_phi * (phi_values - mean_phi)**2))
    estimates['phi'] = {
        'ml': float(phi_values[joint_idx[1]]),
        'mean': float(mean_phi),
        'std': float(std_phi),
    }
    
    return estimates
