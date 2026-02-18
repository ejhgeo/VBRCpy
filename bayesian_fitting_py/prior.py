"""
Prior probability functions for Bayesian inference.

Functions for calculating prior probability distributions and
creating parameter grids for state variable sweeps.

Translated from MATLAB:
- vbr/fitting/priorModelProbs.m
- Projects/bayesian_fitting/functions/make_param_grid.m
- Projects/bayesian_fitting/functions/prep_gs_lognormal.m
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .probability import probability_distributions


def make_param_grid(
    state_names: List[str], sweep: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a parameter grid from the ranges of state variables in sweep.
    Also calculate the mean and standard deviation of each variable.

    Parameters
    ----------
    state_names : list of str
        Names of state variables to grid (e.g., ['T', 'phi', 'gs'])
    sweep : dict
        Dictionary with vectors of values for each state variable

    Returns
    -------
    dict
        Dictionary with:
        - Gridded arrays for each state variable
        - [field]_mean: mean of each variable
        - [field]_std: standard deviation of each variable
    """
    params = {}
    
    n_vars = len(state_names)
    
    if n_vars == 1:
        params[state_names[0]] = sweep[state_names[0]]
    elif n_vars == 2:
        grids = np.meshgrid(
            sweep[state_names[0]], sweep[state_names[1]], indexing='ij'
        )
        params[state_names[0]] = grids[0]
        params[state_names[1]] = grids[1]
    elif n_vars == 3:
        grids = np.meshgrid(
            sweep[state_names[0]],
            sweep[state_names[1]],
            sweep[state_names[2]],
            indexing='ij',
        )
        params[state_names[0]] = grids[0]
        params[state_names[1]] = grids[1]
        params[state_names[2]] = grids[2]
    elif n_vars == 4:
        grids = np.meshgrid(
            sweep[state_names[0]],
            sweep[state_names[1]],
            sweep[state_names[2]],
            sweep[state_names[3]],
            indexing='ij',
        )
        for i, name in enumerate(state_names):
            params[name] = grids[i]
    
    # Calculate mean and std for each variable
    for name in state_names:
        params[f'{name}_mean'] = np.mean(sweep[name])
        params[f'{name}_std'] = np.std(sweep[name])
    
    return params


def prep_gs_lognormal(
    params: Dict[str, Any], sweep: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare parameters for lognormal grain size prior.
    Converts grain size to dimensionless log-space.

    Parameters
    ----------
    params : dict
        Parameter dictionary with gs_mean and gs_std
    sweep : dict
        Sweep dictionary with gs_params (containing gsref)

    Returns
    -------
    dict
        Updated params with normalized values
    """
    # Save original values with units
    gs_mean = params['gs_mean']  # in micrometers
    gs_std = params['gs_std']  # dimensionless (in log-space)
    
    params['gs_mean_units'] = gs_mean
    params['gs_std_units'] = gs_std
    
    # Nondimensionalize and convert to log space
    gs_params = sweep['gs_params']
    params['gs_mean'] = np.log(gs_mean / gs_params['gsref'])
    
    return params


def prior_model_probs(
    states: Dict[str, Any], states_fields: List[str]
) -> Tuple[np.ndarray, float]:
    """
    Calculate prior model probabilities for state variables.

    Assuming that the state variables are independent:
        p(var1, var2, ...) = p(var1) * p(var2) * ...

    Parameters
    ----------
    states : dict
        Dictionary with:
        - [field]: array of state variable values
        - [field]_mean: mean value for normal/lognormal distributions
        - [field]_std: standard deviation
        - [field]_pdf_type: (optional) 'normal', 'lognormal', 
          'uniform_log', or 'uniform'
        - [field]_pdf: (optional) pre-computed PDF values
    states_fields : list of str
        Names of state variables being varied

    Returns
    -------
    tuple
        (Prior_mod, sigmaPreds):
        - Prior_mod: joint probability of all state variable combinations
        - sigmaPreds: joint standard deviation (for uncertainty propagation)
    """
    sigma_preds = 1.0
    marginals = []
    
    for field in states_fields:
        std_field = f'{field}_std'
        mean_field = f'{field}_mean'
        
        # Determine PDF type
        pdf_type_key = f'{field}_pdf_type'
        pdf_key = f'{field}_pdf'
        
        if pdf_type_key in states:
            pdf_type = states[pdf_type_key]
        else:
            pdf_type = 'uniform'
        
        # Override if we have an input PDF
        if pdf_key in states:
            pdf_type = 'input'
        
        # Calculate marginal probability
        if pdf_type == 'input':
            marginal = states[pdf_key]
            sigma = states.get(std_field, 1.0)
        
        elif pdf_type == 'normal':
            sigma = states[std_field]
            mu = states[mean_field]
            x = states[field]
            marginal = probability_distributions('normal', x, mu, sigma)
        
        elif pdf_type == 'lognormal':
            sigma = states[std_field]
            mu = states[mean_field]
            x = states[field]
            marginal = probability_distributions('lognormal', x, mu, sigma)
        
        elif pdf_type == 'uniform_log':
            # Uniform probability over natural log space
            x = np.log(states[field])
            min_val = x.min()
            max_val = x.max()
            sigma = 1.0
            marginal = probability_distributions('uniform', x, min_val, max_val)
        
        else:  # 'uniform' or default
            sigma = 1.0
            x = states[field]
            min_val = x.min()
            max_val = x.max()
            marginal = probability_distributions('uniform', x, min_val, max_val)
        
        marginals.append(marginal)
        
        # Propagate uncertainty (for product of independent variables)
        sigma_preds *= sigma
    
    # Joint probability (assuming independence)
    prior_mod = probability_distributions('joint independent', marginals)
    
    return prior_mod, sigma_preds


def calculate_levels(
    field: np.ndarray, targets: List[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate probability cutoff levels for contour plotting.

    Parameters
    ----------
    field : np.ndarray
        2D probability field
    targets : list of float
        Target confidence levels (e.g., [0.7, 0.8, 0.9, 0.95])

    Returns
    -------
    tuple
        (target_cutoffs, confidences, cutoffs):
        - target_cutoffs: probability values corresponding to targets
        - confidences: cumulative probability at each cutoff
        - cutoffs: array of cutoff values tested
    """
    n_c = 1000
    cutoffs = np.linspace(field.min(), field.max(), n_c)
    confs = np.zeros(n_c)
    
    for i, cutoff in enumerate(cutoffs):
        f = field[field >= cutoff]
        confs[i] = np.sum(f)
    
    target_cutoffs = np.zeros(len(targets))
    for i, target in enumerate(targets):
        c = cutoffs[confs >= target]
        if len(c) > 0:
            target_cutoffs[i] = c.max()
        else:
            target_cutoffs[i] = 0.0
    
    return target_cutoffs, confs, cutoffs


def store_ensemble(
    ensemble_pdf: Dict[str, Any],
    locname: str,
    anelastic_method: str,
    p_joint: np.ndarray,
    posterior_a: Dict[str, Any],
    include_mxw: bool = True,
) -> Dict[str, Any]:
    """
    Store or accumulate ensemble PDF for a location/method combination.

    Parameters
    ----------
    ensemble_pdf : dict
        Dictionary to store/accumulate PDFs
    locname : str
        Location name
    anelastic_method : str
        Anelastic method name
    p_joint : np.ndarray
        Joint posterior probability
    posterior_a : dict
        Posterior containing T and phi arrays
    include_mxw : bool
        If False and anelastic_method is 'xfit_mxw', skip storing

    Returns
    -------
    dict
        Updated ensemble_pdf dictionary
    """
    if anelastic_method == 'xfit_mxw' and not include_mxw:
        return ensemble_pdf
    
    if locname not in ensemble_pdf:
        ensemble_pdf[locname] = {
            'p_joint': p_joint.copy(),
            'post_T': posterior_a['T'].copy(),
            'post_phi': posterior_a['phi'].copy(),
        }
    else:
        ensemble_pdf[locname]['p_joint'] = (
            ensemble_pdf[locname]['p_joint'] + p_joint
        )
    
    return ensemble_pdf
