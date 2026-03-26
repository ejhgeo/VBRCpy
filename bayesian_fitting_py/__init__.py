"""
Bayesian Fitting for Seismic Inversion

Python translation of the MATLAB VBR bayesian_fitting project.
Estimates temperature, melt fraction, and grain size from seismic
Vs and Q observations using Bayesian inference.

Also includes a Python implementation of the VBR (Very Broadband Rheology)
calculator for generating parameter sweeps.
"""

from .probability import (
    probability_normal,
    probability_uniform,
    probability_lognormal,
    likelihood_from_residuals,
    joint_independent_probability,
    conditional_bayes,
    conditionally_independent_c_given_ab,
    probability_distributions,
)
from .prior import prior_model_probs, make_param_grid, prep_gs_lognormal
from .data_processing import (
    process_seismic_models,
    load_locations_from_file,
    load_seismic_model_from_csv,
    load_seismic_model_from_mat,
    load_seismic_model_from_netcdf,
    detect_file_type,
    load_seismic_model_universal,
    Location,
    SeismicModelData,
)
from .fitting import (
    fit_seismic_observations,
    fit_preloaded_observations,
)
from .prior import GrainSizePrior, MeltFractionPrior
from .plotting import generate_colors
from .run_bayes import run_bayesian_inversion, InversionConfig

# VBR module for sweep generation
from .vbr import VBR

__version__ = "1.0.0"
__all__ = [
    # Bayesian inversion
    "probability_distributions",
    "prior_model_probs",
    "process_seismic_models",
    "fit_seismic_observations",
    "fit_preloaded_observations",
    "run_bayesian_inversion",
    "InversionConfig",
    "Location",
    "SeismicModelData",
    "load_locations_from_file",
    "load_seismic_model_from_csv",
    "load_seismic_model_from_mat",
    "load_seismic_model_from_netcdf",
    "detect_file_type",
    "load_seismic_model_universal",
    "generate_colors",
    # VBR sweep generation
    "VBR",
]
