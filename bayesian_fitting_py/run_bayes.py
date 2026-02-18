"""
Main Bayesian inversion script for seismic observations.

Fits asthenospheric Vs and Q with the most likely state variables
(temperature, melt fraction, grain size) using Bayesian inference.

Translated from MATLAB Projects/bayesian_fitting/run_bayes.m
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict
import pickle

# Resolve the bundled data directory: <package_root>/../data/
_PACKAGE_DIR = Path(__file__).resolve().parent  # bayesian_fitting_py/
_PACKAGE_DATA_DIR = _PACKAGE_DIR.parent / 'data'  # vbrc_V2Tpy/data/

def _default_data_path(*parts: str) -> str:
    """Return an absolute path into the package's bundled data directory."""
    return str(_PACKAGE_DATA_DIR.joinpath(*parts))

from .data_processing import (
    Location,
    SeismicModelData,
    load_locations_from_file,
    load_seismic_model_from_csv,
    load_seismic_model_from_mat,
    load_seismic_model_from_netcdf,
)
from .fitting import (
    fit_seismic_observations,
    fit_preloaded_observations,
    GrainSizePrior,
    load_sweep_data,
    extract_ml_estimates,
    extract_ml_from_joint,
)
from .prior import store_ensemble
from .plotting import (
    plot_tradeoffs_posterior,
    plot_regional_fits,
    plot_ensemble_pdfs,
    save_figure_for_posterior,
    generate_colors,
)


# Available anelastic methods
AVAILABLE_ANELASTIC_METHODS = ['andrade_psp', 'eburgers_psp', 'xfit_mxw', 'xfit_premelt']

# Available location/data input modes
LOCATION_MODES = ['manual', 'locations_file', 'csv_model', 'mat_model', 'netcdf_model']


@dataclass
class InversionConfig:
    """Configuration for Bayesian inversion."""
    
    # Location/data input mode determines how locations and seismic data are provided:
    # - 'manual': locations from config, seismic data extracted from vs_file/q_file .mat files
    # - 'locations_file': locations from CSV file, seismic data from vs_file/q_file .mat files
    # - 'csv_model': locations AND seismic data from a single CSV file
    # - 'mat_model': locations AND seismic data from a .mat file (uses model depths)
    # - 'netcdf_model': locations AND seismic data from a NetCDF file (uses model depths)
    location_mode: str = 'manual'
    
    # For 'locations_file' mode: path to location file (CSV/text with columns: lon, lat, [name], z_min, z_max)
    location_file: Optional[str] = None
    
    # For 'csv_model', 'mat_model', 'netcdf_model' modes:
    # Path to the seismic model file containing locations AND observations
    seismic_model_file: Optional[str] = None
    
    # Default errors for seismic observations (used when not provided in data files)
    default_vs_error: float = 0.05  # km/s
    default_q_error: float = 10.0  # dimensionless
    
    # For model modes: depth range filter (optional)
    # For 'csv_model', 'mat_model', 'netcdf_model': filters which depths to include
    model_z_range: Optional[Tuple[float, float]] = None
    
    # Skip every N points to reduce computation (1 = use all points)
    model_subsample: int = 1
    
    # For 'manual' mode: locations to fit (lat, lon)
    locations: List[Tuple[float, float]] = field(
        default_factory=lambda: [(40.7, -117.5), (39, -109.8), (37.2, -100.9)]
    )
    names: List[str] = field(
        default_factory=lambda: ['BasinRange', 'ColoradoPlateau', 'Interior']
    )
    z_ranges: List[Tuple[float, float]] = field(
        default_factory=lambda: [(75, 105), (120, 150), (120, 150)]
    )
    location_colors: List[Tuple[float, ...]] = field(
        default_factory=lambda: [(1, 0.6, 0), (0, 0.8, 0), (0, 0.3, 0)]
    )
    
    # File paths for seismic data sources.
    # Users should set these in their config file (paths relative to cwd or absolute).
    # If omitted, defaults point to the example data bundled with the package.
    vs_file: str = field(default_factory=lambda: _default_data_path('vel_models', 'Shen_Ritzwoller_2016.mat'))
    q_file: str = field(default_factory=lambda: _default_data_path('Q_models', 'Dalton_Ekstrom_2008.mat'))
    lab_file: str = field(default_factory=lambda: _default_data_path('LAB_models', 'HopperFischer2018.mat'))
    sweep_file: str = field(default_factory=lambda: _default_data_path('plate_VBR', 'sweep_log_gs.mat'))
    
    # Anelastic methods to use (can be single method or list)
    # Options: 'andrade_psp', 'eburgers_psp', 'xfit_mxw', 'xfit_premelt', or 'all'
    anelastic_methods: List[str] = field(
        default_factory=lambda: [
            'eburgers_psp', 'xfit_mxw', 'xfit_premelt', 'andrade_psp'
        ]
    )
    
    # Grain size prior type: 'log_uniform', 'log_normal_1mm', 'log_normal_1cm'
    gs_prior_case: str = 'log_uniform'
    
    # Observation types to use: 'Vs', 'Q', or 'VsQ' (both)
    obs_types: str = 'VsQ'
    
    # Output settings
    output_dir: str = 'plots/output_plots'
    save_plots: bool = True
    
    # For large-scale runs, option to save ML estimates to CSV
    save_ml_csv: bool = False
    ml_csv_file: str = 'ml_estimates.csv'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InversionConfig':
        """Create config from dictionary."""
        # Convert lists back to tuples where needed
        if 'locations' in data:
            data['locations'] = [tuple(loc) for loc in data['locations']]
        if 'z_ranges' in data:
            data['z_ranges'] = [tuple(zr) for zr in data['z_ranges']]
        if 'model_z_range' in data and data['model_z_range'] is not None:
            data['model_z_range'] = tuple(data['model_z_range'])
        if 'location_colors' in data:
            data['location_colors'] = [tuple(c) for c in data['location_colors']]
        return cls(**data)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'InversionConfig':
        """Load configuration from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'InversionConfig':
        """Load configuration from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load YAML files. Install with: pip install pyyaml")
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def to_yaml(self, filepath: str) -> None:
        """Save configuration to a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to save YAML files. Install with: pip install pyyaml")
        
        # Convert to dict and ensure lists instead of tuples for cleaner YAML
        data = self.to_dict()
        data['locations'] = [list(loc) for loc in data['locations']]
        data['z_ranges'] = [list(zr) for zr in data['z_ranges']]
        data['location_colors'] = [list(c) for c in data['location_colors']]
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> InversionConfig:
    """
    Load configuration from a file (JSON or YAML).
    
    Parameters
    ----------
    filepath : str
        Path to configuration file (.json or .yaml/.yml)
    
    Returns
    -------
    InversionConfig
        Loaded configuration
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.json':
        return InversionConfig.from_json(str(filepath))
    elif filepath.suffix.lower() in ['.yaml', '.yml']:
        return InversionConfig.from_yaml(str(filepath))
    else:
        raise ValueError(f"Unsupported config file format: {filepath.suffix}")


def parse_anelastic_methods(methods_input: Union[str, List[str]]) -> List[str]:
    """
    Parse anelastic methods input into a list of valid method names.
    
    Parameters
    ----------
    methods_input : str or list
        Can be:
        - 'all': use all available methods
        - Single method name: 'xfit_premelt'
        - Comma-separated string: 'xfit_premelt,eburgers_psp'
        - List of method names
    
    Returns
    -------
    list
        List of valid anelastic method names
    """
    if isinstance(methods_input, list):
        methods = methods_input
    elif methods_input.lower() == 'all':
        return AVAILABLE_ANELASTIC_METHODS.copy()
    elif ',' in methods_input:
        methods = [m.strip() for m in methods_input.split(',')]
    else:
        methods = [methods_input.strip()]
    
    # Validate methods
    invalid = [m for m in methods if m not in AVAILABLE_ANELASTIC_METHODS]
    if invalid:
        raise ValueError(
            f"Invalid anelastic method(s): {invalid}. "
            f"Available methods: {AVAILABLE_ANELASTIC_METHODS}"
        )
    
    return methods


def get_grain_size_prior(gs_prior_case: str) -> Tuple[GrainSizePrior, str]:
    """
    Get grain size prior configuration based on case name.

    Parameters
    ----------
    gs_prior_case : str
        One of 'log_uniform', 'log_normal_1mm', 'log_normal_1cm'

    Returns
    -------
    tuple
        (GrainSizePrior, fig_prefix_dir)
    """
    if gs_prior_case == 'log_uniform':
        prior = GrainSizePrior(gs_pdf_type='uniform_log')
        fig_prefix = 'gsLogUniform'
    
    elif gs_prior_case == 'log_normal_1mm':
        prior = GrainSizePrior(
            gs_pdf_type='lognormal',
            gs_mean=0.001 * 1e6,  # 1 mm in micrometers
            gs_std=0.25,  # dimensionless in log-space
        )
        fig_prefix = 'gsLogNormal_1mm'
    
    elif gs_prior_case == 'log_normal_1cm':
        prior = GrainSizePrior(
            gs_pdf_type='lognormal',
            gs_mean=0.01 * 1e6,  # 1 cm in micrometers
            gs_std=0.25,  # dimensionless in log-space
        )
        fig_prefix = 'gsLogNormal_1cm'
    
    else:
        print(f"Warning: unexpected gs_prior_case '{gs_prior_case}', using log_uniform")
        prior = GrainSizePrior(gs_pdf_type='uniform_log')
        fig_prefix = 'gsLogUniform'
    
    return prior, fig_prefix


def prepare_locations(config: InversionConfig) -> Tuple[
    List[Tuple[float, float]], 
    List[str], 
    List[Tuple[float, float]], 
    List[Tuple[float, ...]],
    Optional[SeismicModelData]
]:
    """
    Prepare locations based on the configured location mode.

    Parameters
    ----------
    config : InversionConfig
        Configuration object

    Returns
    -------
    tuple
        (locations, names, z_ranges, colors, seismic_model_data)
        
        For 'manual' and 'locations_file' modes:
            seismic_model_data is None (observations loaded from files during fitting)
        
        For 'csv_model', 'mat_model', and 'netcdf_model' modes:
            seismic_model_data contains pre-loaded observations
    """
    seismic_model_data = None
    
    if config.location_mode == 'manual':
        # Use the manually specified locations
        locations = config.locations
        names = config.names
        z_ranges = config.z_ranges
        colors = config.location_colors
        
    elif config.location_mode == 'locations_file':
        if config.location_file is None:
            raise ValueError("location_mode='locations_file' requires location_file to be specified")
        locations, names, z_ranges = load_locations_from_file(config.location_file)
        # Generate colors for file-loaded locations
        colors = generate_colors(len(locations))
    
    elif config.location_mode == 'csv_model':
        # Load locations AND seismic data from a single CSV file
        if config.seismic_model_file is None:
            raise ValueError("location_mode='csv_model' requires seismic_model_file to be specified")
        
        seismic_model_data = load_seismic_model_from_csv(
            config.seismic_model_file,
            z_range=config.model_z_range,
            subsample=config.model_subsample,
            default_vs_error=config.default_vs_error,
            default_q_error=config.default_q_error,
        )
        locations = seismic_model_data.locations
        names = seismic_model_data.names
        z_ranges = seismic_model_data.z_ranges
        colors = generate_colors(len(locations))
        
    elif config.location_mode == 'mat_model':
        # Load locations AND seismic data from .mat file
        if config.seismic_model_file is None:
            raise ValueError("location_mode='mat_model' requires seismic_model_file to be specified")
        
        seismic_model_data = load_seismic_model_from_mat(
            config.seismic_model_file,
            z_range=config.model_z_range,
            subsample=config.model_subsample,
            default_vs_error=config.default_vs_error,
            default_q_error=config.default_q_error,
        )
        locations = seismic_model_data.locations
        names = seismic_model_data.names
        z_ranges = seismic_model_data.z_ranges
        colors = generate_colors(len(locations))
        
    elif config.location_mode == 'netcdf_model':
        # Load locations AND seismic data from NetCDF file
        if config.seismic_model_file is None:
            raise ValueError("location_mode='netcdf_model' requires seismic_model_file to be specified")
        
        seismic_model_data = load_seismic_model_from_netcdf(
            config.seismic_model_file,
            z_range=config.model_z_range,
            subsample=config.model_subsample,
            default_vs_error=config.default_vs_error,
            default_q_error=config.default_q_error,
        )
        locations = seismic_model_data.locations
        names = seismic_model_data.names
        z_ranges = seismic_model_data.z_ranges
        colors = generate_colors(len(locations))
        
    else:
        raise ValueError(f"Unknown location_mode: {config.location_mode}. "
                        f"Must be one of: {LOCATION_MODES}")
    
    # Ensure we have enough names and z_ranges
    if len(names) < len(locations):
        names.extend([f"point_{i}" for i in range(len(names), len(locations))])
    if len(z_ranges) < len(locations):
        # Use the last z_range for remaining points, or default
        default_z = z_ranges[-1] if z_ranges else (75, 150)
        z_ranges.extend([default_z] * (len(locations) - len(z_ranges)))
    if len(colors) < len(locations):
        colors.extend(generate_colors(len(locations) - len(colors)))
    
    return locations, names, z_ranges, colors, seismic_model_data


def run_bayesian_inversion(
    config: Optional[InversionConfig] = None,
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Bayesian inversion for seismic observations.

    This function fits asthenospheric Vs and Q with the most likely
    state variables (temperature, melt fraction, grain size) using
    Bayesian inference, for multiple locations and anelastic methods.

    Parameters
    ----------
    config : InversionConfig, optional
        Configuration object. If None, uses defaults.
    working_dir : str, optional
        Working directory for data files. If None, uses current directory.

    Returns
    -------
    dict
        Dictionary containing:
        - RegionalFits: results for each anelastic method and location
        - EnsemblePDF: combined PDF across all methods
        - EnsemblePDF_no_mxw: combined PDF excluding xfit_mxw
        - MLEstimates: ML estimates per method and location
        - EnsembleMLEstimates: ensemble ML estimates per location
    """
    if config is None:
        config = InversionConfig()
    
    if working_dir:
        os.chdir(working_dir)
    
    # Prepare locations based on mode
    print(f"Location mode: {config.location_mode}")
    locations, names, z_ranges, colors, seismic_model_data = prepare_locations(config)
    n_locations = len(locations)
    print(f"Total locations to process: {n_locations}")
    
    # Determine if we're using pre-loaded observations (model modes)
    use_preloaded = seismic_model_data is not None
    if use_preloaded:
        print(f"Using pre-loaded observations from seismic model file")
    
    # Determine if this is a large-scale run (affects output behavior)
    large_scale_run = n_locations > 20
    if large_scale_run:
        print("Large-scale run detected - will save summary CSV and suppress individual plots")
        # Override plot settings for large runs
        save_individual_plots = False
        config.save_ml_csv = True  # Force CSV output for large runs
    else:
        save_individual_plots = config.save_plots
    
    # Setup grain size prior
    grain_size_prior, fig_prefix_dir = get_grain_size_prior(config.gs_prior_case)
    
    # Setup file paths based on obs_types
    filenames = {}
    use_vs = config.obs_types in ['Vs', 'VsQ', 'both']
    use_q = config.obs_types in ['Q', 'VsQ', 'both']
    
    if use_vs:
        filenames['Vs'] = config.vs_file
    if use_q:
        filenames['Q'] = config.q_file
    filenames['LAB'] = config.lab_file
    
    print(f"Using observations: {config.obs_types}")
    
    # Create output directory
    output_dir = os.path.join(config.output_dir, fig_prefix_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize storage
    regional_fits: Dict[str, Dict[str, Any]] = {}
    ensemble_pdf: Dict[str, Any] = {}
    ensemble_pdf_no_mxw: Dict[str, Any] = {}
    ml_estimates: Dict[str, Dict[str, Any]] = {}  # Store ML estimates per method
    
    # For CSV output, collect all ML data
    ml_records: List[Dict[str, Any]] = []
    
    sweep = None  # Will be loaded on first iteration
    first_run = True
    
    # Loop over anelastic methods
    for anelastic_method in config.anelastic_methods:
        print(f"Calculating inference for {anelastic_method}")
        regional_fits[anelastic_method] = {}
        ml_estimates[anelastic_method] = {}
        
        # Loop over locations
        for il, (lat, lon) in enumerate(locations):
            locname = names[il]
            z_min, z_max = z_ranges[il]
            
            # Progress indicator for large runs
            if large_scale_run:
                if il % 10 == 0 or il == n_locations - 1:
                    print(f"     Processing location {il+1}/{n_locations}: {locname}")
            else:
                print(f"     fitting {locname}")
            
            # Create location object (only needed for non-preloaded modes)
            location = Location(
                lat=lat,
                lon=lon + 360 if lon < 0 else lon,  # Convert to positive longitude if needed
                z_min=z_min,
                z_max=z_max,
                smooth_rad=0.5,
            )
            
            # Run fitting - use different approach based on whether observations are preloaded
            try:
                if use_preloaded:
                    # Get pre-loaded observations for this location
                    obs_vs = None
                    sigma_vs = None
                    obs_q = None
                    sigma_q = None
                    
                    if seismic_model_data.Vs is not None:
                        obs_vs = float(seismic_model_data.Vs[il])
                        sigma_vs = float(seismic_model_data.Vs_error[il]) if seismic_model_data.Vs_error is not None else config.default_vs_error
                    if seismic_model_data.Q is not None:
                        obs_q = float(seismic_model_data.Q[il])
                        sigma_q = float(seismic_model_data.Q_error[il]) if seismic_model_data.Q_error is not None else config.default_q_error
                    
                    if first_run:
                        posterior, sweep = fit_preloaded_observations(
                            obs_vs, sigma_vs, obs_q, sigma_q,
                            (z_min, z_max), anelastic_method, grain_size_prior,
                            sweep_file=config.sweep_file,
                        )
                        first_run = False
                    else:
                        posterior, sweep = fit_preloaded_observations(
                            obs_vs, sigma_vs, obs_q, sigma_q,
                            (z_min, z_max), anelastic_method, grain_size_prior,
                            sweep=sweep,
                        )
                else:
                    # Load observations from files (original behavior)
                    if first_run:
                        posterior, sweep = fit_seismic_observations(
                            filenames, location, anelastic_method, grain_size_prior,
                            sweep_file=config.sweep_file,
                        )
                        first_run = False
                    else:
                        posterior, sweep = fit_seismic_observations(
                            filenames, location, anelastic_method, grain_size_prior,
                            sweep=sweep,
                        )
            except Exception as e:
                if not large_scale_run:
                    print(f"        Error fitting {locname}: {e}")
                continue
            
            # Determine obs_type label for filenames
            obs_label = config.obs_types.replace('VsQ', 'VQ').replace('both', 'VQ')
            
            # Save plots (only for small runs)
            if save_individual_plots:
                print("        saving plots...")
                save_figure_for_posterior(
                    posterior, sweep, locname, anelastic_method, output_dir, obs_label
                )
                print(f"        plots saved to {output_dir}/")
            
            # Extract maximum likelihood estimates (with predicted Vs/Q)
            ml_est = extract_ml_estimates(posterior, sweep, anelastic_method)
            # Also store observed values and location info
            ml_est['lat'] = lat
            ml_est['lon'] = lon
            ml_est['z_min'] = z_min
            ml_est['z_max'] = z_max
            if 'obs_Vs' in posterior:
                ml_est['obs_Vs'] = posterior['obs_Vs']
                ml_est['sigma_Vs'] = posterior['sigma_Vs']
            if 'obs_Q' in posterior:
                ml_est['obs_Q'] = posterior['obs_Q']
                ml_est['sigma_Q'] = posterior['sigma_Q']
            ml_estimates[anelastic_method][locname] = ml_est
            
            # Collect record for CSV output
            if config.save_ml_csv:
                record = {
                    'name': locname,
                    'lat': lat,
                    'lon': lon,
                }
                # Add depth (z) if available from preloaded model data
                if use_preloaded and seismic_model_data.depths is not None:
                    record['z'] = seismic_model_data.depths[il]
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
                chi2_total = 0.0
                n_obs = 0
                if 'obs_Vs' in ml_est:
                    record['Vs_obs'] = ml_est['obs_Vs']
                    vs_pred = ml_est.get('predicted_Vs', np.nan)
                    sigma_vs = ml_est['sigma_Vs']
                    record['Vs_pred'] = vs_pred
                    # Vs_misfit is the difference between observed and predicted
                    record['Vs_misfit'] = ml_est['obs_Vs'] - vs_pred if not np.isnan(vs_pred) else np.nan
                    # Chi-squared for Vs: ((obs - pred) / sigma)^2
                    if not np.isnan(vs_pred) and sigma_vs > 0:
                        vs_chi2 = ((ml_est['obs_Vs'] - vs_pred) / sigma_vs) ** 2
                        record['Vs_chi2'] = vs_chi2
                        chi2_total += vs_chi2
                        n_obs += 1
                    else:
                        record['Vs_chi2'] = np.nan
                if 'obs_Q' in ml_est:
                    record['Q_obs'] = ml_est['obs_Q']
                    q_pred = ml_est.get('predicted_Q', np.nan)
                    sigma_q = ml_est['sigma_Q']
                    record['Q_pred'] = q_pred
                    # Q_misfit is the difference between observed and predicted
                    record['Q_misfit'] = ml_est['obs_Q'] - q_pred if not np.isnan(q_pred) else np.nan
                    # Chi-squared for Q: ((obs - pred) / sigma)^2
                    if not np.isnan(q_pred) and sigma_q > 0:
                        q_chi2 = ((ml_est['obs_Q'] - q_pred) / sigma_q) ** 2
                        record['Q_chi2'] = q_chi2
                        chi2_total += q_chi2
                        n_obs += 1
                    else:
                        record['Q_chi2'] = np.nan
                # Total chi-squared (sum of individual chi-squared terms)
                record['chi2_total'] = chi2_total if n_obs > 0 else np.nan
                ml_records.append(record)
            
            # Calculate marginal P(phi, T | S)
            posterior_p = posterior['pS']
            posterior_p = posterior_p / np.sum(posterior_p)
            p_joint = np.sum(posterior_p, axis=2)  # Sum over grain size
            
            # Store in ensemble PDFs (skip for very large runs to save memory)
            if not large_scale_run or n_locations <= 1000:
                ensemble_pdf = store_ensemble(
                    ensemble_pdf, locname, anelastic_method, p_joint, posterior, include_mxw=True
                )
                ensemble_pdf_no_mxw = store_ensemble(
                    ensemble_pdf_no_mxw, locname, anelastic_method, p_joint, posterior, include_mxw=False
                )
                
                # Store regional fits
                regional_fits[anelastic_method][locname] = {
                    'p_joint': p_joint,
                    'phi_post': posterior['phi'],
                    'T_post': posterior['T'],
                }
    
    # Normalize ensemble PDFs (equal weighting across methods)
    n_methods = len(config.anelastic_methods)
    n_methods_no_mxw = n_methods - (1 if 'xfit_mxw' in config.anelastic_methods else 0)
    
    for locname in names:
        if locname in ensemble_pdf:
            ensemble_pdf[locname]['p_joint'] /= n_methods
        if locname in ensemble_pdf_no_mxw:
            ensemble_pdf_no_mxw[locname]['p_joint'] /= n_methods_no_mxw
    
    # Calculate ensemble ML estimates
    ensemble_ml_estimates: Dict[str, Any] = {}
    for locname in names:
        if locname in ensemble_pdf:
            ensemble_ml_estimates[locname] = extract_ml_from_joint(
                ensemble_pdf[locname]['p_joint'],
                ensemble_pdf[locname]['post_T'],
                ensemble_pdf[locname]['post_phi'],
            )
    
    # Generate summary plots (only for small runs)
    if config.save_plots and not large_scale_run:
        print("Building summary plots...")
        
        locs_array = np.array(locations)
        
        plot_regional_fits(
            regional_fits, locs_array, names,
            colors, fig_prefix_dir,
            save_dir=config.output_dir,
        )
        
        plot_ensemble_pdfs(
            ensemble_pdf, ensemble_pdf_no_mxw,
            locs_array, names, colors,
            fig_prefix_dir, save_dir=config.output_dir,
        )
    
    # Print ML estimates summary (abbreviated for large runs)
    print_ml_summary(ml_estimates, ensemble_ml_estimates, config, names, large_scale_run)
    
    # Package results
    results = {
        'RegionalFits': regional_fits,
        'EnsemblePDF': ensemble_pdf,
        'EnsemblePDF_no_mxw': ensemble_pdf_no_mxw,
        'MLEstimates': ml_estimates,
        'EnsembleMLEstimates': ensemble_ml_estimates,
        'config': config,
        'locations': locations,
        'names': names,
        'z_ranges': z_ranges,
    }
    
    # Save ML estimates to CSV if requested (save to current working directory)
    if config.save_ml_csv and ml_records:
        import csv
        csv_filename = config.ml_csv_file or f'{fig_prefix_dir}_ml_estimates.csv'
        # Save to current working directory, not output_dir
        csv_path = csv_filename
        
        # Get all field names from the records
        fieldnames = list(ml_records[0].keys())
        
        # Format floating point values to 3 decimal places or scientific notation
        def format_value(v):
            if isinstance(v, float):
                if abs(v) < 0.001 and v != 0:
                    return f'{v:.3e}'
                else:
                    return f'{v:.3f}'
            return v
        
        formatted_records = [
            {k: format_value(v) for k, v in record.items()}
            for record in ml_records
        ]
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(formatted_records)
        
        print(f"\nML estimates saved to {csv_path}")
    
    # Save results to pickle
    save_path = os.path.join(config.output_dir, f'{fig_prefix_dir}_ensembles.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Full results saved to {save_path}")
    
    return results


def print_ml_summary(
    ml_estimates: Dict[str, Dict[str, Any]],
    ensemble_ml_estimates: Dict[str, Any],
    config: InversionConfig,
    names: List[str],
    large_scale_run: bool = False,
) -> None:
    """
    Print a summary table of maximum likelihood estimates.

    Parameters
    ----------
    ml_estimates : dict
        ML estimates per anelastic method and location
    ensemble_ml_estimates : dict
        Ensemble ML estimates per location
    config : InversionConfig
        Configuration object
    names : list
        List of location names
    large_scale_run : bool
        If True, print abbreviated summary
    """
    n_locations = len(names)
    
    print("\n" + "="*90)
    print("SEISMIC TO TEMPERATURE INVERSION - MAXIMUM LIKELIHOOD ESTIMATES")
    print("="*90)
    
    if large_scale_run:
        print(f"\n  Large-scale run with {n_locations} locations")
        print(f"  Full results saved to CSV file.")
        
        # Just print summary statistics
        for method in config.anelastic_methods:
            if method not in ml_estimates:
                continue
            
            # Collect all ML T values for this method
            T_values = [est['T']['ml'] for locname, est in ml_estimates[method].items()]
            phi_values = [est['phi']['ml'] for locname, est in ml_estimates[method].items()]
            
            if T_values:
                print(f"\n  {method.upper()}: T range = {min(T_values):.0f} - {max(T_values):.0f} °C, "
                      f"mean = {np.mean(T_values):.0f} °C")
                print(f"  {' '*len(method)}  φ range = {min(phi_values):.4f} - {max(phi_values):.4f}, "
                      f"mean = {np.mean(phi_values):.4f}")
        
        print("\n" + "="*90)
        return
    
    # Determine what observations we have
    first_method = config.anelastic_methods[0]
    has_vs = 'obs_Vs' in ml_estimates.get(first_method, {}).get(names[0], {}) if names else False
    has_q = 'obs_Q' in ml_estimates.get(first_method, {}).get(names[0], {}) if names else False
    
    # Print per-method results with observed vs predicted comparison
    for method in config.anelastic_methods:
        if method not in ml_estimates:
            continue
        
        print(f"\n{method.upper().replace('_', ' ')}")
        print("=" * 90)
        
        # State variables section
        print("\n  State Variables (inverted from seismic observations):")
        print("  " + "-" * 75)
        print(f"  {'Location':<18} {'T (°C)':<18} {'φ (melt frac)':<18} {'d (mm)':<18}")
        print(f"  {'':<18} {'ML ± σ':<18} {'ML ± σ':<18} {'ML ± σ':<18}")
        print("  " + "-" * 75)
        
        for locname in names:
            if locname not in ml_estimates[method]:
                continue
            
            est = ml_estimates[method][locname]
            T_str = f"{est['T']['ml']:.0f} ± {est['T']['std']:.0f}"
            phi_str = f"{est['phi']['ml']:.4f} ± {est['phi']['std']:.4f}"
            gs_str = f"{est['gs']['ml_mm']:.2f} ± {est['gs']['std_mm']:.2f}"
            
            print(f"  {locname:<18} {T_str:<18} {phi_str:<18} {gs_str:<18}")
        
        # Data fit section (observed vs predicted)
        print("\n  Data Fit (observed vs predicted at ML solution):")
        print("  " + "-" * 85)
        
        # Build header based on available observations
        if has_vs and has_q:
            print(f"  {'Location':<18} {'Vs obs (km/s)':<15} {'Vs pred':<12} {'Q obs':<12} {'Q pred':<12} {'Misfit':<10}")
        elif has_vs:
            print(f"  {'Location':<18} {'Vs obs (km/s)':<15} {'Vs pred':<12} {'Misfit':<10}")
        else:
            print(f"  {'Location':<18} {'Q obs':<15} {'Q pred':<12} {'Misfit':<10}")
        print("  " + "-" * 85)
        
        for locname in names:
            if locname not in ml_estimates[method]:
                continue
            
            est = ml_estimates[method][locname]
            parts = [f"  {locname:<18}"]
            
            misfit_terms = []
            if has_vs:
                obs_vs = est['obs_Vs']
                pred_vs = est.get('predicted_Vs', float('nan'))
                sigma_vs = est['sigma_Vs']
                parts.append(f"{obs_vs:.3f} ± {sigma_vs:.2f}  ")
                parts.append(f"{pred_vs:.3f}       ")
                if not np.isnan(pred_vs):
                    misfit_terms.append(((obs_vs - pred_vs) / sigma_vs) ** 2)
            
            if has_q:
                obs_q = est['obs_Q']
                pred_q = est.get('predicted_Q', float('nan'))
                sigma_q = est['sigma_Q']
                parts.append(f"{obs_q:.1f} ± {sigma_q:.0f}  ")
                parts.append(f"{pred_q:.1f}        ")
                if not np.isnan(pred_q):
                    misfit_terms.append(((obs_q - pred_q) / sigma_q) ** 2)
            
            # Compute total misfit (chi-squared)
            if misfit_terms:
                chi_sq = sum(misfit_terms)
                parts.append(f"χ²={chi_sq:.2f}")
            
            print("".join(parts))
    
    # Print ensemble results
    if ensemble_ml_estimates and len(config.anelastic_methods) > 1:
        print(f"\n{'ENSEMBLE (all methods combined)'.upper()}")
        print("=" * 90)
        print("\n  State Variables (grain size marginalized out):")
        print("  " + "-" * 55)
        print(f"  {'Location':<18} {'T (°C)':<18} {'φ (melt frac)':<18}")
        print(f"  {'':<18} {'ML ± σ':<18} {'ML ± σ':<18}")
        print("  " + "-" * 55)
        
        for locname in names:
            if locname not in ensemble_ml_estimates:
                continue
            
            est = ensemble_ml_estimates[locname]
            T_str = f"{est['T']['ml']:.0f} ± {est['T']['std']:.0f}"
            phi_str = f"{est['phi']['ml']:.4f} ± {est['phi']['std']:.4f}"
            
            print(f"  {locname:<18} {T_str:<18} {phi_str:<18}")
    
    print("\n" + "="*90)
    print("Note: ML = Maximum Likelihood (mode of posterior), σ = standard deviation")
    print("      χ² = sum of squared normalized residuals (lower is better fit)")
    if len(config.anelastic_methods) > 1:
        print("      Grain size (d) not shown for ensemble (marginalized out)")
    print("="*90)


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Bayesian inversion for seismic Vs and Q observations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (all anelastic methods, both Vs and Q)
  python -m bayesian_fitting_py

  # Use only xfit_premelt method
  python -m bayesian_fitting_py --anelastic-methods xfit_premelt

  # Use multiple specific methods
  python -m bayesian_fitting_py --anelastic-methods xfit_premelt,eburgers_psp

  # Use all available methods
  python -m bayesian_fitting_py --anelastic-methods all

  # Use a parameter file
  python -m bayesian_fitting_py --config my_config.yaml

  # Generate a template parameter file
  python -m bayesian_fitting_py --generate-config my_config.yaml

  # Load locations from a file (seismic data still from .mat files)
  python -m bayesian_fitting_py --location-mode locations_file --location-file locations.txt

  # Load entire seismic model from CSV file (locations + Vs/Q from same file)
  python -m bayesian_fitting_py --location-mode csv_model --seismic-model-file model.csv

  # Load seismic model from .mat file using exact depths
  python -m bayesian_fitting_py --location-mode mat_model --seismic-model-file model.mat

  # Load seismic model from NetCDF file
  python -m bayesian_fitting_py --location-mode netcdf_model --seismic-model-file model.nc
        """
    )
    parser.add_argument(
        '--config', '-c', type=str, default=None,
        help='Path to configuration file (JSON or YAML). Command-line options override file settings.'
    )
    parser.add_argument(
        '--generate-config', type=str, default=None, metavar='FILE',
        help='Generate a template configuration file and exit'
    )
    parser.add_argument(
        '--gs-prior', type=str, default=None,
        choices=['log_uniform', 'log_normal_1mm', 'log_normal_1cm'],
        help='Grain size prior type'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for plots'
    )
    parser.add_argument(
        '--data-dir', type=str, default='.',
        help='Data directory containing .mat files'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Disable plot generation'
    )
    parser.add_argument(
        '--obs-types', type=str, default=None,
        choices=['Vs', 'Q', 'VsQ'],
        help='Observation types to use: Vs only, Q only, or both (VsQ)'
    )
    parser.add_argument(
        '--anelastic-methods', type=str, default=None,
        help=(
            'Anelastic method(s) to use. Options: single method (e.g., "xfit_premelt"), '
            'comma-separated list (e.g., "xfit_premelt,eburgers_psp"), '
            'or "all" for all methods. '
            f'Available: {", ".join(AVAILABLE_ANELASTIC_METHODS)}'
        )
    )
    parser.add_argument(
        '--list-methods', action='store_true',
        help='List available anelastic methods and exit'
    )
    parser.add_argument(
        '--location-mode', type=str, default=None,
        choices=LOCATION_MODES,
        help=(
            'Location specification mode: '
            'manual (use config), '
            'locations_file (load locations from file, seismic from .mat), '
            'csv_model (locations + seismic from CSV), '
            'mat_model (locations + seismic from .mat with depths), '
            'netcdf_model (locations + seismic from NetCDF)'
        )
    )
    parser.add_argument(
        '--location-file', type=str, default=None,
        help='Path to location file (for --location-mode locations_file). Format: lon,lat[,name],z_min,z_max'
    )
    parser.add_argument(
        '--seismic-model-file', type=str, default=None,
        help='Path to seismic model file (for csv_model, mat_model, or netcdf_model modes)'
    )
    parser.add_argument(
        '--model-z-range', type=str, default=None,
        help='Depth range filter for model modes: "z_min,z_max" in km (e.g., "100,200")'
    )
    parser.add_argument(
        '--model-subsample', type=int, default=None,
        help='Subsampling factor for model modes (e.g., 2 = every 2nd point)'
    )
    parser.add_argument(
        '--default-vs-error', type=float, default=None,
        help='Default Vs error (km/s) for model modes if error not in file'
    )
    parser.add_argument(
        '--default-q-error', type=float, default=None,
        help='Default Q error for model modes if error not in file'
    )
    parser.add_argument(
        '--save-csv', action='store_true',
        help='Save ML estimates to CSV file'
    )
    parser.add_argument(
        '--csv-file', type=str, default=None,
        help='Output CSV filename (default: auto-generated based on run parameters)'
    )
    
    args = parser.parse_args()
    
    # Handle --list-methods
    if args.list_methods:
        print("Available anelastic methods:")
        for method in AVAILABLE_ANELASTIC_METHODS:
            print(f"  - {method}")
        return
    
    # Handle --generate-config
    if args.generate_config:
        config = InversionConfig()
        filepath = args.generate_config
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            config.to_yaml(filepath)
        else:
            if not filepath.endswith('.json'):
                filepath += '.json'
            config.to_json(filepath)
        return
    
    # Load config from file or use defaults
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        config = InversionConfig()
    
    # Override with command-line arguments
    if args.gs_prior is not None:
        config.gs_prior_case = args.gs_prior
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.no_plots:
        config.save_plots = False
    if args.obs_types is not None:
        config.obs_types = args.obs_types
    if args.anelastic_methods is not None:
        config.anelastic_methods = parse_anelastic_methods(args.anelastic_methods)
    if args.location_mode is not None:
        config.location_mode = args.location_mode
    if args.location_file is not None:
        config.location_file = args.location_file
    if args.seismic_model_file is not None:
        config.seismic_model_file = args.seismic_model_file
    if args.model_z_range is not None:
        z_min, z_max = map(float, args.model_z_range.split(','))
        config.model_z_range = (z_min, z_max)
    if args.model_subsample is not None:
        config.model_subsample = args.model_subsample
    if args.default_vs_error is not None:
        config.default_vs_error = args.default_vs_error
    if args.default_q_error is not None:
        config.default_q_error = args.default_q_error
    if args.save_csv:
        config.save_ml_csv = True
    if args.csv_file is not None:
        config.ml_csv_file = args.csv_file
    
    # Run the inversion
    results = run_bayesian_inversion(config, working_dir=args.data_dir)
    
    n_locations = len(results.get('names', config.names))
    print("\nInversion complete!")
    print(f"Processed {n_locations} locations with {len(config.anelastic_methods)} anelastic method(s)")
    print(f"Anelastic methods used: {', '.join(config.anelastic_methods)}")


if __name__ == '__main__':
    main()