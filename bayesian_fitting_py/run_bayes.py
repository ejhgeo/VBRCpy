"""
Main Bayesian inversion script for seismic observations.

Fits asthenospheric Vs and Q with the most likely state variables
(temperature, melt fraction, grain size) using Bayesian inference.

Translated from MATLAB Projects/bayesian_fitting/run_bayes.m
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict, fields as _dc_fields
import pickle

# Resolve the bundled data directory: <package_root>/../data/
_PACKAGE_DIR = Path(__file__).resolve().parent  # bayesian_fitting_py/
_PACKAGE_DATA_DIR = _PACKAGE_DIR.parent / 'data'  # vbrc_V2Tpy/data/

# Guard: the data/ and data/reference_models/ directories live outside the
# Python package tree, so they are only accessible when the package is
# installed in editable mode (pip install -e).  Warn early if missing.
if not _PACKAGE_DATA_DIR.is_dir():
    raise RuntimeError(
        f"Data directory not found: {_PACKAGE_DATA_DIR}\n"
        "This package must be installed in editable mode:\n"
        "    pip install -e ./vbrc_V2Tpy\n"
        "A regular 'pip install' will not work because the data/ "
        "directory lives outside the Python package tree."
    )

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
    detect_file_type,
    load_seismic_model_universal,
)
from .fitting import (
    fit_seismic_observations,
    fit_preloaded_observations,
    load_sweep_data,
    extract_ml_estimates,
    extract_ml_from_joint,
)
from .prior import (
    store_ensemble,
    GrainSizePrior,
    MeltFractionPrior,
    TemperaturePrior,
)
from .parallel import run_locations_parallel
from .vbr.thermal import load_q_from_earth_model, load_vs_from_earth_model
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
LOCATION_MODES = ['manual', 'locations_file', 'model']


@dataclass
class InversionConfig:
    """Configuration for Bayesian inversion."""
    
    # Location/data input mode determines how locations are provided:
    # - 'manual': locations from config, seismic data extracted from vs_file/q_file
    # - 'locations_file': locations from CSV file, seismic data from vs_file/q_file
    # - 'model': locations AND seismic data from vs_file (file format auto-detected)
    location_mode: str = 'manual'
    
    # For 'locations_file' mode: path to location file (CSV/text with columns: lon, lat, [name], z_min, z_max)
    location_file: Optional[str] = None
    
    # Default errors for seismic observations (used when not provided in data files)
    default_vs_error: float = 0.05  # km/s
    default_q_error: float = 10.0  # dimensionless
    # Q error mode: 'absolute' (default) or 'percent' (error as % of Q value)
    q_error_mode: str = 'absolute'
    
    # For 'model' mode: depth range filter (optional)
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
    # vs_file / q_file accept any supported format (.mat, .csv, .nc) or a built-in
    # 1D Earth model name ('prem', 'stw105', 'stw105_nocrust').  File type is auto-detected from the
    # extension.  In 'model' mode vs_file also supplies the location grid.
    # If the same file contains both Vs and Q, set both to the same path.
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
    
    # Grain size prior configuration
    # gs_prior_type: 'log_uniform' (flat in log-space) or 'log_normal'
    gs_prior_type: str = 'log_uniform'
    # For log_normal: mean grain size in mm (e.g. 0.1, 1.0, 4.0, 10.0)
    gs_prior_mean_mm: Optional[float] = None
    # For log_normal: std dev in log-space (dimensionless, default 0.25)
    gs_prior_std: Optional[float] = None
    
    # Melt fraction prior configuration
    # phi_prior_type: 'uniform' (flat), 'zero_melt' (suppress all melt),
    #   'piecewise_depth' (uniform above onset, zero below),
    #   'temperature_dependent' (placeholder for future T-aware prior)
    phi_prior_type: str = 'uniform'
    # For piecewise_depth: depth (km) below which melt is suppressed
    phi_onset_depth_km: float = 80.0
    
    # Temperature prior configuration
    # t_prior_type: 'uniform' (flat, original behaviour) or 'geotherm'
    #   (Gaussian centred on a reference geotherm at each depth)
    t_prior_type: str = 'uniform'
    # Built-in geotherm name ('sc2006') or path to a CSV file with
    # columns depth_km, temperature_C
    geotherm_file: str = 'sc2006'
    # Standard deviation (°C) of the Gaussian T prior
    geotherm_std_C: float = 200.0
    
    # Observation types to use: 'Vs', 'Q', or 'VsQ' (both)
    obs_types: str = 'VsQ'
    
    # Output settings
    output_dir: str = 'plots/output_plots'
    save_plots: bool = True
    force_plots: bool = False  # Force posterior plots even for large-scale runs
    plot_every_n: int = 1  # Plot every N-th location (1 = all, useful with force_plots)
    
    # For large-scale runs, option to save ML estimates to CSV
    save_ml_csv: bool = False
    # Path for ML estimates CSV.  When None (default), derived as
    # {output_dir}/ml_estimates.csv so that the user only needs to
    # set output_dir once.
    ml_csv_file: Optional[str] = None
    
    # Parallelization: number of worker processes for large-scale runs.
    # 0 = auto (use all available CPU cores), 1 = sequential (no parallelization),
    # N>1 = use N worker processes.  Only used for large-scale (preloaded) runs.
    parallel_workers: int = 1
    
    # Run tagging for inversion output subdirectory naming.
    # 'none'  — use 'inversion_results' (current default behaviour)
    # 'auto'  — auto-generate a compact tag from inversion parameters
    # <str>   — use 'inversion_<str>' as the subdirectory name
    run_tag: str = 'none'
    
    def _auto_run_tag(self) -> str:
        """Build a compact, human-readable tag from inversion parameters."""
        parts = []
        # Anelastic method(s)
        if len(self.anelastic_methods) == 1:
            parts.append(self.anelastic_methods[0])
        else:
            parts.append('+'.join(self.anelastic_methods))
        # Temperature prior
        if self.t_prior_type == 'uniform':
            parts.append('Tuni')
        elif self.t_prior_type == 'geotherm':
            parts.append(f'Tgeo{self.geotherm_std_C:g}')
        else:
            parts.append(f'T{self.t_prior_type}')
        # Grain-size prior
        if self.gs_prior_type in ('log_uniform', 'uniform'):
            parts.append('gsLU')
        elif self.gs_prior_type == 'log_normal':
            mean = self.gs_prior_mean_mm if self.gs_prior_mean_mm is not None else '?'
            std = self.gs_prior_std if self.gs_prior_std is not None else 0.25
            parts.append(f'gsLN{mean}s{std}')
        else:
            parts.append(f'gs{self.gs_prior_type}')
        # Melt-fraction prior
        if self.phi_prior_type == 'uniform':
            parts.append('phiU')
        elif self.phi_prior_type == 'piecewise_depth':
            parts.append(f'phiPD{self.phi_onset_depth_km:g}')
        elif self.phi_prior_type == 'zero_melt':
            parts.append('phiZM')
        else:
            parts.append(f'phi{self.phi_prior_type}')
        # Observation types
        parts.append(self.obs_types)
        # Q error
        parts.append(f'qe{self.default_q_error:g}')
        return '_'.join(parts)
    
    def resolve_inversion_dir(self) -> str:
        """Return the inversion output subdirectory path.
        
        The path is relative to (and inside) ``output_dir``:
        - run_tag='none'  → ``{output_dir}/inversion_results``
        - run_tag='auto'  → ``{output_dir}/inversion_{auto_tag}``
        - run_tag=<str>   → ``{output_dir}/inversion_{run_tag}``
        """
        if self.run_tag == 'none':
            return os.path.join(self.output_dir, 'inversion_results')
        elif self.run_tag == 'auto':
            return os.path.join(self.output_dir, f'inversion_{self._auto_run_tag()}')
        else:
            return os.path.join(self.output_dir, f'inversion_{self.run_tag}')
    
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
        # Backward compatibility: translate old gs_prior_case to new fields
        if 'gs_prior_case' in data and 'gs_prior_type' not in data:
            old = data.pop('gs_prior_case')
            if old == 'log_normal_1mm':
                data['gs_prior_type'] = 'log_normal'
                data.setdefault('gs_prior_mean_mm', 1.0)
                data.setdefault('gs_prior_std', 0.25)
            elif old == 'log_normal_1cm':
                data['gs_prior_type'] = 'log_normal'
                data.setdefault('gs_prior_mean_mm', 10.0)
                data.setdefault('gs_prior_std', 0.25)
            else:
                data['gs_prior_type'] = 'log_uniform'
        elif 'gs_prior_case' in data:
            data.pop('gs_prior_case')  # new fields take precedence

        # Backward compatibility: translate old location modes & seismic_model_file
        if 'location_mode' in data:
            mode = data['location_mode']
            if mode in ('csv_model', 'mat_model', 'netcdf_model'):
                data['location_mode'] = 'model'
        # Old configs may have seismic_model_file instead of vs_file
        if 'seismic_model_file' in data:
            smf = data.pop('seismic_model_file')
            if smf is not None:
                data.setdefault('vs_file', smf)
                data.setdefault('q_file', smf)
        # Old configs may have q_model; fold into q_file
        if 'q_model' in data:
            qm = data.pop('q_model')
            if qm is not None:
                data['q_file'] = qm

        # Strip keys unknown to this dataclass (allows combined config files
        # that include a sweep_generation section or other extra top-level keys)
        known = {f.name for f in _dc_fields(cls)}
        data = {k: v for k, v in data.items() if k in known}

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


def get_grain_size_prior(config: 'InversionConfig') -> Tuple[GrainSizePrior, str]:
    """
    Build grain size prior from config fields.

    Parameters
    ----------
    config : InversionConfig
        Inversion configuration with gs_prior_type, gs_prior_mean_mm,
        and gs_prior_std fields.

    Returns
    -------
    tuple
        (GrainSizePrior, fig_prefix_dir)
    """
    gs_type = config.gs_prior_type

    if gs_type == 'log_uniform':
        prior = GrainSizePrior(gs_pdf_type='uniform_log')
        fig_prefix = 'gsLogUniform'

    elif gs_type == 'log_normal':
        mean_mm = config.gs_prior_mean_mm
        std = config.gs_prior_std
        if mean_mm is None:
            raise ValueError(
                "gs_prior_type is 'log_normal' but gs_prior_mean_mm is not set. "
                "Specify the mean grain size in mm (e.g. gs_prior_mean_mm: 1.0)."
            )
        if std is None:
            std = 0.25  # sensible default
        prior = GrainSizePrior(
            gs_pdf_type='lognormal',
            gs_mean=mean_mm * 1e3,  # mm → micrometers
            gs_std=std,
        )
        # Build a readable directory name like gsLogNormal_1mm or gsLogNormal_0.5mm
        if mean_mm == int(mean_mm):
            label = f'{int(mean_mm)}mm'
        else:
            label = f'{mean_mm}mm'
        fig_prefix = f'gsLogNormal_{label}'

    else:
        print(f"Warning: unexpected gs_prior_type '{gs_type}', using log_uniform")
        prior = GrainSizePrior(gs_pdf_type='uniform_log')
        fig_prefix = 'gsLogUniform'

    return prior, fig_prefix


def get_temperature_prior(config: 'InversionConfig') -> TemperaturePrior:
    """Build temperature prior from config fields.

    Returns
    -------
    TemperaturePrior
    """
    return TemperaturePrior(
        t_prior_type=config.t_prior_type,
        geotherm_file=config.geotherm_file,
        geotherm_std_C=config.geotherm_std_C,
    )


def get_melt_fraction_prior(config: 'InversionConfig') -> MeltFractionPrior:
    """Build melt fraction prior from config fields.

    Returns
    -------
    MeltFractionPrior
    """
    ptype = config.phi_prior_type

    if ptype == 'uniform':
        return MeltFractionPrior(phi_prior_type='uniform')
    elif ptype == 'zero_melt':
        return MeltFractionPrior(phi_prior_type='zero_melt')
    elif ptype == 'piecewise_depth':
        return MeltFractionPrior(
            phi_prior_type='piecewise_depth',
            onset_depth_km=config.phi_onset_depth_km,
        )
    elif ptype == 'temperature_dependent':
        return MeltFractionPrior(phi_prior_type='temperature_dependent')
    else:
        print(f"Warning: unknown phi_prior_type '{ptype}', using uniform")
        return MeltFractionPrior(phi_prior_type='uniform')


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
        
        For 'model' mode:
            seismic_model_data contains pre-loaded observations from vs_file
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
    
    elif config.location_mode == 'model':
        # Load locations AND seismic data from vs_file (format auto-detected)
        seismic_model_data = load_seismic_model_universal(
            config.vs_file,
            z_range=config.model_z_range,
            subsample=config.model_subsample,
            default_vs_error=config.default_vs_error,
            default_q_error=config.default_q_error,
            q_error_mode=config.q_error_mode,
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
    
    # ---- Merge Q data from q_file when it differs from vs_file ----
    # In 'model' mode, seismic_model_data was loaded from vs_file which may
    # already contain Q.  If q_file points to a different source we override.
    # In 'manual'/'locations_file' modes, seismic_model_data is None; overrides
    # are passed directly to fit_seismic_observations via the override params.

    # Detect file types for vs_file and q_file
    vs_ftype = detect_file_type(config.vs_file)
    q_ftype = detect_file_type(config.q_file)

    # Per-location override arrays (used in the sequential non-preloaded loop)
    vs_override_values = None
    vs_override_errors = None
    q_override_values = None
    q_override_errors = None

    if config.location_mode == 'model':
        # seismic_model_data already has Vs from vs_file.
        # Merge Q from q_file if it is a different source.
        if os.path.abspath(config.q_file) != os.path.abspath(config.vs_file):
            if q_ftype == 'earth_model':
                q_depths = np.array([
                    float(d) if d is not None else (zr[0] + zr[1]) / 2.0
                    for d, zr in zip(seismic_model_data.depths, z_ranges)
                ])
                q_vals = load_q_from_earth_model(
                    config.q_file.lower(), q_depths,
                )
                if config.q_error_mode == 'percent':
                    q_errs = q_vals * config.default_q_error / 100.0
                else:
                    q_errs = np.full_like(q_vals, config.default_q_error)
                seismic_model_data.Q = q_vals
                seismic_model_data.Q_error = q_errs
                print(f"Q observations loaded from 1D model: {config.q_file}")
            else:
                # Load Q from a separate multi-point file
                q_data = load_seismic_model_universal(
                    config.q_file,
                    z_range=config.model_z_range,
                    subsample=config.model_subsample,
                    default_vs_error=config.default_vs_error,
                    default_q_error=config.default_q_error,
                    q_error_mode=config.q_error_mode,
                )
                if q_data.has_q():
                    if len(q_data) == len(seismic_model_data):
                        seismic_model_data.Q = q_data.Q
                        seismic_model_data.Q_error = q_data.Q_error
                    else:
                        print(f"Warning: q_file has {len(q_data)} points vs "
                              f"{len(seismic_model_data)} in vs_file — "
                              "Q from q_file ignored (grid mismatch)")
                print(f"Q observations loaded from: {config.q_file}")

    else:
        # manual / locations_file modes — prepare per-location overrides
        # for any source that is an Earth model (not a .mat file)
        mid_depths = np.array([(zr[0] + zr[1]) / 2.0 for zr in z_ranges])

        if vs_ftype == 'earth_model':
            vs_override_values = load_vs_from_earth_model(
                config.vs_file.lower(), mid_depths,
            )
            vs_override_errors = np.full_like(
                vs_override_values, config.default_vs_error,
            )
            print(f"Vs observations loaded from 1D model: {config.vs_file}")

        if q_ftype == 'earth_model':
            q_override_values = load_q_from_earth_model(
                config.q_file.lower(), mid_depths,
            )
            if config.q_error_mode == 'percent':
                q_override_errors = q_override_values * config.default_q_error / 100.0
            else:
                q_override_errors = np.full_like(
                    q_override_values, config.default_q_error,
                )
            print(f"Q observations loaded from 1D model: {config.q_file}")

    # Determine if we're using pre-loaded observations (model modes)
    use_preloaded = seismic_model_data is not None
    if use_preloaded:
        print(f"Using pre-loaded observations from seismic model file")
    
    # Determine if this is a large-scale run (affects output behavior)
    large_scale_run = n_locations > 20
    if large_scale_run:
        n_methods = len(config.anelastic_methods)
        n_plotted_locs = len(range(0, n_locations, max(1, config.plot_every_n)))
        n_plots = n_plotted_locs * n_methods
        if config.force_plots and config.save_plots:
            every_n_msg = (f" (every {config.plot_every_n}th)"
                           if config.plot_every_n > 1 else "")
            print(f"\nWarning: this will generate {n_plots} posterior plots "
                  f"({n_plotted_locs}{every_n_msg} locations × {n_methods} method(s)).")
            if sys.stdin.isatty():
                print("Are you sure you want to force plotting output? [y/N] ", end='')
                sys.stdout.flush()
                answer = input().strip().lower()
                if answer in ('y', 'yes'):
                    print("Forcing individual plots for large-scale run.")
                    save_individual_plots = True
                else:
                    print("Plot generation cancelled — suppressing individual plots.")
                    save_individual_plots = False
            else:
                # Non-interactive (e.g. subprocess) — honor force_plots without prompt
                print("Non-interactive mode — forcing individual plots.")
                save_individual_plots = True
        else:
            print("Large-scale run detected - will save summary CSV and suppress individual plots")
            save_individual_plots = False
        config.save_ml_csv = True  # Force CSV output for large runs
    else:
        save_individual_plots = config.save_plots
    
    # Setup grain size prior
    grain_size_prior, fig_prefix_dir = get_grain_size_prior(config)
    
    # Setup melt fraction prior
    melt_fraction_prior = get_melt_fraction_prior(config)
    
    # Setup temperature prior
    temperature_prior = get_temperature_prior(config)
    
    # Setup file paths based on obs_types
    # filenames dict is only used for .mat file loading in manual/locations_file
    # modes.  Earth model names ('prem', 'stw105') are handled via overrides.
    filenames = {}
    use_vs = config.obs_types in ['Vs', 'VsQ', 'both']
    use_q = config.obs_types in ['Q', 'VsQ', 'both']
    
    if use_vs and vs_ftype != 'earth_model':
        filenames['Vs'] = config.vs_file
    if use_q and q_ftype != 'earth_model':
        filenames['Q'] = config.q_file
    filenames['LAB'] = config.lab_file
    
    print(f"Using observations: {config.obs_types}")
    
    # Resolve inversion output directory (respects run_tag)
    inversion_dir = config.resolve_inversion_dir()
    output_dir = os.path.join(inversion_dir, fig_prefix_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Inversion output directory: {inversion_dir}")
    
    # Initialize storage
    regional_fits: Dict[str, Dict[str, Any]] = {}
    ensemble_pdf: Dict[str, Any] = {}
    ensemble_pdf_no_mxw: Dict[str, Any] = {}
    ml_estimates: Dict[str, Dict[str, Any]] = {}  # Store ML estimates per method
    
    # For CSV output, collect all ML data
    ml_records: List[Dict[str, Any]] = []
    
    sweep = None  # Will be loaded on first iteration
    first_run = True
    
    # Resolve parallel worker count
    n_workers = config.parallel_workers
    if n_workers == 0:
        import multiprocessing as mp
        n_workers = mp.cpu_count() or 1
    
    # For parallel mode, pre-load sweep before entering the method loop
    if n_workers > 1 and use_preloaded:
        print(f"Parallel mode: {n_workers} workers — pre-loading sweep data...")
        sweep = load_sweep_data(config.sweep_file)
        first_run = False
    
    # Loop over anelastic methods
    for anelastic_method in config.anelastic_methods:
        print(f"Calculating inference for {anelastic_method}")
        regional_fits[anelastic_method] = {}
        ml_estimates[anelastic_method] = {}
        
        # ----- Parallel path (preloaded observations, multiple workers) -----
        if use_preloaded and n_workers > 1:
            import time as _time
            _t0 = _time.time()
            par_results = run_locations_parallel(
                locations, names, z_ranges,
                seismic_model_data, sweep, anelastic_method,
                grain_size_prior, config,
                n_workers=n_workers,
                use_vs=use_vs,
                use_q=use_q,
                melt_fraction_prior=melt_fraction_prior,
                temperature_prior=temperature_prior,
            )
            _elapsed = _time.time() - _t0
            print(f"     {anelastic_method} completed in {_elapsed:.1f}s ({n_workers} workers)")

            # Collect results back into the same data structures
            for res in par_results:
                if res is None:
                    continue
                locname = res['locname']
                ml_est = res['ml_est']
                ml_estimates[anelastic_method][locname] = ml_est

                if res['record'] is not None:
                    ml_records.append(res['record'])

                if not large_scale_run or n_locations <= 1000:
                    # Build a minimal posterior-like dict for store_ensemble
                    post_stub = {
                        'phi': res['posterior_phi'],
                        'T': res['posterior_T'],
                    }
                    ensemble_pdf = store_ensemble(
                        ensemble_pdf, locname, anelastic_method,
                        res['p_joint'], post_stub, include_mxw=True,
                    )
                    ensemble_pdf_no_mxw = store_ensemble(
                        ensemble_pdf_no_mxw, locname, anelastic_method,
                        res['p_joint'], post_stub, include_mxw=False,
                    )
                    regional_fits[anelastic_method][locname] = {
                        'p_joint': res['p_joint'],
                        'phi_post': res['posterior_phi'],
                        'T_post': res['posterior_T'],
                    }

            # Generate posterior plots for parallel results (if requested)
            if save_individual_plots:
                obs_label = config.obs_types.replace('VsQ', 'VQ').replace('both', 'VQ')
                n_plotted = 0
                for res in par_results:
                    if res is None:
                        continue
                    if res['il'] % config.plot_every_n != 0:
                        continue
                    depth_km = None
                    if use_preloaded and seismic_model_data.depths is not None:
                        depth_km = float(seismic_model_data.depths[res['il']])
                    save_figure_for_posterior(
                        res['posterior'], sweep, res['locname'],
                        anelastic_method, output_dir, obs_label,
                        depth_km=depth_km,
                    )
                    n_plotted += 1
                print(f"     Saved {n_plotted} posterior plots to {output_dir}/")

            continue  # next anelastic_method — skip the sequential loop below
        
        # ----- Sequential path (original behavior) -----
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
                    
                    if use_vs and seismic_model_data.Vs is not None:
                        obs_vs = float(seismic_model_data.Vs[il])
                        sigma_vs = float(seismic_model_data.Vs_error[il]) if seismic_model_data.Vs_error is not None else config.default_vs_error
                    if use_q and seismic_model_data.Q is not None:
                        obs_q = float(seismic_model_data.Q[il])
                        sigma_q = float(seismic_model_data.Q_error[il]) if seismic_model_data.Q_error is not None else config.default_q_error
                        # Apply percent mode if Q_error wasn't already converted during loading
                        if seismic_model_data.Q_error is None and config.q_error_mode == 'percent':
                            sigma_q = obs_q * config.default_q_error / 100.0
                    
                    if first_run:
                        posterior, sweep = fit_preloaded_observations(
                            obs_vs, sigma_vs, obs_q, sigma_q,
                            (z_min, z_max), anelastic_method, grain_size_prior,
                            sweep_file=config.sweep_file,
                            melt_fraction_prior=melt_fraction_prior,
                            temperature_prior=temperature_prior,
                        )
                        first_run = False
                    else:
                        posterior, sweep = fit_preloaded_observations(
                            obs_vs, sigma_vs, obs_q, sigma_q,
                            (z_min, z_max), anelastic_method, grain_size_prior,
                            sweep=sweep,
                            melt_fraction_prior=melt_fraction_prior,
                            temperature_prior=temperature_prior,
                        )
                else:
                    # Load observations from files (original behavior)
                    # Optionally override Vs/Q with values from 1D Earth model
                    vs_ovr = float(vs_override_values[il]) if vs_override_values is not None and use_vs else None
                    svs_ovr = float(vs_override_errors[il]) if vs_override_errors is not None and use_vs else None
                    q_ovr = float(q_override_values[il]) if q_override_values is not None and use_q else None
                    sq_ovr = float(q_override_errors[il]) if q_override_errors is not None and use_q else None
                    if first_run:
                        posterior, sweep = fit_seismic_observations(
                            filenames, location, anelastic_method, grain_size_prior,
                            sweep_file=config.sweep_file,
                            melt_fraction_prior=melt_fraction_prior,
                            temperature_prior=temperature_prior,
                            obs_vs_override=vs_ovr,
                            sigma_vs_override=svs_ovr,
                            obs_q_override=q_ovr,
                            sigma_q_override=sq_ovr,
                        )
                        first_run = False
                    else:
                        posterior, sweep = fit_seismic_observations(
                            filenames, location, anelastic_method, grain_size_prior,
                            sweep=sweep,
                            melt_fraction_prior=melt_fraction_prior,
                            temperature_prior=temperature_prior,
                            obs_vs_override=vs_ovr,
                            sigma_vs_override=svs_ovr,
                            obs_q_override=q_ovr,
                            sigma_q_override=sq_ovr,
                        )
            except Exception as e:
                if not large_scale_run:
                    print(f"        Error fitting {locname}: {e}")
                continue
            
            # Determine obs_type label for filenames
            obs_label = config.obs_types.replace('VsQ', 'VQ').replace('both', 'VQ')
            
            # Save plots (only for small runs, respecting plot_every_n)
            if save_individual_plots and il % config.plot_every_n == 0:
                print("        saving plots...")
                # Get depth for title/filename
                depth_km = None
                if use_preloaded and seismic_model_data.depths is not None:
                    depth_km = float(seismic_model_data.depths[il])
                save_figure_for_posterior(
                    posterior, sweep, locname, anelastic_method, output_dir, obs_label,
                    depth_km=depth_km,
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
                # Viscosity (log10 Pa·s) — full posterior if available
                if 'log10_eta' in ml_est:
                    record['log10_eta_ml'] = ml_est['log10_eta']['ml']
                    record['log10_eta_std'] = ml_est['log10_eta']['std']
                    record['log10_eta_mean'] = ml_est['log10_eta']['mean']
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
            save_dir=inversion_dir,
        )
        
        plot_ensemble_pdfs(
            ensemble_pdf, ensemble_pdf_no_mxw,
            locs_array, names, colors,
            fig_prefix_dir, save_dir=inversion_dir,
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
        # If user set ml_csv_file explicitly, use that; otherwise put it
        # inside output_dir so the user only needs to specify output_dir.
        if config.ml_csv_file:
            csv_path = config.ml_csv_file
        else:
            csv_path = os.path.join(inversion_dir, 'ml_estimates.csv')
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        
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
    save_path = os.path.join(inversion_dir, f'{fig_prefix_dir}_ensembles.pkl')
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
        print("  " + "-" * 93)
        print(f"  {'Location':<18} {'T (°C)':<18} {'φ (melt frac)':<18} {'d (mm)':<18} {'log₁₀η (Pa·s)':<18}")
        print(f"  {'':<18} {'ML ± σ':<18} {'ML ± σ':<18} {'ML ± σ':<18} {'ML ± σ':<18}")
        print("  " + "-" * 93)
        
        for locname in names:
            if locname not in ml_estimates[method]:
                continue
            
            est = ml_estimates[method][locname]
            T_str = f"{est['T']['ml']:.0f} ± {est['T']['std']:.0f}"
            phi_str = f"{est['phi']['ml']:.4f} ± {est['phi']['std']:.4f}"
            gs_str = f"{est['gs']['ml_mm']:.2f} ± {est['gs']['std_mm']:.2f}"
            if 'log10_eta' in est:
                eta_str = f"{est['log10_eta']['ml']:.1f} ± {est['log10_eta']['std']:.1f}"
            else:
                eta_str = "N/A"
            
            print(f"  {locname:<18} {T_str:<18} {phi_str:<18} {gs_str:<18} {eta_str:<18}")
        
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

  # Load seismic model from a file (locations + Vs/Q, format auto-detected)
  python -m bayesian_fitting_py --location-mode model --vs-file model.csv

  # Separate Vs and Q sources (same or different file)
  python -m bayesian_fitting_py --location-mode model --vs-file model.nc --q-file q_model.nc

  # Use PREM Q values with a 3D Vs model
  python -m bayesian_fitting_py --location-mode model --vs-file model.mat --q-file prem
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
        choices=['log_uniform', 'log_normal'],
        help='Grain size prior type: log_uniform or log_normal'
    )
    parser.add_argument(
        '--gs-prior-mean-mm', type=float, default=None,
        help='Mean grain size in mm for log_normal prior (e.g. 1.0, 0.1, 4.0)'
    )
    parser.add_argument(
        '--gs-prior-std', type=float, default=None,
        help='Std dev in log-space for log_normal prior (default: 0.25)'
    )
    parser.add_argument(
        '--phi-prior-type', type=str, default=None,
        choices=['uniform', 'zero_melt', 'piecewise_depth', 'temperature_dependent'],
        help='Melt fraction prior: uniform (default), zero_melt, piecewise_depth, temperature_dependent'
    )
    parser.add_argument(
        '--phi-onset-depth', type=float, default=None,
        help='Onset depth (km) for piecewise_depth melt prior (default: 80.0)'
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
        '--force-plots', action='store_true',
        help='Force posterior plots even for large-scale runs (>20 locations)'
    )
    parser.add_argument(
        '--obs-types', type=str, default=None,
        choices=['Vs', 'Q', 'VsQ'],
        help='Observation types to use: Vs only, Q only, or both (VsQ)'
    )
    parser.add_argument(
        '--vs-file', type=str, default=None,
        help=(
            'Vs data source: path to .mat/.csv/.nc file, or a built-in '
            'model name (prem, stw105).  In model mode also provides the '
            'location grid.'
        )
    )
    parser.add_argument(
        '--q-file', type=str, default=None,
        help=(
            'Q data source: path to .mat/.csv/.nc file, or a built-in '
            'model name (prem, stw105).  Can be the same file as --vs-file.'
        )
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
            'locations_file (load locations from file), '
            'model (locations + seismic data from vs_file, format auto-detected)'
        )
    )
    parser.add_argument(
        '--location-file', type=str, default=None,
        help='Path to location file (for --location-mode locations_file). Format: lon,lat[,name],z_min,z_max'
    )
    parser.add_argument(
        '--seismic-model-file', type=str, default=None,
        help='Deprecated: use --vs-file instead.  Sets both vs_file and q_file for backward compat.'
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
    parser.add_argument(
        '--parallel', '-j', type=int, default=None, metavar='N',
        help=(
            'Number of parallel worker processes for large-scale runs. '
            '0 = auto (all CPU cores), 1 = sequential (default). '
            'Only used with preloaded model modes (csv_model, mat_model, netcdf_model).'
        )
    )
    parser.add_argument(
        '--sweep-file', type=str, default=None,
        help='Path to sweep .npz file (overrides the sweep_file in config)'
    )
    parser.add_argument(
        '--run-tag', type=str, default=None,
        help=(
            'Inversion output subdirectory tag. '
            "'none' = inversion_results (default), "
            "'auto' = auto-generated from parameters, "
            'or any custom string.'
        )
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
        config.gs_prior_type = args.gs_prior
    if args.gs_prior_mean_mm is not None:
        config.gs_prior_mean_mm = args.gs_prior_mean_mm
    if args.gs_prior_std is not None:
        config.gs_prior_std = args.gs_prior_std
    if args.phi_prior_type is not None:
        config.phi_prior_type = args.phi_prior_type
    if args.phi_onset_depth is not None:
        config.phi_onset_depth_km = args.phi_onset_depth
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.no_plots:
        config.save_plots = False
    if args.force_plots:
        config.force_plots = True
    if args.obs_types is not None:
        config.obs_types = args.obs_types
    if args.vs_file is not None:
        config.vs_file = args.vs_file
    if args.q_file is not None:
        config.q_file = args.q_file
    if args.anelastic_methods is not None:
        config.anelastic_methods = parse_anelastic_methods(args.anelastic_methods)
    if args.location_mode is not None:
        config.location_mode = args.location_mode
    if args.location_file is not None:
        config.location_file = args.location_file
    if args.seismic_model_file is not None:
        # Backward compat: --seismic-model-file sets both vs_file and q_file
        config.vs_file = args.seismic_model_file
        config.q_file = args.seismic_model_file
        if config.location_mode == 'manual':
            config.location_mode = 'model'
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
    if args.parallel is not None:
        config.parallel_workers = args.parallel
    if args.sweep_file is not None:
        config.sweep_file = args.sweep_file
    if args.run_tag is not None:
        config.run_tag = args.run_tag
    
    # Run the inversion
    results = run_bayesian_inversion(config, working_dir=args.data_dir)
    
    n_locations = len(results.get('names', config.names))
    print("\nInversion complete!")
    print(f"Processed {n_locations} locations with {len(config.anelastic_methods)} anelastic method(s)")
    print(f"Anelastic methods used: {', '.join(config.anelastic_methods)}")


if __name__ == '__main__':
    main()