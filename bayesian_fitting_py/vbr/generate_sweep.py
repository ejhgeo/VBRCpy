"""
Parameter sweep generation for VBR Bayesian inversion.

This module generates look-up tables of Vs and Q as a function of
temperature, melt fraction, and grain size for use in Bayesian inversion.

The sweep generation is computationally intensive and should be run
separately from the inversion itself.
"""

import numpy as np
from scipy.io import savemat
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import os

from .core import VBR, StateVariables
from .thermal import calculate_solidus_K, _load_earth_model
from .params import C2K


def load_density_profile(model: str = 'prem',
                         filepath: Optional[str] = None) -> interp1d:
    """
    Load a depth-dependent density profile and return an interpolation function.

    Parameters
    ----------
    model : str
        Density model to use:
        - ``'prem'``: built-in PREM reference model (Dziewonski & Anderson, 1981)
        - ``'stw105'``: built-in STW105 reference model (Kustowski et al., 2008)
        - ``'custom'``: user-supplied file (same 5-column whitespace-delimited
          format as built-in models: ``radius depth density Vs Qmu``)
    filepath : str, optional
        Path to a custom file.  Required when *model* is ``'custom'``.

    Returns
    -------
    scipy.interpolate.interp1d
        Interpolation function mapping depth in **meters** to density in
        kg/m³.  Extrapolates linearly outside the data range.
    """
    depth_m, density = _load_earth_model(model, custom_file=filepath)
    return interp1d(depth_m, density,
                    kind='linear', fill_value='extrapolate',
                    bounds_error=False)


@dataclass
class SweepParams:
    """
    Parameters for generating a VBR parameter sweep.
    
    Attributes
    ----------
    T : array_like
        Temperature values in degrees Celsius
    phi : array_like
        Melt fraction values (volume fraction, 0-1)
    gs : array_like  
        Grain size values in micrometers
    z_min : float
        Minimum depth in km (default: 50)
    z_max : float
        Maximum depth in km (default: 170)
    n_z : int
        Number of depth points (default: 100)
    per_bw_min : float
        Minimum period (maximum frequency) in seconds (default: 20)
    per_bw_max : float
        Maximum period (minimum frequency) in seconds (default: 150)
    n_freq : int
        Number of frequency points (default: 10)
    rho : float
        Density in kg/m^3 used when density_model='constant' (default: 3300)
    density_model : str
        Density model: 'constant' (uniform rho), 'prem' (depth-dependent PREM),
        'stw105' (depth-dependent STW105), or 'custom' (user-provided file).
        Default: 'constant'.
    density_file : str or None
        Path to custom density file (required when density_model='custom').
        Must use the same 5-column whitespace-delimited format as the
        bundled models (``radius depth density Vs Qmu``).
    sig_MPa : float
        Differential stress in MPa (default: 0.1)
    Ch2o : float
        Water content in ppm (default: 0)
    anelastic_methods : list of str
        Anelastic methods to calculate (default: all available)
    viscous_method : str
        Viscous flow law for stored viscosity: 'HK2003' (Hirth & Kohlstedt
        2003, diffusion+dislocation+GBS) or 'xfit_premelt' (Yamauchi & Takei
        2016 diffusion creep).  Default: 'HK2003'.  Note: the xfit_premelt
        *anelastic* Q calculation always uses its own viscosity internally
        regardless of this setting.
    eburgers_method : str
        eBurgers calculation method: 'PointWise' or 'FastBurger' (default: 'PointWise')
    gs_type : str
        Grain size spacing: 'linear' or 'log' (default: 'log')
    output_file : str
        Output filename (default: 'sweep.mat')
    elastic_method : str
        Base elastic method: 'anharmonic' (linear Taylor expansion, olivine
        parameters — suitable for upper mantle) or 'cammarano2003' (Cammarano
        et al. 2003 finite-strain mineral physics with depth-dependent
        mineralogy — suitable for upper mantle through lower mantle).
        Default: 'anharmonic'.
    temperature_scaling : str
        Elastic temperature derivative source: 'isaak', 'cammarano', or 'upper_mantle' 
        (default: 'isaak')
    pressure_scaling : str
        Elastic pressure derivative source: 'cammarano', 'abramson', or 'upper_mantle'
        (default: 'cammarano')
    reference_scaling : str
        Elastic reference values: 'default' (STP olivine) or 'upper_mantle' 
        (Abers and Hacker 2016 pyrolitic upper mantle) (default: 'default')
    """
    T: np.ndarray = field(default_factory=lambda: np.arange(1100, 1651, 25))
    phi: np.ndarray = field(default_factory=lambda: np.concatenate([
        [0], np.arange(0.0001, 0.001, 0.0001),
        np.arange(0.001, 0.01, 0.001),
        np.arange(0.01, 0.051, 0.005)
    ]))
    gs: np.ndarray = field(default_factory=lambda: np.logspace(np.log10(100), np.log10(50000), 20))
    z_min: float = 50.0
    z_max: float = 170.0
    n_z: int = 100
    per_bw_min: float = 20.0
    per_bw_max: float = 150.0
    n_freq: int = 10
    # Frequency logspace exponents to match MATLAB's logspace(-2.2, -1.3, 10)
    # When set, these override per_bw_min/max for frequency calculation
    freq_log_min: float = -2.2  # log10(f_min)
    freq_log_max: float = -1.3  # log10(f_max)
    rho: float = 3300.0
    density_model: str = 'constant'  # 'constant', 'prem', 'prem_nocrust', 'stw105', or 'custom'
    density_file: Optional[str] = None  # path to custom density CSV
    sig_MPa: float = 0.1
    Ch2o: float = 0.0
    anelastic_methods: List[str] = field(default_factory=lambda: [
        'eburgers_psp', 'andrade_psp', 'xfit_mxw', 'xfit_premelt'
    ])
    viscous_method: str = 'HK2003'  # Options: 'HK2003', 'xfit_premelt'
    eburgers_method: str = 'FastBurger'
    # xfit_premelt melt effect mode:
    # 0 = YT2016 (poroelastic via external anh_poro only, default)
    # 1 = YT2024 (direct melt effects on anelasticity via Beta_B, Beta_P, poro_Lambda)
    include_direct_melt_effect: int = 0
    gs_type: str = 'log'
    # Base output directory.  When set, output_file and plot_lut_dir are
    # derived automatically (sweep.npz and lut_plots/ inside this dir)
    # unless the user overrides them explicitly.
    output_dir: Optional[str] = None
    output_file: str = 'sweep.mat'
    # Elastic method: 'anharmonic' (linear Taylor expansion, olivine-based) or
    # 'cammarano2003' (finite-strain mineral physics with depth-dependent mineralogy)
    elastic_method: str = 'anharmonic'  # Options: 'anharmonic', 'cammarano2003'
    # Elastic scaling options (for anharmonic calculations only)
    temperature_scaling: str = 'isaak'  # Options: 'isaak', 'cammarano', 'upper_mantle'
    pressure_scaling: str = 'cammarano'  # Options: 'cammarano', 'abramson', 'upper_mantle'
    reference_scaling: str = 'default'  # Options: 'default', 'upper_mantle'
    solidus_method: str = 'hirschmann'  # Options: 'hirschmann', 'katz', 'yk2001'
    # Lookup-table plot options
    plot_lut: bool = False            # generate LUT figures during sweep
    plot_lut_dir: str = 'lut_plots'   # output directory for LUT figures
    plot_lut_every_n: int = 1         # plot every n-th depth (1 = all)
    
    def __post_init__(self):
        """Convert to numpy arrays and resolve output_dir inheritance."""
        self.T = np.atleast_1d(self.T).astype(float)
        self.phi = np.atleast_1d(self.phi).astype(float)
        self.gs = np.atleast_1d(self.gs).astype(float)
        # If output_dir is set, derive output_file and plot_lut_dir from it
        # (unless the user already provided explicit values)
        if self.output_dir is not None:
            if self.output_file == 'sweep.mat':
                self.output_file = os.path.join(self.output_dir, 'sweep.npz')
            if self.plot_lut_dir == 'lut_plots':
                self.plot_lut_dir = os.path.join(self.output_dir, 'lut_plots')


def generate_parameter_sweep(
    params: Optional[SweepParams] = None,
    config_file: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate a parameter sweep for VBR Bayesian inversion.
    
    Calculates mean Vs and Q at a range of depths for combinations of
    temperature, melt fraction, and grain size. This creates a look-up
    table for efficient Bayesian inversion.
    
    Parameters
    ----------
    params : SweepParams, optional
        Sweep parameters. If None, uses defaults or loads from config_file.
    config_file : str, optional
        Path to YAML configuration file with sweep parameters.
    verbose : bool
        Print progress information (default: True)
        
    Returns
    -------
    dict
        Sweep structure with:
        - T, phi, gs: parameter vectors
        - z: depth vector in meters
        - P_GPa: pressure at each depth
        - Box: nested dict of {method: {meanVs, meanQ}} with shape (nT, nphi, ngs, nz)
        - state_names: ['T', 'phi', 'gs']
        - per_bw_min, per_bw_max: period bounds
        - gs_params: grain size parameter info
        
    Examples
    --------
    >>> params = SweepParams(
    ...     T=np.arange(1100, 1601, 50),
    ...     phi=np.array([0, 0.001, 0.01]),
    ...     gs=np.logspace(2, 4, 10),
    ...     z_min=75, z_max=150,
    ... )
    >>> sweep = generate_parameter_sweep(params)
    >>> print(sweep['Box']['eburgers_psp']['meanVs'].shape)
    
    See Also
    --------
    save_sweep : Save sweep to .mat file
    load_sweep_params_from_yaml : Load parameters from YAML file
    """
    if params is None:
        if config_file is not None:
            params = load_sweep_params_from_yaml(config_file)
        else:
            params = SweepParams()
    
    if verbose:
        print("=" * 60)
        print("VBR Parameter Sweep Generator")
        print("=" * 60)
        print(f"Elastic method: {params.elastic_method}")
        if params.elastic_method == 'anharmonic':
            print(f"  Temperature derivatives from: {params.temperature_scaling}")
            print(f"  Pressure derivatives from: {params.pressure_scaling}")
            print(f"  Reference scaling set to: {params.reference_scaling}")
        elif params.elastic_method == 'cammarano2003':
            print(f"  Cammarano et al. (2003) finite-strain mineral physics")
            print(f"  Composition: pyrolite (depth-dependent assemblage)")
        print(f"Temperature: {len(params.T)} values from {params.T.min():.0f} to {params.T.max():.0f} °C")
        print(f"Melt fraction: {len(params.phi)} values from {params.phi.min():.4f} to {params.phi.max():.4f}")
        print(f"Grain size: {len(params.gs)} values from {params.gs.min():.0f} to {params.gs.max():.0f} μm")
        print(f"Depth range: {params.z_min:.0f} to {params.z_max:.0f} km ({params.n_z} points)")
        print(f"Period range: {params.per_bw_min:.0f} to {params.per_bw_max:.0f} s")
        print(f"Methods: {', '.join(params.anelastic_methods)}")
        if 'xfit_premelt' in params.anelastic_methods:
            melt_mode = 'YT2024 (direct melt effects)' if params.include_direct_melt_effect else 'YT2016 (default)'
            print(f"  xfit_premelt melt mode: {melt_mode}")
        # Density info
        if params.density_model == 'constant':
            print(f"Density: constant {params.rho:.0f} kg/m³")
        else:
            print(f"Density: depth-dependent ({params.density_model})")
            if params.density_file:
                print(f"  Custom file: {params.density_file}")
        n_total = len(params.T) * len(params.phi) * len(params.gs) * params.n_z
        print(f"Total calculations: {n_total:,}")
        print("=" * 60)
    
    # Create depth array (meters)
    z = np.linspace(params.z_min, params.z_max, params.n_z) * 1e3  # Convert to meters

    # Compute pressure and per-depth density
    if params.density_model == 'constant':
        P_GPa = z * params.rho * 9.8 / 1e9
        rho_at_depth = np.full(params.n_z, params.rho)
    else:
        rho_func = load_density_profile(params.density_model, params.density_file)
        rho_at_depth = rho_func(z)
        if verbose:
            print(f"  Density range over sweep depths: "
                  f"{rho_at_depth.min():.0f} – {rho_at_depth.max():.0f} kg/m³")
        # Lithostatic pressure: P(z) = g * ∫₀ᶻ ρ(z') dz'
        # Use a fine grid from surface to deepest sweep depth for accuracy
        n_fine = max(1000, params.n_z * 10)
        z_fine = np.linspace(0, z.max(), n_fine)
        rho_fine = rho_func(z_fine)
        P_fine_Pa = np.zeros(n_fine)
        P_fine_Pa[1:] = cumulative_trapezoid(rho_fine * 9.8, z_fine)
        P_GPa = np.interp(z, z_fine, P_fine_Pa / 1e9)
    
    # Frequency array - use explicit logspace exponents to match MATLAB exactly
    # MATLAB uses: VBR.in.SV.f = logspace(-2.2, -1.3, 10)
    f = np.logspace(params.freq_log_min, params.freq_log_max, params.n_freq)
    
    # Create state variable grids
    n_T = len(params.T)
    n_phi = len(params.phi)
    n_gs = len(params.gs)
    n_P = len(P_GPa)
    
    # Initialize output Box structure
    # Box[method]['meanVs'], Box[method]['meanQ'], Box[method]['meanEta']
    # all have shape (nT, nphi, ngs, nP)
    Box = {}
    for method in params.anelastic_methods:
        Box[method] = {
            'meanVs': np.zeros((n_T, n_phi, n_gs, n_P)),
            'meanQ': np.zeros((n_T, n_phi, n_gs, n_P)),
            'meanEta': np.zeros((n_T, n_phi, n_gs, n_P)),
        }
    
    # Create 3D grids for T, phi, gs
    T_grid, phi_grid, gs_grid = np.meshgrid(
        params.T, params.phi, params.gs, indexing='ij'
    )
    
    T_K = T_grid + C2K  # Convert to Kelvin
    
    # Loop over pressure/depth
    start_time = time.time()
    
    for i_P in range(n_P):
        if verbose:
            print(f"  Calculating depth step {i_P + 1}/{n_P} (z = {z[i_P]/1e3:.1f} km)...", end='', flush=True)
        
        step_start = time.time()
        
        P = P_GPa[i_P]
        
        # Create pressure and other state variable arrays matching T_grid shape
        P_arr = np.full(T_grid.shape, P)
        rho_arr = np.full(T_grid.shape, rho_at_depth[i_P])
        sig_arr = np.full(T_grid.shape, params.sig_MPa)
        Ch2o_arr = np.full(T_grid.shape, params.Ch2o)
        
        # Calculate solidus at this pressure/depth
        Tsolidus_K = calculate_solidus_K(
            P, method=params.solidus_method,
            depth_km=z[i_P] / 1e3,
            density_model=params.density_model,
            density_rho=params.rho,
            density_file=params.density_file,
        )
        Tsolidus_arr = np.full(T_grid.shape, Tsolidus_K)
        
        # Create state variables
        sv = StateVariables(
            T_K=T_K,
            P_GPa=P_arr,
            rho=rho_arr,
            dg_um=gs_grid,
            phi=phi_grid,
            sig_MPa=sig_arr,
            f=f,
            Ch2o=Ch2o_arr,
            Tsolidus_K=Tsolidus_arr,
        )
        
        # Determine elastic methods based on user selection
        base_elastic = params.elastic_method  # 'anharmonic' or 'cammarano2003'
        if np.any(params.phi > 0):
            elastic_methods = [base_elastic, 'anh_poro']
        else:
            elastic_methods = [base_elastic]
        
        # Build viscous methods list: always include the user's chosen method,
        # plus xfit_premelt if its anelastic method is requested (it needs its
        # own viscosity internally for Q calculation).
        viscous_methods_list = [params.viscous_method]
        if 'xfit_premelt' in params.anelastic_methods and 'xfit_premelt' not in viscous_methods_list:
            viscous_methods_list.append('xfit_premelt')

        # Run VBR
        vbr = VBR(
            sv,
            elastic_methods=elastic_methods,
            anelastic_methods=params.anelastic_methods,
            viscous_methods=viscous_methods_list,
        )
        
        # Set elastic scaling options (only relevant for anharmonic method)
        if 'anharmonic' in elastic_methods:
            vbr.input['elastic']['anharmonic']['temperature_scaling'] = params.temperature_scaling
            vbr.input['elastic']['anharmonic']['pressure_scaling'] = params.pressure_scaling
            vbr.input['elastic']['anharmonic']['reference_scaling'] = params.reference_scaling
        
        # Set eBurgers method
        if 'eburgers_psp' in params.anelastic_methods:
            vbr.input['anelastic']['eburgers_psp']['method'] = params.eburgers_method
        
        # Set xfit_premelt direct melt effect mode (YT2016 vs YT2024)
        if 'xfit_premelt' in params.anelastic_methods:
            vbr.input['anelastic']['xfit_premelt']['include_direct_melt_effect'] = params.include_direct_melt_effect
        
        vbr.run()
        
        # Extract mean Vs, Q, and viscosity for each method.
        # Stored viscosity uses the user's chosen viscous_method for all
        # anelastic methods.  (xfit_premelt Q always uses its own viscosity
        # internally regardless of this setting.)
        vm = params.viscous_method
        if vm == 'HK2003' and 'HK2003' in vbr.output.get('viscous', {}):
            eta_store = vbr.output['viscous']['HK2003']['eta_total']
        elif vm == 'xfit_premelt' and 'xfit_premelt' in vbr.output.get('viscous', {}):
            eta_store = vbr.output['viscous']['xfit_premelt']['diff']['eta']
        else:
            # Fallback: try HK2003, then xfit_premelt
            if 'HK2003' in vbr.output.get('viscous', {}):
                eta_store = vbr.output['viscous']['HK2003']['eta_total']
            elif 'xfit_premelt' in vbr.output.get('viscous', {}):
                eta_store = vbr.output['viscous']['xfit_premelt']['diff']['eta']
            else:
                eta_store = None

        for method in params.anelastic_methods:
            if method in vbr.output['anelastic']:
                result = vbr.output['anelastic'][method]
                # V has shape (nT, nphi, ngs, nfreq) - average over frequency
                V = result['V'] / 1e3  # Convert m/s to km/s
                Q = result['Q']
                
                # Mean over frequency dimension (last axis)
                Box[method]['meanVs'][:, :, :, i_P] = np.mean(V, axis=-1)
                Box[method]['meanQ'][:, :, :, i_P] = np.mean(Q, axis=-1)
                
                # Viscosity (Pa·s) — same viscous model for all methods
                if eta_store is not None:
                    Box[method]['meanEta'][:, :, :, i_P] = eta_store
        
        step_time = time.time() - step_start
        if verbose:
            print(f" done ({step_time:.1f}s)")
    
    total_time = time.time() - start_time
    if verbose:
        print("=" * 60)
        print(f"Sweep complete in {total_time / 60:.1f} minutes")
        print("=" * 60)
    
    # Build output structure
    sweep = {
        'T': params.T,
        'phi': params.phi,
        'gs': params.gs,
        'z': z,
        'P_GPa': P_GPa,
        'rho': rho_at_depth,  # density at each depth point (kg/m³)
        'density_model': params.density_model,
        'per_bw_min': params.per_bw_min,
        'per_bw_max': params.per_bw_max,
        'Box': Box,
        'state_names': ['T', 'phi', 'gs'],
        'gs_params': {
            'type': params.gs_type,
            'gsmin': params.gs.min(),
            'gsmax': params.gs.max(),
            'gsref': 1e3,
        },
    }
    
    return sweep


def save_sweep(sweep: Dict[str, Any], filename: str, verbose: bool = True) -> None:
    """
    Save a parameter sweep to file.
    
    Supported formats:
    - .mat: MATLAB-compatible format (works with existing load_sweep_data)
    - .npz: NumPy compressed archive (recommended for Python-only workflows)
    - .pkl/.pickle: Python pickle format
    
    Parameters
    ----------
    sweep : dict
        Sweep structure from generate_parameter_sweep
    filename : str
        Output filename. Extension determines format.
    verbose : bool
        Print confirmation message
    """
    out_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(out_dir, exist_ok=True)

    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.mat':
        _save_sweep_mat(sweep, filename, verbose)
    elif ext == '.npz':
        _save_sweep_npz(sweep, filename, verbose)
    elif ext in ['.pkl', '.pickle']:
        _save_sweep_pickle(sweep, filename, verbose)
    else:
        # Default to .mat for backward compatibility
        if verbose:
            print(f"Unknown extension '{ext}', saving as .mat format")
        _save_sweep_mat(sweep, filename, verbose)


def _save_sweep_mat(sweep: Dict[str, Any], filename: str, verbose: bool = True) -> None:
    """
    Save sweep to MATLAB-compatible .mat file.
    
    Creates a structure that matches the original MATLAB generate_parameter_sweep.m output,
    with Box as a 3D array of structs (nT x nphi x ngs) where each element has
    method attributes containing meanVs and meanQ arrays.
    """
    from scipy.io import savemat
    
    n_T = len(sweep['T'])
    n_phi = len(sweep['phi'])
    n_gs = len(sweep['gs'])
    n_z = len(sweep['z'])
    methods = list(sweep['Box'].keys())
    
    # Create Box as object array matching MATLAB structure
    # In MATLAB: Box(i_T, i_phi, i_gs).<method>.meanVs is a (n_z,) array
    # scipy.io.savemat with struct_as_record=False expects nested dicts
    
    # Build the Box structure as a numpy object array of dicts
    Box = np.empty((n_T, n_phi, n_gs), dtype=object)
    
    for i_T in range(n_T):
        for i_phi in range(n_phi):
            for i_gs in range(n_gs):
                elem = {}
                for method in methods:
                    method_data = {
                        'meanVs': sweep['Box'][method]['meanVs'][i_T, i_phi, i_gs, :],
                        'meanQ': sweep['Box'][method]['meanQ'][i_T, i_phi, i_gs, :],
                    }
                    if 'meanEta' in sweep['Box'][method]:
                        method_data['meanEta'] = sweep['Box'][method]['meanEta'][i_T, i_phi, i_gs, :]
                    elem[method] = method_data
                Box[i_T, i_phi, i_gs] = elem
    
    # Build the sweep structure
    mat_sweep = {
        'T': sweep['T'],
        'phi': sweep['phi'],
        'gs': sweep['gs'],
        'z': sweep['z'],
        'P_GPa': sweep['P_GPa'],
        'rho': sweep['rho'],
        'density_model': sweep.get('density_model', 'constant'),
        'per_bw_min': float(sweep['per_bw_min']),
        'per_bw_max': float(sweep['per_bw_max']),
        'state_names': np.array(sweep['state_names'], dtype=object),
        'gs_params': sweep['gs_params'],
        'Box': Box,
    }
    
    # Save with savemat
    savemat(filename, {'sweep': mat_sweep}, do_compression=True)
    
    if verbose:
        print(f"Saved sweep to {filename} (MATLAB format)")
        file_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"File size: {file_size:.1f} MB")


def _save_sweep_npz(sweep: Dict[str, Any], filename: str, verbose: bool = True) -> None:
    """Save sweep to NumPy compressed archive."""
    # Flatten the nested structure for npz storage
    save_dict = {
        'T': sweep['T'],
        'phi': sweep['phi'],
        'gs': sweep['gs'],
        'z': sweep['z'],
        'P_GPa': sweep['P_GPa'],
        'rho': sweep['rho'],
        'density_model': np.array(sweep.get('density_model', 'constant')),
        'per_bw_min': np.array(sweep['per_bw_min']),
        'per_bw_max': np.array(sweep['per_bw_max']),
        'state_names': np.array(sweep['state_names']),
        'gs_params_type': np.array(sweep['gs_params']['type']),
        'gs_params_gsmin': np.array(sweep['gs_params']['gsmin']),
        'gs_params_gsmax': np.array(sweep['gs_params']['gsmax']),
        'gs_params_gsref': np.array(sweep['gs_params']['gsref']),
        'anelastic_methods': np.array(list(sweep['Box'].keys())),
    }
    
    # Add Box data with method names as keys
    for method, data in sweep['Box'].items():
        save_dict[f'Box_{method}_meanVs'] = data['meanVs']
        save_dict[f'Box_{method}_meanQ'] = data['meanQ']
        if 'meanEta' in data:
            save_dict[f'Box_{method}_meanEta'] = data['meanEta']
    
    np.savez_compressed(filename, **save_dict)
    
    if verbose:
        print(f"Saved sweep to {filename} (NumPy format)")
        file_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"File size: {file_size:.1f} MB")


def _save_sweep_pickle(sweep: Dict[str, Any], filename: str, verbose: bool = True) -> None:
    """Save sweep to pickle file."""
    import pickle
    
    with open(filename, 'wb') as f:
        pickle.dump(sweep, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    if verbose:
        print(f"Saved sweep to {filename} (pickle format)")
        file_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"File size: {file_size:.1f} MB")


def load_sweep(filename: str) -> Dict[str, Any]:
    """
    Load a parameter sweep from file.
    
    Supports .mat, .npz, and .pkl/.pickle formats.
    
    Parameters
    ----------
    filename : str
        Path to sweep file
        
    Returns
    -------
    dict
        Sweep structure with:
        - T, phi, gs: parameter vectors
        - z: depth vector
        - P_GPa: pressure at each depth  
        - Box: nested dict of {method: {meanVs, meanQ}}
        - state_names, per_bw_min, per_bw_max, gs_params
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.mat':
        return _load_sweep_mat(filename)
    elif ext == '.npz':
        return _load_sweep_npz(filename)
    elif ext in ['.pkl', '.pickle']:
        return _load_sweep_pickle(filename)
    else:
        raise ValueError(f"Unknown file extension: {ext}. Supported: .mat, .npz, .pkl")


def _load_sweep_mat(filename: str) -> Dict[str, Any]:
    """Load sweep from MATLAB .mat file."""
    from scipy.io import loadmat
    
    mat_data = loadmat(filename, squeeze_me=True, struct_as_record=False)
    sweep_obj = mat_data['sweep']
    
    sweep = {}
    
    # Extract basic fields
    for attr in ['T', 'phi', 'gs', 'z', 'P_GPa', 'rho', 'per_bw_max', 'per_bw_min']:
        if hasattr(sweep_obj, attr):
            value = getattr(sweep_obj, attr)
            sweep[attr] = np.atleast_1d(value)
    # Density model metadata (may not exist in older files)
    if hasattr(sweep_obj, 'density_model'):
        sweep['density_model'] = str(sweep_obj.density_model)
    else:
        sweep['density_model'] = 'constant'
    
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
    has_eta = False
    for method in methods:
        box[method] = {
            'meanVs': np.zeros((nT, nphi, ngs, nz)),
            'meanQ': np.zeros((nT, nphi, ngs, nz)),
            'meanEta': np.zeros((nT, nphi, ngs, nz)),
        }
    
    # Try to extract from Box structure
    if hasattr(sweep_obj, 'Box'):
        box_obj = sweep_obj.Box
        if hasattr(box_obj, '__iter__') and not isinstance(box_obj, str):
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
                        if hasattr(method_obj, 'meanEta'):
                            eta_vals = np.atleast_1d(method_obj.meanEta)
                            box[method]['meanEta'][i_T, i_phi, i_gs, :] = eta_vals
                            has_eta = True
    
    # If no eta was found in the sweep file, remove the empty arrays
    if not has_eta:
        for method in methods:
            del box[method]['meanEta']
    
    sweep['Box'] = box
    return sweep


def _load_sweep_npz(filename: str) -> Dict[str, Any]:
    """Load sweep from NumPy .npz file."""
    data = np.load(filename, allow_pickle=True)
    
    sweep = {
        'T': data['T'],
        'phi': data['phi'],
        'gs': data['gs'],
        'z': data['z'],
        'P_GPa': data['P_GPa'],
        'rho': data['rho'] if 'rho' in data else np.full(len(data['z']), 3300.0),
        'density_model': str(data['density_model']) if 'density_model' in data else 'constant',
        'per_bw_min': float(data['per_bw_min']),
        'per_bw_max': float(data['per_bw_max']),
        'state_names': list(data['state_names']),
        'gs_params': {
            'type': str(data['gs_params_type']),
            'gsmin': float(data['gs_params_gsmin']),
            'gsmax': float(data['gs_params_gsmax']),
            'gsref': float(data['gs_params_gsref']),
        },
    }
    
    # Reconstruct Box structure
    methods = list(data['anelastic_methods'])
    box = {}
    for method in methods:
        box[method] = {
            'meanVs': data[f'Box_{method}_meanVs'],
            'meanQ': data[f'Box_{method}_meanQ'],
        }
        eta_key = f'Box_{method}_meanEta'
        if eta_key in data:
            box[method]['meanEta'] = data[eta_key]
    
    sweep['Box'] = box
    return sweep


def _load_sweep_pickle(filename: str) -> Dict[str, Any]:
    """Load sweep from pickle file."""
    import pickle
    
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_sweep_params_from_yaml(config_file: str) -> SweepParams:
    """
    Load sweep parameters from a YAML configuration file.
    
    Parameters
    ----------
    config_file : str
        Path to YAML configuration file
        
    Returns
    -------
    SweepParams
        Sweep parameters
        
    Example YAML format
    -------------------
    ```yaml
    sweep_generation:
      temperature:
        min: 1100
        max: 1600
        step: 25
      melt_fraction:
        values: [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05]
      grain_size:
        min: 100  # micrometers
        max: 50000
        n_points: 20
        type: log
      depth:
        min: 50  # km
        max: 170
        n_points: 100
      period:
        min: 20  # seconds
        max: 150
      frequency:
        n_points: 10
        log_min: -2.2  # log10(f_min in Hz)
        log_max: -1.3  # log10(f_max in Hz)
      # Physical parameters
      rho: 3300.0  # density in kg/m^3 (used when density_model is 'constant')
      # Density model: 'constant', 'prem', or 'custom'
      density_model: constant
      # density_file: /path/to/custom.csv  # required when density_model is 'custom'
      sig_MPa: 0.1  # differential stress in MPa
      Ch2o: 0.0  # water content in ppm
      # Methods
      anelastic_methods:
        - eburgers_psp
        - andrade_psp
        - xfit_premelt
      # Viscous method for stored viscosity profiles:
      # 'HK2003' (Hirth & Kohlstedt 2003, default) or 'xfit_premelt' (YT2016)
      # Note: xfit_premelt Q always uses its own viscosity internally.
      viscous_method: HK2003
      eburgers_method: FastBurger  # or 'PointWise'
      # xfit_premelt direct melt effect (Yamauchi & Takei 2024):
      #   0 = YT2016 mode (poroelastic effect via external anh_poro only, default)
      #   1 = YT2024 mode (direct melt effects on anelasticity: Beta_B, Beta_P, poro_Lambda)
      include_direct_melt_effect: 0
      # Elastic scaling options
      elastic:
        # Method: 'anharmonic' (linear Taylor, olivine) or 'cammarano2003'
        # (finite-strain mineral physics with depth-dependent mineralogy)
        method: anharmonic
        # Temperature derivative source: 'isaak', 'cammarano', or 'upper_mantle'
        temperature_scaling: isaak
        # Pressure derivative source: 'cammarano', 'abramson', or 'upper_mantle'
        pressure_scaling: cammarano
        # Reference values: 'default' (STP olivine) or 'upper_mantle' (Abers & Hacker 2016)
        reference_scaling: default
      # Output directory — when set, output_file and plot_lut output_dir
      # are automatically placed inside this directory (unless overridden).
      output_dir: my_output/
      # output_file: my_sweep.mat  # optional override (default: output_dir/sweep.npz)
    ```
    """
    import yaml
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'sweep_generation' not in config:
        raise ValueError("Config file must have 'sweep_generation' section")
    
    cfg = config['sweep_generation']
    
    # Parse temperature
    if 'temperature' in cfg:
        t_cfg = cfg['temperature']
        if 'values' in t_cfg:
            T = np.array(t_cfg['values'])
        else:
            T = np.arange(t_cfg['min'], t_cfg['max'] + t_cfg.get('step', 25), t_cfg.get('step', 25))
    else:
        T = np.arange(1100, 1651, 25)
    
    # Parse melt fraction
    if 'melt_fraction' in cfg:
        phi_cfg = cfg['melt_fraction']
        if 'values' in phi_cfg:
            phi = np.array(phi_cfg['values'])
        else:
            phi = np.linspace(phi_cfg['min'], phi_cfg['max'], phi_cfg.get('n_points', 20))
    else:
        phi = np.concatenate([[0], np.logspace(-4, -1.3, 20)])
    
    # Parse grain size
    if 'grain_size' in cfg:
        gs_cfg = cfg['grain_size']
        if 'values' in gs_cfg:
            gs = np.array(gs_cfg['values'])
        elif gs_cfg.get('type', 'log') == 'log':
            gs = np.logspace(np.log10(gs_cfg['min']), np.log10(gs_cfg['max']), gs_cfg.get('n_points', 20))
        else:
            gs = np.linspace(gs_cfg['min'], gs_cfg['max'], gs_cfg.get('n_points', 20))
        gs_type = gs_cfg.get('type', 'log')
    else:
        gs = np.logspace(2, np.log10(50000), 20)
        gs_type = 'log'
    
    # Parse depth
    if 'depth' in cfg:
        d_cfg = cfg['depth']
        z_min = d_cfg.get('min', 50)
        z_max = d_cfg.get('max', 170)
        n_z = d_cfg.get('n_points', 100)
    else:
        z_min, z_max, n_z = 50, 170, 100
    
    # Parse period
    if 'period' in cfg:
        p_cfg = cfg['period']
        per_bw_min = p_cfg.get('min', 20)
        per_bw_max = p_cfg.get('max', 150)
    else:
        per_bw_min, per_bw_max = 20, 150
    
    # Parse frequency (alternative to period)
    if 'frequency' in cfg:
        f_cfg = cfg['frequency']
        n_freq = f_cfg.get('n_points', 10)
        freq_log_min = f_cfg.get('log_min', -2.2)
        freq_log_max = f_cfg.get('log_max', -1.3)
    else:
        n_freq = cfg.get('n_freq', 10)
        freq_log_min = cfg.get('freq_log_min', -2.2)
        freq_log_max = cfg.get('freq_log_max', -1.3)
    
    # Parse physical parameters
    rho = cfg.get('rho', 3300.0)
    density_model = cfg.get('density_model', 'constant')
    density_file = cfg.get('density_file', None)
    sig_MPa = cfg.get('sig_MPa', 0.1)
    Ch2o = cfg.get('Ch2o', 0.0)
    
    # Parse methods
    anelastic_methods = cfg.get('anelastic_methods', ['eburgers_psp', 'andrade_psp', 'xfit_mxw', 'xfit_premelt'])
    viscous_method = cfg.get('viscous_method', 'HK2003')
    eburgers_method = cfg.get('eburgers_method', 'FastBurger')
    include_direct_melt_effect = cfg.get('include_direct_melt_effect', 0)
    solidus_method = cfg.get('solidus_method', 'hirschmann')
    
    # Parse elastic scaling options
    if 'elastic' in cfg:
        e_cfg = cfg['elastic']
        elastic_method = e_cfg.get('method', 'anharmonic')
        temperature_scaling = e_cfg.get('temperature_scaling', 'isaak')
        pressure_scaling = e_cfg.get('pressure_scaling', 'cammarano')
        reference_scaling = e_cfg.get('reference_scaling', 'default')
    else:
        elastic_method = 'anharmonic'
        temperature_scaling = 'isaak'
        pressure_scaling = 'cammarano'
        reference_scaling = 'default'
    
    # Parse LUT plot options
    plot_cfg = cfg.get('plot_lut', {})
    if isinstance(plot_cfg, bool):
        plot_lut = plot_cfg
        plot_lut_dir = 'lut_plots'
        plot_lut_every_n = 1
    elif isinstance(plot_cfg, dict):
        plot_lut = plot_cfg.get('enabled', False)
        plot_lut_dir = plot_cfg.get('output_dir', 'lut_plots')
        plot_lut_every_n = plot_cfg.get('every_n', 1)
    else:
        plot_lut = False
        plot_lut_dir = 'lut_plots'
        plot_lut_every_n = 1

    # Parse output_dir (used to auto-derive output_file and plot_lut_dir)
    output_dir = cfg.get('output_dir', None)

    return SweepParams(
        T=T,
        phi=phi,
        gs=gs,
        z_min=z_min,
        z_max=z_max,
        n_z=n_z,
        per_bw_min=per_bw_min,
        per_bw_max=per_bw_max,
        n_freq=n_freq,
        freq_log_min=freq_log_min,
        freq_log_max=freq_log_max,
        rho=rho,
        density_model=density_model,
        density_file=density_file,
        sig_MPa=sig_MPa,
        Ch2o=Ch2o,
        anelastic_methods=anelastic_methods,
        viscous_method=viscous_method,
        eburgers_method=eburgers_method,
        include_direct_melt_effect=include_direct_melt_effect,
        gs_type=gs_type,
        output_dir=output_dir,
        output_file=cfg.get('output_file', 'sweep.mat'),
        elastic_method=elastic_method,
        temperature_scaling=temperature_scaling,
        pressure_scaling=pressure_scaling,
        reference_scaling=reference_scaling,
        solidus_method=solidus_method,
        plot_lut=plot_lut,
        plot_lut_dir=plot_lut_dir,
        plot_lut_every_n=plot_lut_every_n,
    )


def main():
    """Command-line interface for sweep generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate VBR parameter sweep for Bayesian inversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sweep with default parameters
  python -m bayesian_fitting_py.vbr.generate_sweep

  # Generate sweep from config file
  python -m bayesian_fitting_py.vbr.generate_sweep --config sweep_config.yaml

  # Generate sweep with custom parameters
  python -m bayesian_fitting_py.vbr.generate_sweep --t-min 1200 --t-max 1500 --z-min 75 --z-max 150
        """
    )
    
    parser.add_argument('--config', '-c', type=str, help='YAML configuration file')
    parser.add_argument('--output', '-o', type=str, default='sweep.mat', help='Output file (default: sweep.mat)')
    
    # Temperature options
    parser.add_argument('--t-min', type=float, default=1100, help='Minimum temperature [°C]')
    parser.add_argument('--t-max', type=float, default=1650, help='Maximum temperature [°C]')
    parser.add_argument('--t-step', type=float, default=25, help='Temperature step [°C]')
    
    # Depth options
    parser.add_argument('--z-min', type=float, default=50, help='Minimum depth [km]')
    parser.add_argument('--z-max', type=float, default=170, help='Maximum depth [km]')
    parser.add_argument('--n-z', type=int, default=100, help='Number of depth points')
    
    # Grain size options
    parser.add_argument('--gs-min', type=float, default=100, help='Minimum grain size [μm]')
    parser.add_argument('--gs-max', type=float, default=50000, help='Maximum grain size [μm]')
    parser.add_argument('--n-gs', type=int, default=20, help='Number of grain size points')
    
    # Method options
    parser.add_argument('--methods', nargs='+', 
                       default=['eburgers_psp', 'andrade_psp', 'xfit_mxw', 'xfit_premelt'],
                       help='Anelastic methods to calculate')
    
    # Density options
    parser.add_argument('--density-model', type=str, default='constant',
                       choices=['constant', 'prem', 'custom'],
                       help="Density model: 'constant' (default), 'prem', or 'custom'")
    parser.add_argument('--density-file', type=str, default=None,
                       help='Path to custom density CSV (columns: depth_km, density)')
    parser.add_argument('--elastic-method', type=str, default='anharmonic',
                       choices=['anharmonic', 'cammarano2003'],
                       help="Elastic method: 'anharmonic' (default) or 'cammarano2003'")
    
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress progress output')
    
    args = parser.parse_args()
    
    if args.config:
        params = load_sweep_params_from_yaml(args.config)
    else:
        params = SweepParams(
            T=np.arange(args.t_min, args.t_max + args.t_step, args.t_step),
            gs=np.logspace(np.log10(args.gs_min), np.log10(args.gs_max), args.n_gs),
            z_min=args.z_min,
            z_max=args.z_max,
            n_z=args.n_z,
            anelastic_methods=args.methods,
            elastic_method=args.elastic_method,
            density_model=args.density_model,
            density_file=args.density_file,
            output_file=args.output,
        )
    
    sweep = generate_parameter_sweep(params, verbose=not args.quiet)
    save_sweep(sweep, params.output_file, verbose=not args.quiet)

    # Generate LUT plots if requested
    if params.plot_lut:
        from .plot_lut import generate_sweep_lut_plots
        if not args.quiet:
            print(f"\nGenerating LUT plots (every {params.plot_lut_every_n} depths)...")
        generate_sweep_lut_plots(
            sweep, params.plot_lut_dir,
            every_n=params.plot_lut_every_n,
            verbose=not args.quiet,
        )


if __name__ == '__main__':
    main()
