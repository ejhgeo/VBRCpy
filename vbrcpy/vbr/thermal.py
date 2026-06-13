"""
Thermal properties for VBR calculations.

Includes solidus calculations with volatile dependence.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
from numpy.typing import ArrayLike
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from .params import C2K

R_EARTH_M = 6371e3  # Earth radius in meters

# Resolve the reference-models directory:
# <package_root> = vbrcpy/  (parent of vbr/)
# data lives at  <package_root>/../data/reference_models/
_VBR_DIR = Path(__file__).resolve().parent           # vbrcpy/vbr/
_REF_MODELS_DIR = _VBR_DIR.parent.parent / 'data' / 'reference_models'

# Mapping of built-in Earth model names to filenames
_BUILTIN_MODELS = {
    'prem': 'PREM_for_VBRc.txt',
    'prem_nocrust': 'PREMnoCrust_for_VBRc.txt',
    'stw105': 'STW105_for_VBRc.txt',
    'stw105_nocrust': 'STW105noCrust_for_VBRc.txt',
}

# Built-in geotherm files
_BUILTIN_GEOTHERMS = {
    'sc2006': 'SC2006_geotherm.csv',
}


def _load_earth_model(
    model: str,
    custom_file: Optional[str] = None,
    fields: Optional[list] = None,
) -> tuple:
    """Load a radial Earth model and return arrays for requested fields.

    Handles the common 5-column space-delimited format
    (``radius depth density Vs Qmu``) used by the bundled PREM and
    STW105 files as well as user-supplied custom files in the same format.

    Parameters
    ----------
    model : str
        ``'prem'``, ``'stw105'``, or ``'custom'``.
    custom_file : str, optional
        Path to a user-supplied file.  Required when *model* is ``'custom'``.
    fields : list of str, optional
        Extra column names to return in addition to ``depth`` and ``density``.
        Valid names: ``'Vs'``, ``'Qmu'``.  When *None* (default) only
        ``(depth_m, density)`` is returned for backward compatibility.

    Returns
    -------
    tuple
        ``(depth_m, density)`` when *fields* is None, otherwise
        ``(depth_m, density, field1_array, field2_array, ...)`` with one
        extra array per requested field, all sorted with discontinuities
        preserved by applying a tiny depth offset to duplicates.
    """
    if model in _BUILTIN_MODELS:
        filepath = _REF_MODELS_DIR / _BUILTIN_MODELS[model]
    elif model == 'custom':
        if custom_file is None:
            raise ValueError("density_file is required when density_model='custom'")
        filepath = Path(custom_file)
    else:
        valid = ', '.join([repr(k) for k in _BUILTIN_MODELS] + ["'custom'"])
        raise ValueError(
            f"Unknown density model '{model}'. Use one of: {valid}"
        )

    # Read the 5-column space/whitespace-delimited file
    data = np.genfromtxt(filepath, names=True)
    # Columns: radius, depth, density, Vs, Qmu
    depth_m = data['depth'].astype(float).copy()
    density = data['density']

    # Gather any extra requested columns
    extras_raw = []
    if fields:
        for fname in fields:
            extras_raw.append(data[fname])

    # Preserve discontinuities: shift duplicate depths by a small epsilon
    # *before* sorting so the file's natural ordering is retained.
    # Files are pre-sorted by ascending depth with the shallow-side entry
    # first at discontinuities, so shifting later duplicates by +epsilon
    # places the deep-side values at the correct (slightly deeper) depth.
    depth_eps_m = 100.0  # 0.1 km
    i = 0
    n = len(depth_m)
    while i < n:
        j = i + 1
        while j < n and depth_m[j] == depth_m[i]:
            j += 1
        for k in range(1, j - i):
            depth_m[i + k] = depth_m[i] + k * depth_eps_m
        i = j

    # Sort ascending by the (now-unique) depths
    sort_idx = np.argsort(depth_m)
    depth_m = depth_m[sort_idx]
    density = density[sort_idx]
    extras_raw = [arr[sort_idx] for arr in extras_raw]

    if fields:
        return (depth_m, density, *extras_raw)
    return depth_m, density


def load_q_from_earth_model(
    model: str,
    depth_km: np.ndarray,
    custom_file: Optional[str] = None,
) -> np.ndarray:
    """Return Qmu values interpolated at the requested depths.

    Parameters
    ----------
    model : str
        ``'prem'``, ``'stw105'``, or ``'custom'``.
    depth_km : array_like
        Depths in km at which to evaluate Qmu.
    custom_file : str, optional
        Path to a custom Earth-model file (same 5-column format).

    Returns
    -------
    ndarray
        Qmu values at the requested depths.
    """
    depth_m, _density, qmu = _load_earth_model(
        model, custom_file=custom_file, fields=['Qmu'],
    )
    depth_km_profile = depth_m / 1e3
    return np.interp(np.asarray(depth_km), depth_km_profile, qmu)


def load_vs_from_earth_model(
    model: str,
    depth_km: np.ndarray,
    custom_file: Optional[str] = None,
) -> np.ndarray:
    """Return Vs values interpolated at the requested depths.

    Parameters
    ----------
    model : str
        ``'prem'``, ``'stw105'``, or ``'custom'``.
    depth_km : array_like
        Depths in km at which to evaluate Vs.
    custom_file : str, optional
        Path to a custom Earth-model file (same 5-column format).

    Returns
    -------
    ndarray
        Vs values (km/s) at the requested depths.
    """
    depth_m, _density, vs = _load_earth_model(
        model, custom_file=custom_file, fields=['Vs'],
    )
    depth_km_profile = depth_m / 1e3
    return np.interp(np.asarray(depth_km), depth_km_profile, vs)


def load_geotherm(
    name_or_file: str,
    depth_km: Optional[np.ndarray] = None,
) -> tuple:
    """Load a geotherm profile and optionally interpolate to requested depths.

    Parameters
    ----------
    name_or_file : str
        A built-in geotherm name (``'sc2006'``) or a path to a CSV file
        with columns ``depth_km`` and ``temperature_C``.
    depth_km : array_like, optional
        Depths in km at which to interpolate.  When *None*, returns the
        raw profile from the file.

    Returns
    -------
    tuple of (depth_km_arr, temperature_C_arr)
        Depth (km) and temperature (°C) arrays.
    """
    key = name_or_file.lower().strip()
    if key in _BUILTIN_GEOTHERMS:
        filepath = _REF_MODELS_DIR / _BUILTIN_GEOTHERMS[key]
    else:
        filepath = Path(name_or_file)

    if not filepath.exists():
        valid = ', '.join([repr(k) for k in _BUILTIN_GEOTHERMS])
        raise FileNotFoundError(
            f"Geotherm file not found: {filepath}.  "
            f"Built-in options: {valid}"
        )

    data = np.genfromtxt(filepath, delimiter=',', names=True)
    z_prof = data['depth_km']
    t_prof = data['temperature_C']

    if depth_km is not None:
        depth_km = np.asarray(depth_km)
        t_interp = np.interp(depth_km, z_prof, t_prof)
        return depth_km, t_interp

    return z_prof, t_prof


def solidus(
    P_Pa: ArrayLike,
    H2O: ArrayLike = 0.0,
    CO2: ArrayLike = 0.0,
    method: str = 'hirschmann',
    depth_km: Optional[ArrayLike] = None,
    density_model: str = 'constant',
    density_rho: float = 3400.0,
    density_file: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Calculate peridotite solidus with volatile dependence.
    
    Water depression calculated with Katz et al (2003), CO2 depression 
    calculated with Dasgupta et al (2007). The dry solidus can be calculated
    using either Katz et al (2003) or Hirschmann (2000).
    
    Parameters
    ----------
    P_Pa : array_like
        Pressure in Pa
    H2O : array_like
        Weight percent of water in the melt phase
    CO2 : array_like
        Weight percent of CO2 in the melt phase
    method : str
        Which dry solidus to use: 'katz', 'hirschmann', or 'yk2001'
    depth_km : array_like, optional
        Depth in km. Only used by 'yk2001'. When provided, bypasses
        pressure-to-depth conversion for self-consistency.
    density_model : str
        Density model for P-to-depth conversion ('constant', 'prem',
        'stw105', or 'custom'). Only used by 'yk2001' when *depth_km* is None.
    density_rho : float
        Constant density in kg/m³. Only used when density_model='constant'.
    density_file : str, optional
        Path to custom density file. Required when density_model='custom'.
        
    Returns
    -------
    dict
        Solidus structure with fields:
        - Tsol: effective solidus [C]
        - Tsol_dry: volatile-free solidus [C]
        
    References
    ----------
    Dasgupta, R., Hirschmann, M. M., & Smith, N. D. (2007). Water follows carbon:
        CO2 incites deep silicate melting and dehydration beneath mid-ocean ridges.
        Geology, 35(2), 135-138.
    Hirschmann, M. M. (2000). Mantle solidus: Experimental constraints and
        the effects of peridotite composition. Geochemistry, Geophysics, Geosystems.
    Katz, R. F., Spiegelman, M., & Langmuir, C. H. (2003). A new parameterization
        of hydrous mantle melting. Geochemistry, Geophysics, Geosystems, 4(9).
    """
    P_Pa = np.atleast_1d(P_Pa).astype(float)
    H2O = np.atleast_1d(H2O).astype(float)
    CO2 = np.atleast_1d(CO2).astype(float)
    
    # Broadcast to same shape
    P_GPa = P_Pa * 1e-9
    
    # Calculate volatile depressions
    dT_H2O, _ = _depression_katz(H2O, P_GPa)
    dT_CO2 = _depression_dasgupta(CO2)
    
    result = {}
    
    if method.lower() == 'katz':
        sol = _solidus_katz(P_GPa)
        result['Tsol_dry'] = sol['Tsol_dry']
        result['Tsol'] = sol['Tsol_dry'] - dT_H2O - dT_CO2
        result['Tliq'] = sol['Tliq_dry'] - dT_H2O - dT_CO2
        result['Tlherz'] = sol['Tlherz_dry'] - dT_H2O - dT_CO2
    elif method.lower() == 'hirschmann':
        sol = _solidus_hirschmann(P_GPa)
        result['Tsol_dry'] = sol['Tsol_dry']
        result['Tsol'] = sol['Tsol_dry'] - dT_H2O - dT_CO2
    elif method.lower() == 'yk2001':
        _depth_km = np.atleast_1d(depth_km).astype(float) if depth_km is not None else None
        sol = _solidus_yk2001(
            P_GPa, depth_km=_depth_km,
            density_model=density_model, density_rho=density_rho,
            density_file=density_file,
        )
        result['Tsol_dry'] = sol['Tsol_dry']
        result['Tsol'] = sol['Tsol_dry'] - dT_H2O - dT_CO2
    else:
        raise ValueError(f"Unknown solidus method: {method}")
    
    return result


def _solidus_katz(P_GPa: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Dry peridotite solidus from Katz et al (2003).
    
    Parameters
    ----------
    P_GPa : array
        Pressure in GPa
        
    Returns
    -------
    dict
        Solidus temperatures in C
    """
    # Solidus parameters
    A1 = 1085.7  # C
    A2 = 132.9   # C/GPa
    A3 = -5.1    # C/GPa^2
    
    # Lherzolite liquidus
    B1 = 1475  # C
    B2 = 80    # C/GPa
    B3 = -3.2  # C/GPa^2
    
    # True liquidus
    C1 = 1780  # C
    C2 = 45    # C/GPa
    C3 = -2    # C/GPa^2
    
    Tsol_dry = A1 + A2 * P_GPa + A3 * P_GPa**2      # solidus
    Tlherz_dry = B1 + B2 * P_GPa + B3 * P_GPa**2    #lherzolite liquidus
    Tliq_dry = C1 + C2 * P_GPa + C3 * P_GPa**2      # true liquidus
    
    # Warn if extrapolating beyond calibration range
    P_max_sol = -A2 / (2 * A3)  # ~13.0 GPa
    if np.any(P_GPa > P_max_sol):
        print(f"Warning: Katz solidus extrapolated beyond quadratic maximum "
              f"(P > {P_max_sol:.1f} GPa). Solidus may decrease unphysically.")
    
    return {
        'Tsol_dry': Tsol_dry,
        'Tlherz_dry': Tlherz_dry,
        'Tliq_dry': Tliq_dry,
    }


def _solidus_hirschmann(P_GPa: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Dry peridotite solidus from Hirschmann (2000).
    
    Parameters
    ----------
    P_GPa : array
        Pressure in GPa
        
    Returns
    -------
    dict
        Solidus temperature in C
    """
    # Equation 2 in Hirschmann 2000
    A1 = 1108.08 #1120.661  # C
    A2 = 139.44 #132.899   # C/GPa
    A3 = -5.904 #-5.104    # C/GPa^2
    
    Tsol_dry = A1 + A2 * P_GPa + A3 * P_GPa**2
    
    # Warn if extrapolating beyond calibration range
    P_max = -A2 / (2 * A3)  # ~11.8 GPa
    if np.any(P_GPa > P_max):
        print(f"Warning: Hirschmann solidus extrapolated beyond quadratic maximum "
              f"(P > {P_max:.1f} GPa). Solidus may decrease unphysically.")
    
    return {'Tsol_dry': Tsol_dry}


def _pressure_to_depth_km(
    P_GPa: np.ndarray,
    density_model: str = 'constant',
    density_rho: float = 3400.0,
    density_file: Optional[str] = None,
) -> np.ndarray:
    """
    Convert pressure (GPa) to depth (km) using a specified density model.

    Parameters
    ----------
    P_GPa : array
        Pressure in GPa.
    density_model : str
        ``'constant'`` for a uniform density, or ``'prem'`` / ``'stw105'`` /
        ``'custom'`` for a depth-dependent radial Earth model.
    density_rho : float
        Density in kg/m³ (used only when *density_model* is ``'constant'``).
    density_file : str, optional
        Path to a custom density file.  Required when *density_model* is
        ``'custom'``.  Must use the same 5-column whitespace-delimited
        format as the bundled models (``radius depth density Vs Qmu``).

    Returns
    -------
    array
        Depth in km.
    """
    g = 9.81  # m/s²

    if density_model == 'constant':
        return P_GPa * 1e9 / (density_rho * g) / 1e3

    # Build a forward z → P(z) profile from the density model, then invert
    depth_m, density = _load_earth_model(density_model, custom_file=density_file)

    # Lithostatic pressure: P(z) = g * ∫₀ᶻ ρ(z') dz'
    P_Pa = np.zeros(len(depth_m))
    P_Pa[1:] = cumulative_trapezoid(density * g, depth_m)
    P_GPa_profile = P_Pa / 1e9

    # Invert: interpolate depth as a function of pressure
    return np.interp(P_GPa, P_GPa_profile, depth_m / 1e3)


def _solidus_yk2001(
    P_GPa: np.ndarray,
    depth_km: Optional[np.ndarray] = None,
    density_model: str = 'constant',
    density_rho: float = 3400.0,
    density_file: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Dry peridotite solidus from Yamazaki and Karato (2001).

    Piecewise quadratic parameterization in terms of depth (km).
    Upper mantle (depth < 660 km):
        Ts = 2100 + 1.4848 * depth - 5e-4 * depth^2   [K]
    Lower mantle (depth >= 660 km):
        Ts = 2916 + 1.25 * depth - 1.65e-4 * depth^2   [K]

    Parameters
    ----------
    P_GPa : array
        Pressure in GPa.
    depth_km : array, optional
        Depth in km.  When provided, used directly (avoids P-to-depth
        conversion).  When *None*, depth is computed from *P_GPa* using the
        specified density model.
    density_model : str
        ``'constant'``, ``'prem'``, ``'stw105'``, or ``'custom'``.  Only used if
        *depth_km* is None.
    density_rho : float
        Constant density in kg/m³ (only used when density_model='constant').
    density_file : str, optional
        Path to custom density file (only used when density_model='custom').

    Returns
    -------
    dict
        Solidus temperature in C (for consistency with other methods).
    """
    if depth_km is None:
        depth_km = _pressure_to_depth_km(
            P_GPa, density_model, density_rho, density_file
        )
    depth_km = np.atleast_1d(depth_km)

    # Piecewise solidus in Kelvin
    Tsol_K = np.where(
        depth_km < 660,
        2100.0 + 1.4848 * depth_km - 5e-4 * depth_km**2,
        2916.0 + 1.25 * depth_km - 1.65e-4 * depth_km**2,
    )

    # Convert K -> C for consistency with other solidus functions
    Tsol_dry = Tsol_K - C2K

    return {'Tsol_dry': Tsol_dry}


def _depression_katz(H2O: np.ndarray, P_GPa: np.ndarray) -> tuple:
    """
    Water-induced solidus depression from Katz et al (2003).
    
    Depression of peridotite solidus due to water from Katz et al,
    "A new parametrization of hydrous mantle melting", G3, 2003,
    DOI: 10.1029/2002GC000433
    
    Parameters
    ----------
    H2O : array
        Water content in wt% IN MELT (not bulk)
    P_GPa : array
        Pressure in GPa
        
    Returns
    -------
    tuple
        (dT, dT_dH2O) depression in C and derivative [C/wt%]
    """
    # Constants from Katz 2003
    gamma = 0.75  # temperature depression exponent
    K = 43.0      # [C/wt%^gamma]
    
    # H2O Saturation
    H2Osat = 12.0 * P_GPa**0.6 + P_GPa
    H2O_capped = np.where(H2O <= H2Osat, H2O, H2Osat)
    
    # Calculate freezing point depression
    dT = K * H2O_capped**gamma
    
    # Derivative (handle H2O=0 case to avoid divide by zero)
    H2O_safe = np.where(H2O_capped > 0, H2O_capped, 1.0)
    dT_dH2O = np.where(H2O_capped > 0, gamma * K * H2O_safe**(gamma - 1), 0.0)
    
    return dT, dT_dH2O


def _depression_dasgupta(CO2: np.ndarray) -> np.ndarray:
    """
    CO2-induced solidus depression from Dasgupta et al (2007).
    
    Water follows carbon: CO2 incites deep silicate melting and dehydration
    beneath mid-ocean ridges, Geology, 2007
    DOI: 10.1130/G22856A
    
    Parameters
    ----------
    CO2 : array
        CO2 content in wt% IN MELT
        
    Returns
    -------
    array
        Temperature depression in C (dTz)
    """
    CO2 = np.atleast_1d(CO2).astype(float)
    dTz = np.zeros_like(CO2)
    
    for iz in range(len(CO2.flat)):
        co2_val = CO2.flat[iz]
        if co2_val <= 25:
            dT = 27.04 * co2_val + 1490.75 * np.log((100 - 1.18 * co2_val) / 100)
        elif co2_val > 25 and co2_val <= 37:
            dTmax = 27.04 * 25 + 1490.75 * np.log((100 - 1.18 * 25) / 100)
            dT = dTmax + (160 - dTmax) / (37 - 25) * (co2_val - 25)
        else:  # co2_val > 37
            dTmax = 27.04 * 25 + 1490.75 * np.log((100 - 1.18 * 25) / 100)
            dTmax = dTmax + (160 - dTmax)
            dT = dTmax + 150
        dTz.flat[iz] = dT
    
    return dTz


def calculate_solidus_K(
    P_GPa: ArrayLike,
    method: str = 'hirschmann',
    depth_km: Optional[ArrayLike] = None,
    density_model: str = 'constant',
    density_rho: float = 3400.0,
    density_file: Optional[str] = None,
) -> np.ndarray:
    """
    Convenience function to calculate solidus in Kelvin.
    
    Parameters
    ----------
    P_GPa : array_like
        Pressure in GPa
    method : str
        Solidus parameterization ('hirschmann', 'katz', or 'yk2001')
    depth_km : array_like, optional
        Depth in km. Only used by 'yk2001'; bypasses P-to-depth conversion.
    density_model : str
        Density model for P-to-depth conversion ('constant', 'prem', or
        'custom'). Only used by 'yk2001' when *depth_km* is None.
    density_rho : float
        Constant density in kg/m³ (when density_model='constant').
    density_file : str, optional
        Path to custom density CSV (when density_model='custom').
        
    Returns
    -------
    array
        Solidus temperature in Kelvin
    """
    P_Pa = np.atleast_1d(P_GPa) * 1e9
    result = solidus(
        P_Pa, H2O=0.0, CO2=0.0, method=method,
        depth_km=depth_km, density_model=density_model,
        density_rho=density_rho, density_file=density_file,
    )
    return result['Tsol'] + C2K
