"""
Thermal properties for VBR calculations.

Includes solidus calculations with volatile dependence.
"""

import numpy as np
from typing import Dict, Any, Union
from numpy.typing import ArrayLike

from .params import C2K


def solidus(
    P_Pa: ArrayLike,
    H2O: ArrayLike = 0.0,
    CO2: ArrayLike = 0.0,
    method: str = 'hirschmann'
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
        Which dry solidus to use: 'katz' or 'hirschmann'
        
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


def calculate_solidus_K(P_GPa: ArrayLike, method: str = 'hirschmann') -> np.ndarray:
    """
    Convenience function to calculate solidus in Kelvin.
    
    Parameters
    ----------
    P_GPa : array_like
        Pressure in GPa
    method : str
        Solidus parameterization ('hirschmann' or 'katz')
        
    Returns
    -------
    array
        Solidus temperature in Kelvin
    """
    P_Pa = np.atleast_1d(P_GPa) * 1e9
    result = solidus(P_Pa, H2O=0.0, CO2=0.0, method=method)
    return result['Tsol'] + C2K
