"""
Cammarano et al. (2003) finite-strain mineral physics model.

Computes anharmonic elastic moduli (G, K_S) and velocities (Vs, Vp) at
arbitrary (T, P) conditions using third-order Birch-Murnaghan finite-strain
theory with mineral parameters from Table A.1 of:

    Cammarano, F., Goes, S., Vacher, P., & Giardini, D. (2003).
    Inferring upper-mantle temperatures from seismic velocities.
    Physics of the Earth and Planetary Interiors, 138(3-4), 197-222.

Implements:
  - Temperature extrapolation of reference properties (Appendix A)
  - 3rd-order finite-strain pressure extrapolation
  - Voigt-Reuss-Hill (VRH) averaging for composite mineralogy
  - Automatic depth/pressure-dependent mineral assemblage switching

Usage
-----
From the sweep generator or VBR core:

    >>> from bayesian_fitting_py.vbr.cammarano import cammarano_elastic
    >>> G, K, rho, Vs, Vp = cammarano_elastic(T_K, P_GPa)

The default composition model is pyrolite, which automatically selects
the appropriate mineralogy for upper mantle, transition zone, or lower
mantle based on pressure.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# ============================================================================
# Reference conditions (ambient)
# ============================================================================
T_REF_K = 300.0   # Reference temperature [K]
P_REF_GPA = 1e-4  # Reference pressure [GPa] (~1 atm)

# ============================================================================
# Mineral database from Table A.1 of Cammarano et al. (2003)
# ============================================================================
# All moduli in GPa, temperature derivatives in GPa/K, density in g/cm³.
# Iron and composition dependencies are encoded as base + coefficient * X.
# Thermal expansion (alpha_0) values are representative constants (K⁻¹);
# the paper references Saxena & Shen (1992) polynomials.

MINERAL_DB = {
    'olivine': {
        'rho_params': {'base': 3.222, 'Fe': 1.182},
        'K_S_params': {'base': 129.0, 'Fe': 0.0},
        'G_params':   {'base': 81.0,  'Fe': -31.0},
        'K_prime_params': {'base': 4.2},
        'G_prime_params': {'base': 1.4},
        'dK_dT': -0.017,   # GPa/K
        'dG_dT': -0.014,   # GPa/K
        'alpha_0': 2.83e-5, # K⁻¹
    },
    'wadsleyite': {
        'rho_params': {'base': 3.472, 'Fe': 1.24},
        'K_S_params': {'base': 172.0, 'Fe': 0.0},
        'G_params':   {'base': 112.0, 'Fe': -40.0},
        'K_prime_params': {'base': 4.5},
        'G_prime_params': {'base': 1.5},
        'dK_dT': -0.014,
        'dG_dT': -0.014,
        'alpha_0': 2.0e-5,
    },
    'ringwoodite': {
        'rho_params': {'base': 3.548, 'Fe': 1.30},
        'K_S_params': {'base': 185.0, 'Fe': 35.0},
        'G_params':   {'base': 120.4, 'Fe': -28.0},
        'K_prime_params': {'base': 4.1},
        'G_prime_params': {'base': 1.3},
        'dK_dT': -0.024,
        'dG_dT': -0.015,
        'alpha_0': 1.9e-5,
    },
    'clinopyroxene': {
        'rho_params': {'base': 3.277, 'Fe': 0.38},
        'K_S_params': {'base': 105.0, 'Fe': 12.0},
        'G_params':   {'base': 67.0,  'Fe': -6.0},
        'K_prime_params': {'base': 6.2, 'Fe': -1.9},
        'G_prime_params': {'base': 1.7},
        'dK_dT': -0.013,
        'dG_dT': -0.010,
        'alpha_0': 3.0e-5,
    },
    'orthopyroxene': {
        'rho_params': {'base': 3.215, 'Fe': 0.799},
        'K_S_params': {'base': 109.0, 'Fe': 20.0},
        'G_params':   {'base': 75.0,  'Fe': 10.0},
        'K_prime_params': {'base': 7.0},
        'G_prime_params': {'base': 1.6},
        'dK_dT': -0.027,
        'dG_dT': -0.012,
        'alpha_0': 2.7e-5,
    },
    # Py-Mj-Alm garnet solid solution — depends on X_Alm and X_Mj
    'garnet': {
        'rho_params': {'base': 3.565, 'Alm': 0.76, 'Mj': -0.05},
        'K_S_params': {'base': 171.0, 'Alm': 15.0, 'Mj': -5.0},
        'G_params':   {'base': 92.0,  'Alm': 7.0,  'Mj': -5.0},
        'K_prime_params': {'base': 4.4, 'Mj': 1.4},
        'G_prime_params': {'base': 1.4, 'Mj': 0.3},
        'dK_dT': -0.019,
        'dG_dT': -0.010,
        'alpha_0': 2.5e-5,
    },
    'ca_garnet': {
        'rho_params': {'base': 3.597},
        'K_S_params': {'base': 168.0},
        'G_params':   {'base': 107.0},
        'K_prime_params': {'base': 5.2},
        'G_prime_params': {'base': 1.6},
        'dK_dT': -0.016,
        'dG_dT': -0.012,
        'alpha_0': 2.5e-5,
    },
    'mg_perovskite': {
        'rho_params': {'base': 4.107, 'Fe': 1.07},
        'K_S_params': {'base': 263.0, 'Fe': 0.0},
        'G_params':   {'base': 175.0, 'Fe': 0.0},
        'K_prime_params': {'base': 4.0},
        'G_prime_params': {'base': 1.8},
        'dK_dT': -0.017,
        'dG_dT': -0.029,
        'alpha_0': 1.5e-5,
    },
    'ca_perovskite': {
        'rho_params': {'base': 4.210},
        'K_S_params': {'base': 236.0},
        'G_params':   {'base': 165.0},
        'K_prime_params': {'base': 4.4},
        'G_prime_params': {'base': 2.5},
        'dK_dT': -0.022,
        'dG_dT': -0.023,
        'alpha_0': 1.6e-5,
    },
    'mg_wustite': {
        'rho_params': {'base': 3.584, 'Fe': 2.28},
        'K_S_params': {'base': 162.0, 'Fe': 0.0},
        'G_params':   {'base': 130.0, 'Fe': -77.0},
        'K_prime_params': {'base': 4.0},
        'G_prime_params': {'base': 2.35},
        'dK_dT': -0.021,
        'dG_dT': -0.024,
        'alpha_0': 3.1e-5,
    },
}


# ============================================================================
# Mineral property evaluation
# ============================================================================

def _eval_param(params: dict, X_Fe: float = 0.1,
                X_Alm: float = 0.1, X_Mj: float = 0.0) -> float:
    """Evaluate a composition-dependent parameter from its coefficient dict."""
    val = params.get('base', 0.0)
    val += params.get('Fe', 0.0) * X_Fe
    val += params.get('Alm', 0.0) * X_Alm
    val += params.get('Mj', 0.0) * X_Mj
    return val


def get_mineral_properties(name: str, X_Fe: float = 0.1,
                           X_Alm: float = 0.1,
                           X_Mj: float = 0.0) -> Dict[str, float]:
    """
    Evaluate mineral elastic parameters at a given composition.

    Parameters
    ----------
    name : str
        Mineral name (key in MINERAL_DB).
    X_Fe : float
        Iron mole fraction (Mg# = 1 - X_Fe). Default 0.1.
    X_Alm : float
        Almandine fraction in garnet. Default 0.1.
    X_Mj : float
        Majorite fraction in garnet. Default 0.0.

    Returns
    -------
    dict
        Keys: rho (g/cm³), K_S (GPa), G (GPa), K_prime, G_prime,
        dK_dT (GPa/K), dG_dT (GPa/K), alpha (1/K).
    """
    m = MINERAL_DB[name]
    return {
        'rho':     _eval_param(m['rho_params'],    X_Fe, X_Alm, X_Mj),
        'K_S':     _eval_param(m['K_S_params'],    X_Fe, X_Alm, X_Mj),
        'G':       _eval_param(m['G_params'],       X_Fe, X_Alm, X_Mj),
        'K_prime': _eval_param(m['K_prime_params'], X_Fe, X_Alm, X_Mj),
        'G_prime': _eval_param(m['G_prime_params'], X_Fe, X_Alm, X_Mj),
        'dK_dT':   m['dK_dT'],
        'dG_dT':   m['dG_dT'],
        'alpha':   m['alpha_0'],
    }


# ============================================================================
# Pyrolite composition model
# ============================================================================
# Volume fractions for a pyrolite mantle at different depth/pressure ranges.
# Pressure boundaries correspond approximately to the 410, 520, and 660 km
# seismic discontinuities.

def pyrolite_assemblage(
    P_GPa: float,
    X_Fe: float = 0.1,
) -> List[Tuple[Dict[str, float], float]]:
    """
    Return the mineral assemblage (properties + volume fraction) for a
    pyrolite composition at a given pressure.

    Parameters
    ----------
    P_GPa : float
        Pressure in GPa.
    X_Fe : float
        Iron mole fraction (default 0.1, Mg# 90).

    Returns
    -------
    list of (mineral_props_dict, volume_fraction)
    """
    if P_GPa < 14.0:
        # Upper mantle (< ~410 km)
        return [
            (get_mineral_properties('olivine',        X_Fe=X_Fe), 0.60),
            (get_mineral_properties('orthopyroxene',  X_Fe=X_Fe), 0.12),
            (get_mineral_properties('clinopyroxene',  X_Fe=X_Fe), 0.14),
            (get_mineral_properties('garnet', X_Fe=X_Fe, X_Alm=0.10, X_Mj=0.0), 0.14),
        ]
        # return [
        #     (get_mineral_properties('olivine',        X_Fe=X_Fe), 0.56),
        #     (get_mineral_properties('orthopyroxene',  X_Fe=X_Fe), 0.17),
        #     (get_mineral_properties('clinopyroxene',  X_Fe=X_Fe), 0.14),
        #     (get_mineral_properties('garnet', X_Fe=X_Fe, X_Alm=0.10, X_Mj=0.0), 0.13),
        # ]
    elif P_GPa < 18.0:
        # Transition zone – wadsleyite field (~410–520 km)
        return [
            (get_mineral_properties('wadsleyite',     X_Fe=X_Fe), 0.57),
            (get_mineral_properties('garnet', X_Fe=X_Fe, X_Alm=0.10, X_Mj=0.50), 0.28),
            (get_mineral_properties('clinopyroxene',  X_Fe=X_Fe), 0.05),
            (get_mineral_properties('ca_garnet'),                  0.10),
        ]
    elif P_GPa < 23.0:
        # Transition zone – ringwoodite field (~520–660 km)
        return [
            (get_mineral_properties('ringwoodite',    X_Fe=X_Fe), 0.57),
            (get_mineral_properties('garnet', X_Fe=X_Fe, X_Alm=0.10, X_Mj=0.75), 0.28),
            (get_mineral_properties('ca_garnet'),                  0.08),
            (get_mineral_properties('ca_perovskite'),              0.07),
        ]
    else:
        # Lower mantle (> ~660 km)
        return [
            (get_mineral_properties('mg_perovskite',  X_Fe=X_Fe), 0.75),
            (get_mineral_properties('mg_wustite',     X_Fe=X_Fe), 0.18),
            (get_mineral_properties('ca_perovskite'),              0.07),
        ]


# ============================================================================
# Finite-strain solver  (Appendix A of Cammarano et al., 2003)
# ============================================================================

def _mineral_moduli_at_PT(
    props: Dict[str, float],
    T_K: np.ndarray,
    P_GPa: float,
    n_newton: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute elastic moduli of a single mineral at (T, P) using
    finite-strain theory.

    Parameters
    ----------
    props : dict
        Mineral properties from get_mineral_properties().
    T_K : 1-D array
        Temperatures [K].
    P_GPa : float
        Pressure [GPa].
    n_newton : int
        Newton iterations for the Eulerian-strain root solve.

    Returns
    -------
    G_PT : array  — shear modulus [GPa]
    K_PT : array  — adiabatic bulk modulus [GPa]
    rho_PT : array — density [g/cm³]
    """
    # Unpack reference (ambient) properties
    rho0  = props['rho']      # g/cm³
    K0    = props['K_S']      # GPa
    G0    = props['G']        # GPa
    Kp    = props['K_prime']  # dimensionless
    Gp    = props['G_prime']  # dimensionless
    dK_dT = props['dK_dT']   # GPa/K
    dG_dT = props['dG_dT']   # GPa/K
    alpha = props['alpha']    # 1/K

    T_K = np.atleast_1d(T_K).astype(float)
    dT = T_K - T_REF_K

    # ------------------------------------------------------------------
    # Step 1: Extrapolate reference properties to temperature T at P = 0
    # ------------------------------------------------------------------
    K0_T  = K0 + dK_dT * dT                      # GPa
    G0_T  = G0 + dG_dT * dT                      # GPa
    rho0_T = rho0 * np.exp(-alpha * dT)           # g/cm³

    # Pressure derivatives extrapolated via Duffy & Anderson (1989)
    exp_aT = np.exp(alpha * dT)
    Kp_T  = Kp * exp_aT
    Gp_T  = Gp * exp_aT

    # ------------------------------------------------------------------
    # Step 2: Constrain second pressure derivatives (C₃ = 0 truncation)
    # ------------------------------------------------------------------
    # From 27 K_{S,0}[ K_{S,0} K''_S - K'_S(7 - K'_S) + 143/9 ] = 0
    Kpp_T = (Kp_T * (7.0 - Kp_T) - 143.0 / 9.0) / K0_T   # 1/GPa
    Gpp_T = 0.631 * Kpp_T                                    # Stacey (1992)

    # ------------------------------------------------------------------
    # Step 3: Solve for Eulerian strain ε at the target pressure
    #         P = -(1-2ε)^{5/2} (C₁ ε + ½ C₂ ε²)
    #         with C₁ = 3 K₀, C₂ = 9 K₀ (4 - K')
    # ------------------------------------------------------------------
    if P_GPa < 1e-6:
        # Surface / negligible pressure → no strain
        eps = np.zeros_like(T_K)
    else:
        C1 = 3.0 * K0_T
        C2 = 9.0 * K0_T * (4.0 - Kp_T)

        # Initial guess: linear approximation ε ≈ -P / (3 K₀)
        eps = -P_GPa / C1

        for _ in range(n_newton):
            u   = 1.0 - 2.0 * eps
            u52 = np.abs(u) ** 2.5 * np.sign(u)
            u32 = np.abs(u) ** 1.5 * np.sign(u)
            g   = C1 * eps + 0.5 * C2 * eps ** 2

            f_val  = -u52 * g - P_GPa
            f_deriv = u32 * (-C1 + (7.0 * C1 - C2) * eps
                             + 4.5 * C2 * eps ** 2)

            # Guard against zero derivative
            f_deriv = np.where(np.abs(f_deriv) < 1e-30, 1e-30, f_deriv)
            eps = eps - f_val / f_deriv

    # ------------------------------------------------------------------
    # Step 4: Compute moduli at (T, P)
    # ------------------------------------------------------------------
    u   = 1.0 - 2.0 * eps
    u52 = np.abs(u) ** 2.5

    # Density: from eps = 1/2[1 - (rho/rho0)^(2/3)] => rho = rho0 * (1-2*eps)^(3/2)
    rho_PT = rho0_T * np.abs(u) ** 1.5   # g/cm³

    # Coefficients for K_S + 4/3 G
    L1 = K0_T + (4.0 / 3.0) * G0_T
    L2 = 5.0 * L1 - 3.0 * K0_T * (Kp_T + (4.0 / 3.0) * Gp_T)
    L3 = (9.0 * K0_T ** 2 * (Kpp_T + (4.0 / 3.0) * Gpp_T)
           - 3.0 * L2 * (Kp_T - 4.0)
           + 5.0 * L1 * (3.0 * Kp_T - 5.0))

    # Coefficients for G
    M1 = G0_T
    M2 = 5.0 * M1 - 3.0 * K0_T * Gp_T
    M3 = (9.0 * K0_T ** 2 * Gpp_T
           - 3.0 * M2 * (Kp_T - 4.0)
           + 5.0 * M1 * (3.0 * Kp_T - 5.0))

    KG_PT = u52 * (L1 + L2 * eps + 0.5 * L3 * eps ** 2)   # K + 4/3 G
    G_PT  = u52 * (M1 + M2 * eps + 0.5 * M3 * eps ** 2)   # G # should the 0.5 actually be here? Claude says yes, need to check. 

    K_PT  = KG_PT - (4.0 / 3.0) * G_PT

    # Ensure physically meaningful values
    G_PT = np.maximum(G_PT, 0.01)
    K_PT = np.maximum(K_PT, 0.01)

    return G_PT, K_PT, rho_PT


# ============================================================================
# Voigt-Reuss-Hill averaging
# ============================================================================

def _vrh_average(
    G_list: List[np.ndarray],
    K_list: List[np.ndarray],
    rho_list: List[np.ndarray],
    fractions: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Voigt-Reuss-Hill average of multiple mineral phases.

    Parameters
    ----------
    G_list, K_list, rho_list : lists of arrays
        Per-mineral moduli (GPa) and densities (g/cm³).
    fractions : list of float
        Volume fractions (must sum to ~1).

    Returns
    -------
    G_vrh, K_vrh, rho_avg : arrays
    """
    fractions = np.array(fractions)
    fractions = fractions / fractions.sum()   # normalise

    # Density — simple volume-weighted average
    rho_avg = sum(f * rho for f, rho in zip(fractions, rho_list))

    # Voigt (upper bound): M_V = Σ f_i M_i
    G_voigt = sum(f * G for f, G in zip(fractions, G_list))
    K_voigt = sum(f * K for f, K in zip(fractions, K_list))

    # Reuss (lower bound): 1/M_R = Σ f_i / M_i
    G_reuss_inv = sum(f / G for f, G in zip(fractions, G_list))
    K_reuss_inv = sum(f / K for f, K in zip(fractions, K_list))
    G_reuss = 1.0 / G_reuss_inv
    K_reuss = 1.0 / K_reuss_inv

    # Hill average
    G_vrh = 0.5 * (G_voigt + G_reuss)
    K_vrh = 0.5 * (K_voigt + K_reuss)

    return G_vrh, K_vrh, rho_avg


# ============================================================================
# Main entry point
# ============================================================================

def cammarano_elastic(
    T_K: np.ndarray,
    P_GPa: float,
    X_Fe: float = 0.1,
    composition: str = 'pyrolite',
    custom_assemblage: Optional[List[Tuple[Dict, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute anharmonic elastic moduli and velocities at (T, P) using
    the Cammarano et al. (2003) finite-strain approach.

    The mineral assemblage is automatically selected based on pressure
    when ``composition='pyrolite'``.

    Parameters
    ----------
    T_K : array_like
        Temperature(s) in Kelvin.
    P_GPa : float
        Pressure in GPa (scalar — same for all T values at a given depth).
    X_Fe : float
        Iron mole fraction (default 0.1).
    composition : str
        Composition model: ``'pyrolite'`` (default) or ``'custom'``.
    custom_assemblage : list, optional
        For ``composition='custom'``: list of ``(mineral_props, fraction)``
        tuples as returned by ``get_mineral_properties()``.

    Returns
    -------
    G_Pa : array — shear modulus [Pa]
    K_Pa : array — adiabatic bulk modulus [Pa]
    rho : array — density [kg/m³]
    Vs : array — shear-wave velocity [m/s]
    Vp : array — compressional-wave velocity [m/s]
    """
    T_K = np.atleast_1d(T_K).astype(float)

    # Select mineral assemblage
    if composition == 'pyrolite':
        assemblage = pyrolite_assemblage(P_GPa, X_Fe=X_Fe)
    elif composition == 'custom':
        if custom_assemblage is None:
            raise ValueError("custom_assemblage required when composition='custom'")
        assemblage = custom_assemblage
    else:
        raise ValueError(f"Unknown composition '{composition}'. "
                         "Use 'pyrolite' or 'custom'.")

    # Compute moduli for each mineral
    G_list, K_list, rho_list = [], [], []
    fractions = []
    for props, frac in assemblage:
        G_m, K_m, rho_m = _mineral_moduli_at_PT(props, T_K, P_GPa)
        G_list.append(G_m)
        K_list.append(K_m)
        rho_list.append(rho_m)
        fractions.append(frac)

    # VRH average (all still in GPa / g/cm³)
    G_vrh, K_vrh, rho_avg = _vrh_average(G_list, K_list, rho_list, fractions)

    # Convert to SI: GPa → Pa, g/cm³ → kg/m³
    G_Pa   = G_vrh * 1e9
    K_Pa   = K_vrh * 1e9
    rho_SI = rho_avg * 1e3   # kg/m³

    # Velocities
    Vs = np.sqrt(G_Pa / rho_SI)
    Vp = np.sqrt((K_Pa + (4.0 / 3.0) * G_Pa) / rho_SI)

    return G_Pa, K_Pa, rho_SI, Vs, Vp
