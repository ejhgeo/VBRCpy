"""
Parameter classes for VBR calculations.

Provides default parameters for elastic, anelastic, and viscous methods.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# Celsius to Kelvin conversion constant
C2K = 273


@dataclass
class Params_Global:
    """Global parameters affecting all VBR calculations."""
    # Melt effects - small melt enhancement
    melt_enhancement: int = 0  # 0 = off, 1 = on
    # Critical melt fraction [diff, disl., gbs] - matches MATLAB Params_Global.m
    phi_c: List[float] = field(default_factory=lambda: [1e-5, 1e-5, 1e-5])
    # Melt enhancement factor [diff, disl., gbs] - matches MATLAB Params_Global.m  
    x_phi_c: List[float] = field(default_factory=lambda: [5.0, 1.0, 2.5])


def get_global_melt_effects(global_params: Optional[Params_Global] = None):
    """Get melt effect parameters from global settings.
    
    Following MATLAB setGlobalMeltEffects.m:
    - phi_c always uses default values (1e-5)
    - x_phi_c is set to [1,1,1] when melt_enhancement=0, else uses global values
    
    This means even with melt_enhancement=0, the slope term (alpha*phi) still applies.
    """
    if global_params is None:
        global_params = Params_Global()
    
    # phi_c always uses defaults (like MATLAB)
    phi_c = global_params.phi_c.copy() if hasattr(global_params.phi_c, 'copy') else list(global_params.phi_c)
    
    if global_params.melt_enhancement == 0:
        # When melt_enhancement is off, x_phi_c = [1,1,1] (log(1)=0, so step term vanishes)
        # But slope term (alpha * phi) still applies!
        x_phi_c = [1.0, 1.0, 1.0]
    else:
        x_phi_c = global_params.x_phi_c.copy() if hasattr(global_params.x_phi_c, 'copy') else list(global_params.x_phi_c)
    
    return phi_c, x_phi_c


@dataclass
class AnharmonicParams:
    """Parameters for anharmonic elastic modulus calculation."""
    T_K_ref: float = 300.0  # room temp [K]
    P_Pa_ref: float = 1e5  # 1 atm [Pa]
    Gu_0_ol: float = 81.0  # olivine reference shear modulus [GPa]
    Ku_0_ol: float = 129.0  # olivine reference bulk modulus [GPa]
    
    # Temperature scaling (isaak)
    dG_dT: float = -1.36e7  # Pa/K
    dK_dT: float = -1.8e7  # Pa/K
    
    # Pressure scaling (cammarano)
    dG_dP: float = 1.4  # unitless
    dG_dP2: float = 0.0  # 1/Pa
    dK_dP: float = 4.2  # unitless
    dK_dP2: float = 0.0  # 1/Pa
    
    # Chi mixing for crust
    chi_mixing: int = 1


@dataclass
class AhnPoroParams:
    """Parameters for poro-elastic melt effect."""
    Melt_A: float = 1.6  # depends on wetting angle
    Melt_Km: float = 30e9  # melt bulk modulus [Pa]
    Melt_nu: float = 0.25  # Poisson's ratio for melt


def Params_Elastic(method: str = '') -> Dict[str, Any]:
    """
    Get parameters for elastic methods.
    
    Parameters
    ----------
    method : str
        Method name: 'anharmonic', 'anh_poro', 'SLB2005', or '' for info
        
    Returns
    -------
    dict
        Parameter dictionary for the method
    """
    params = {'possible_methods': ['anharmonic', 'anh_poro', 'cammarano2003', 'SLB2005']}
    
    if method == 'anharmonic':
        params['func_name'] = 'el_anharmonic'
        params['T_K_ref'] = 300.0  # room temp [K]
        params['P_Pa_ref'] = 1e5  # 1 atm [Pa]
        # VBRc default: G0 = 81 GPa at STP (Havlin 2021 paper says 80, but sweep uses 81)
        params['Gu_0_ol'] = 81.0  # olivine reference shear modulus [GPa]
        params['Ku_0_ol'] = 129.0  # olivine reference bulk modulus [GPa]
        
        # Temperature scaling options
        # Default to 'isaak' for both T and P to match Havlin et al. 2021
        params['temperature_scaling'] = 'isaak'
        params['pressure_scaling'] = 'cammarano'
        params['reference_scaling'] = 'default'
        
        # Isaak 1992 derivatives + Havlin et al. 2021 (quoting Isaak 1992) pressure scaling
        # From Havlin et al., 2021 paper: dG/dT = -13.6 MPa/K, dG/dP = 1.8, G0 = 80 GPa
        # Isaak 1992 only provides temperature derivatives, so user must use Cammarano 2003 or Abramson for pressure derivatives
        params['isaak'] = {
            'dG_dT': -1.36e7,  # Pa/K (13.6 MPa/K from Isaak 1992)
            'dK_dT': -1.8e7,   # Pa/K
            # 'dG_dP': 1.8,      # unitless (from Havlin et al. 2021)
            # 'dG_dP2': 0.0,     # 1/Pa
            # 'dK_dP': 4.2,      # unitless
            # 'dK_dP2': 0.0,     # 1/Pa
        }
        
        # Cammarano 2003 derivatives
        params['cammarano'] = {
            'dG_dT': -0.014e9,  # Pa/K
            'dG_dP': 1.4,       # unitless
            'dG_dP2': 0.0,      # 1/Pa
            'dK_dT': -0.017e9,  # Pa/K
            'dK_dP': 4.2,       # unitless
            'dK_dP2': 0.0,      # 1/Pa
        }

        # Abramson 1997: only provides pressure derivatives, so user must use Cammarano or Isaak for temperature derivatives
        params['abramson'] = {
            'dG_dP': 1.71,   # shear modulus pressure dependence (Pa/Pa)
            'dG_dP2': -0.027 / 1e9, # 1/Pa
            'dK_dP': 4.2, # bulk modulus pressure dependence (Pa/Pa)
            'dK_dP2': 0.0
        }
        
        # Upper mantle reference values (Abers and Hacker 2016, pyrolitic upper mantle composition)
        # Provides complete set: reference values, temperature derivatives, and pressure derivatives
        params['upper_mantle'] = {
            'T_K_ref': 1300 + C2K,   # K
            'P_Pa_ref': 3.0e9,       # 3 GPa reference pressure
            'Gu_0': 66.410,          # Reference shear modulus [GPa]
            'Ku_0': 119.88,          # Reference bulk modulus [GPa]
            'dG_dT': -1.3816e7,      # Pa/K
            'dG_dP': 1.5734,         # unitless
            'dG_dP2': 0.0,           # 1/Pa
            'dK_dT': -1.9845e7,      # Pa/K
            'dK_dP': 4.4510,         # unitless
            'dK_dP2': 0.0,           # 1/Pa
            'rho_ref': 3323.1,       # kg/m^3
        }
        
        # Available scaling options
        params['available_reference_scaling'] = ['default', 'upper_mantle']
        params['available_temperature_scaling'] = ['isaak', 'cammarano', 'upper_mantle']
        params['available_pressure_scaling'] = ['cammarano', 'abramson', 'upper_mantle']
        
        # Crustal values
        params['chi_mixing'] = 1
        params['crust'] = {
            'Gu_0': 40.0,  # GPa
            'Ku_0': 86.0,  # GPa
            'dG_dT': -3.6e6,
            'dG_dP': 0.011474,
            'dG_dP2': 0.0,
            'dK_dT': -4.7e6,
            'dK_dP': 0.013032,
            'dK_dP2': 0.0,
        }
        
    elif method == 'anh_poro':
        params['func_name'] = 'el_ModUnrlx_MELT_f'  # function for the poroelastic correction
        params['Melt_A'] = 1.6    # 1:2.3 depending upon the wetting angle (see Yoshino)
        params['Melt_Km'] = 30e9  # melt bulk modulus [Pa], Takei 2002, Table 2
        params['Melt_nu'] = 0.25  # Poisson's ratio for melt

    elif method == 'SLB2005':   # Stixrude and Lithgow-Bertelloni 2005 fit of upper mantle Vs
        params['func_name'] = 'slb2005' # el_Vs_SnLG_f in Matlab

    elif method == 'cammarano2003':
        # Cammarano et al. (2003) finite-strain mineral physics model.
        # Uses depth/pressure-dependent mineral assemblage with 3rd-order
        # Birch-Murnaghan EOS and Voigt-Reuss-Hill averaging.
        params['func_name'] = 'el_cammarano2003'
        params['X_Fe'] = 0.1          # Iron mole fraction (Mg# = 0.9)
        params['composition'] = 'pyrolite'  # 'pyrolite' or 'custom'
        
        # Pressure boundaries for assemblage switching (GPa)
        # Approx: 14 GPa ~ 410 km, 18 GPa ~ 520 km, 23 GPa ~ 660 km
        params['P_boundary_410'] = 14.0   # olivine → wadsleyite
        params['P_boundary_520'] = 18.0   # wadsleyite → ringwoodite
        params['P_boundary_660'] = 23.0   # ringwoodite → perovskite+mw
        
        params['note'] = ('Uses Table A.1 mineral parameters and Appendix A '
                          'finite-strain method from Cammarano et al. (2003) PEPI.')
        
    return params


def Params_Anelastic(method: str = '', global_params: Optional[Params_Global] = None) -> Dict[str, Any]:
    """
    Get parameters for anelastic methods.
    
    Parameters
    ----------
    method : str
        Method name: 'eburgers_psp', 'andrade_psp', 'xfit_mxw', 'xfit_premelt', or '' for info
    global_params : Params_Global, optional
        Global parameters for melt effects
        
    Returns
    -------
    dict
        Parameter dictionary for the method
    """
    params = {
        'possible_methods': [
            'eburgers_psp', 'andrade_psp', 'xfit_mxw', 'xfit_premelt',
            'andrade_analytical', 'maxwell_analytical'
        ]
    }
    
    phi_c, x_phi_c = get_global_melt_effects(global_params)
    
    if method == 'eburgers_psp':
        params['func_name'] = 'Q_eBurgers_decider'
        params['method'] = 'FastBurger'      # 'PointWise' or 'FastBurger' (MATLAB default is FastBurger)
        params['nTauGlob'] = 3000           # Points for global Tau discretization 
        params['R'] = 8.314                 # gas constant
        params['eBurgerFit'] = 'bg_only'    # or 'bg_peak'
        params['useJF10visc'] = 1           # if 1, will use the scaling from JF10 for maxwell time. If 0, will calculate
        params['integration_method'] = 0    # 0 for trapezoidal; 1 for quadrature
        params['tau_integration_points'] = 500  # number of points for integration of high-T background if trapezoidal
        
        # Load JF10 parameters
        params = _load_JF10_eBurger_params(params)
        
    elif method == 'andrade_psp':
        params['func_name'] = 'Q_Andrade_PseudoP_f'
        params['n'] = 0.33
        params['Beta'] = 0.020
        params['Tau_MR'] = 10**5.3
        params['E'] = 303e3  # J/mol
        params['G_UR'] = 62.2  # GPa
        params['TR'] = 900 + C2K  # K
        params['PR'] = 0.2  # GPa
        params['dR'] = 3.1  # microns
        params['R'] = 8.314
        params['Vstar'] = 10e-6  # m^3/mol
        params['m'] = 1
        
    elif method == 'xfit_mxw':
        params['func_name'] = 'Q_xfit_mxw'
        params['fit'] = 'fit1'
        params['beta2'] = 1853.0
        params['alpha2'] = 0.5
        params['tau_cutoff'] = 1e-11
        params['beta1'] = 0.32
        params['Alpha_a'] = 0.39
        params['Alpha_b'] = 0.28
        params['Alpha_c'] = 2.6
        params['Alpha_taun'] = 0.1
        
    elif method == 'xfit_premelt':
        params['func_name'] = 'Q_xfit_premelt'
        params['solidus_method'] = 'hirschmann'  # solidus parameterization
        params['density_model'] = 'constant'     # for YK2001 P-to-depth conversion
        params['density_rho'] = 3400.0            # constant density (kg/m^3)
        params['density_file'] = None             # custom density CSV path
        # High temp background spectrum
        params['alpha_B'] = 0.38    # high temp background exponent
        params['A_B'] = 0.664       # high temp background dissipation strength
        
        # Pre-melting dissipation peak
        params['tau_pp'] = 6e-5     # peak center, table 4 of YT16
        params['Ap_fac_1'] = 0.01
        params['Ap_fac_2'] = 0.4
        params['Ap_fac_3'] = 0.03
        params['sig_p_fac_1'] = 4.0
        params['sig_p_fac_2'] = 37.5
        params['sig_p_fac_3'] = 7.0
        params['Ap_Tn_pts'] = [0.91, 0.96, 1.0] # Tn cutoff points
        params['sig_p_Tn_pts'] = [0.92, 1.0]    # Tn cutoff points
        
        # Melt effects (YT2024)
        # The following beta values are set to 0.0 within Q_xfit_premelt
        # If include_direct_melt_effect = 0, corresponding to YT2016. 
        # If include_direct_melt_effect = 1, the scaling will follow YT2024. 
        # Additionally, include_direct_melt_effect = 1 will trigger different poro-elastic behavior.
        params['include_direct_melt_effect'] = 0
        params['Beta'] = 1.38
        params['Beta_B'] = 6.94
        params['poro_Lambda'] = 4.0
        
    # Set steady-state melt dependence for diff. creep (i.e., exp(-alpha * phi))
    # Same for ALL anelastic methods, following MATLAB Params_Anelastic.m lines 186-196
    HK2003 = Params_Viscous('HK2003', global_params)
    params['melt_alpha'] = HK2003['diff']['alf']  # HK2003.diff.alf = 25
    
    # Pull in the small melt effect parameter values -- use diffusion creep value
    params['phi_c'] = phi_c[0]
    params['x_phi_c'] = x_phi_c[0]
    
    return params


def _load_JF10_eBurger_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Load Jackson & Faul 2010 extended Burgers parameters."""
    
    # Multiple sample best high-temp background only fit
    params['bg_only'] = {
        'dR': 13.4,          # ref grain size in microns
        'G_UR': 62.5,        # GPa, unrelaxed G, reference value
        'E': 303000,         # J/mol
        'm_a': 1.19,         # grain size exponent for tau_i, i in (L,H,P)
        'alf': 0.257,        # high temp background tau exponent
        'DeltaB': 1.13,      # relaxation strength
        'Tau_LR': 1e-3,      # relaxation time lower limit reference
        'Tau_HR': 1e7,       # relaxation time upper limit reference
        'Tau_MR': 10**6.95,  # Maxwell relaxation time reference
        'DeltaP': 0.0,       # no peak
        'sig': 0.0,          # no peak, set to 0
        'Tau_PR': 0.0,       # no peak, set to 0
        'TR': 1173,          # ref temp [K]
        'PR': 0.2,           # ref confining pressure [GPa] of experiments
        'Vstar': 10e-6,      # m^3/mol, Activation volume
        'm_v': 3,            # viscous grain size exponent for maxwell time
    }
    
    # Multiple sample best high-temp background + peak fit
    params['bg_peak'] = {
        'DeltaP': 0.057,    # relaxation strength of the peak
        'sig': 4,           # sigma, peak breadth
        'Tau_PR': 10**-3.4, # center maxwell time
        'dR': 13.4,         # ref grain size in microns
        'G_UR': 66.5,       # GPa, unrelaxed G, reference value
        'E': 360000,        # J/mol
        'm_a': 1.31,        # grain size exponent for tau_i, i in (L,H,P)
        'alf': 0.274,       # high temperature background tau exponent
        'DeltaB': 1.13,     # relaxation strength of background
        'Tau_LR': 1e-3,     # relaxation time lower limit reference
        'Tau_HR': 1e7,      # relaxation time upper limit reference
        'Tau_MR': 10**7.48, # reference maxwell relaxation time
        'TR': 1173,         
        'PR': 0.2,
        'Vstar': 10e-6,
        'm_v': 3,
    }
    
    return params


def Params_Viscous(method: str = '', global_params: Optional[Params_Global] = None) -> Dict[str, Any]:
    """
    Get parameters for viscous methods.
    
    Parameters
    ----------
    method : str
        Method name: 'xfit_premelt', 'HK2003', etc., or '' for info
    global_params : Params_Global, optional
        Global parameters for melt effects
        
    Returns
    -------
    dict
        Parameter dictionary for the method
    """
    params = {'possible_methods': ['HK2003', 'HZK2011', 'xfit_premelt']}
    
    phi_c, x_phi_c = get_global_melt_effects(global_params)
    
    if method == 'HK2003':
        params['func_name'] = 'visc_calc_HK2003'
        params['ch2o_o'] = 50 # reference water content [ppm] ("dry" below this value)
        params['P_dep_calc'] = 'yes'    # pressure dependent calculation? yes or no
        params['possible_mechs'] = ['diff', 'disl', 'gbs']
        params['diff'] = {
            # dry
            'A': 1.5e9,       # preexponential for Coble diffusion creep
            'Q': 375e3,       # activation energy (J/mol)
            'V': 10e-6,       # activation volume (m^3/mol)
            'p': 3,           # grain size exponent
            'n': 1,           # stress exponent
            'r': 0,           # water fugacity exponent
            'alf': 25,        # melt factor
            'phi_c': phi_c[0],
            'x_phi_c': x_phi_c[0],
            # wet
            'A_wet': 2.5e7,
            'Q_wet': 375e3,
            'V_wet': 10e-6,
            'p_wet': 3,
            'n_wet': 1,
            'r_wet': 0.7,
            'alf_wet': 25,
            'phi_c_wet': phi_c[0],
            'x_phi_c_wet': x_phi_c[0],
        }
        params['disl'] = {
            # dry
            'A': 1.1e5,       # preexponential
            'Q': 530e3,       # activation energy (J/mol)
            'V': 15e-6,       # activation volume (m^3/mol)
            'n': 3.5,         # stress exponent
            'p': 0,           # grain size exponent
            'alf': 30,        # melt factor
            'r': 0,           # water fugacity exponent
            'phi_c': phi_c[1],
            'x_phi_c': x_phi_c[1],
            # wet
            'A_wet': 1600.0,
            'Q_wet': 520e3,
            'V_wet': 22e-6,
            'n_wet': 3.5,
            'p_wet': 0,
            'alf_wet': 30,
            'r_wet': 1.2,
            'phi_c_wet': phi_c[1],
            'x_phi_c_wet': x_phi_c[1],
        }
        params['gbs'] = {
            'A_lt1250': 6.5e3,  # preexponential for GBS-disl creep (T < 1250C)
            'Q_lt1250': 400e3,  # activation energy for GBS-disl creep (J/mol)
            'V_lt1250': 15e-6,  # activation volume
            'p_lt1250': 2,      # grain size exponent
            'n_lt1250': 3.5,    # stress exponent
            'r_lt1250': 0,      # water fugacity exponent
            'alf_lt1250': 35,   # melt factor
            'phi_c_lt1250': phi_c[2],
            'x_phi_c_lt1250': x_phi_c[2],

            'A_gt1250': 4.7e10, # preexponential for GBS-disl creep (T >= 1250C)
            'Q_gt1250': 600e3,  # activation energy for GBS-disl creep (J/mol)
            'V_gt1250': 15e-6,  # activation volume (m^3/mol)
            'p_gt1250': 2,      # grain size exponent
            'n_gt1250': 3.5,    # stress exponent
            'r_gt1250': 0,      # water fugacity exponent
            'alf_gt1250': 35,   # melt factor
            'phi_c_gt1250': phi_c[2],
            'x_phi_c_gt1250': x_phi_c[2],
        }
        params['R'] = 8.314  # gas constant
    
    elif method == 'xfit_premelt':
        params['func_name'] = 'visc_calc_xfit_premelt' # shouldn't this be _visc_xfit_premelt
        params['solidus_method'] = 'hirschmann'  # solidus parameterization
        params['density_model'] = 'constant'     # for YK2001 P-to-depth conversion
        params['density_rho'] = 3400.0            # constant density (kg/m^3)
        params['density_file'] = None             # custom density CSV path
        
        # Near-solidus and melt effects
        params['alpha'] = 30  # lambda in YT2016
        params['T_eta'] = 0.94
        params['gamma'] = 5
        params['B'] = 1.0
        
        # Method for melt-free viscosity
        params['eta_melt_free_method'] = 'xfit_premelt'
        
        # Flow law constants from YT2016
        params['Tr_K'] = 1200 + C2K  # reference temp
        params['Pr_Pa'] = 1.5e9  # reference pressure
        params['eta_r'] = 6.22e21  # reference viscosity
        params['H'] = 462.5e3  # activation energy [J/mol]
        params['V'] = 7.913e-6  # activation volume [m^3/mol]
        params['R'] = 8.314  # gas constant
        params['m'] = 3  # grain size exponent
        params['dg_um_r'] = 0.004 * 1e6  # reference grain size [um] = 4mm
        
    return params
