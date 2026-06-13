"""
Core VBR calculation engine.

Provides the main VBR class for running elastic, viscous, and anelastic
calculations based on thermodynamic state variables.
"""

import numpy as np
from scipy.special import erf
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

from .params import Params_Elastic, Params_Anelastic, Params_Viscous, Params_Global
from .thermal import calculate_solidus_K
from .cammarano import cammarano_elastic


@dataclass
class StateVariables:
    """
    Thermodynamic state variables for VBR calculations.
    
    All arrays must have the same shape (except frequency).
    """
    T_K: np.ndarray  # Temperature in Kelvin
    P_GPa: np.ndarray  # Pressure in GPa
    rho: np.ndarray  # Density in kg/m^3
    dg_um: np.ndarray  # Grain size in micrometers
    phi: np.ndarray  # Melt fraction (0-1)
    sig_MPa: np.ndarray  # Differential stress in MPa
    f: np.ndarray  # Frequency in Hz (1D array)
    Ch2o: Optional[np.ndarray] = None  # Water content in ppm
    chi: Optional[np.ndarray] = None  # Olivine fraction (for mixing)
    Tsolidus_K: Optional[np.ndarray] = None  # Solidus temperature in K
    
    def __post_init__(self):
        """Validate and set defaults."""
        shape = self.T_K.shape
        
        if self.Ch2o is None:
            self.Ch2o = np.zeros(shape)
        if self.chi is None:
            self.chi = np.ones(shape)
    
    @property
    def shape(self) -> tuple:
        return self.T_K.shape
    
    @property
    def n_freq(self) -> int:
        return len(self.f)


class VBR:
    """
    VBR (Very Broadband Rheology) calculator.
    
    Calculates elastic moduli, viscosity, and anelastic properties
    from thermodynamic state variables.
    
    Parameters
    ----------
    sv : StateVariables
        Thermodynamic state variables
    elastic_methods : list of str, optional
        Elastic methods to use (default: ['anharmonic'])
    anelastic_methods : list of str, optional
        Anelastic methods to use
    viscous_methods : list of str, optional
        Viscous methods to use
        
    Attributes
    ----------
    input : dict
        Input parameters and state variables
    output : dict
        Calculation results
        
    Examples
    --------
    >>> sv = StateVariables(
    ...     T_K=np.array([1400]),
    ...     P_GPa=np.array([3.0]),
    ...     rho=np.array([3300]),
    ...     dg_um=np.array([1000]),
    ...     phi=np.array([0.0]),
    ...     sig_MPa=np.array([0.1]),
    ...     f=np.logspace(-2.2, -1.3, 10)
    ... )
    >>> vbr = VBR(sv, anelastic_methods=['eburgers_psp'])
    >>> vbr.run()
    >>> print(vbr.output['anelastic']['eburgers_psp']['V'])
    """
    
    def __init__(
        self,
        sv: StateVariables,
        elastic_methods: Optional[List[str]] = None,
        anelastic_methods: Optional[List[str]] = None,
        viscous_methods: Optional[List[str]] = None,
        global_params: Optional[Params_Global] = None,
    ):
        self.sv = sv
        self.global_params = global_params or Params_Global()
        
        # Set default methods
        if elastic_methods is None:
            elastic_methods = ['anharmonic']
        if anelastic_methods is None:
            anelastic_methods = []
        if viscous_methods is None:
            viscous_methods = []
            
        self.elastic_methods = elastic_methods
        self.anelastic_methods = anelastic_methods
        self.viscous_methods = viscous_methods
        
        # Initialize input structure
        self.input = {
            'SV': sv,
            'elastic': {
                'methods_list': elastic_methods,
            },
            'anelastic': {
                'methods_list': anelastic_methods,
            },
            'viscous': {
                'methods_list': viscous_methods,
            },
            'GlobalSettings': self.global_params,
        }
        
        # Load default parameters for each method
        for method in elastic_methods:
            self.input['elastic'][method] = Params_Elastic(method)
        for method in anelastic_methods:
            self.input['anelastic'][method] = Params_Anelastic(method, self.global_params)
        for method in viscous_methods:
            self.input['viscous'][method] = Params_Viscous(method, self.global_params)
        
        # Initialize output structure
        self.output = {
            'elastic': {},
            'viscous': {},
            'anelastic': {},
        }
    
    def run(self) -> 'VBR':
        """
        Run all VBR calculations.
        
        Returns
        -------
        VBR
            Self, with results in output attribute
        """
        # Run elastic calculations first (needed by anelastic)
        if self.elastic_methods:
            self._run_elastic()
        
        # Run viscous calculations (may be needed by anelastic)
        if self.viscous_methods:
            self._run_viscous()
        
        # Run anelastic calculations
        if self.anelastic_methods:
            self._run_anelastic()
        
        return self
    
    def _get_base_elastic_output(self):
        """Get the base elastic output dict (cammarano2003 or anharmonic).

        Falls back to running anharmonic if neither has been computed.
        """
        if 'cammarano2003' in self.output['elastic']:
            return self.output['elastic']['cammarano2003']
        elif 'anharmonic' in self.output['elastic']:
            return self.output['elastic']['anharmonic']
        else:
            self._el_anharmonic()
            return self.output['elastic']['anharmonic']

    def _get_unrelaxed_Gu(self):
        """Get unrelaxed shear modulus (melt-corrected if available)."""
        if 'anh_poro' in self.output['elastic']:
            return self.output['elastic']['anh_poro']['Gu']
        return self._get_base_elastic_output()['Gu']

    def _run_elastic(self):
        """Run elastic calculations."""
        for method in self.elastic_methods:
            if method == 'anharmonic':
                self._el_anharmonic()
            elif method == 'anh_poro':
                self._el_anh_poro()
            elif method == 'cammarano2003':
                self._el_cammarano2003()
    
    def _run_viscous(self):
        """Run viscous calculations."""
        for method in self.viscous_methods:
            if method == 'xfit_premelt':
                self._visc_xfit_premelt()
            elif method == 'HK2003':
                self._visc_HK2003()
    
    def _run_anelastic(self):
        """Run anelastic calculations."""
        for method in self.anelastic_methods:
            if method == 'eburgers_psp':
                self._Q_eburgers_psp()
            elif method == 'xfit_premelt':
                self._Q_xfit_premelt()
            elif method == 'andrade_psp':
                self._Q_andrade_psp()
            elif method == 'xfit_mxw':
                self._Q_xfit_mxw()
    
    def _el_cammarano2003(self):
        """
        Calculate anharmonic elastic moduli using Cammarano et al. (2003)
        finite-strain mineral physics model.
        
        Uses depth/pressure-dependent mineral assemblage with 3rd-order
        Birch-Murnaghan finite-strain theory and Voigt-Reuss-Hill averaging.
        Automatically selects upper mantle, transition zone, or lower mantle
        mineralogy based on pressure.
        """
        sv = self.sv
        params = self.input['elastic']['cammarano2003']
        
        X_Fe = params.get('X_Fe', 0.1)
        composition = params.get('composition', 'pyrolite')
        
        # P is constant across the 3D grid at each depth step
        P_GPa_scalar = float(sv.P_GPa.flat[0])
        
        # Extract unique temperatures (T varies only along axis 0 in the meshgrid)
        if sv.T_K.ndim >= 2:
            T_unique = sv.T_K[:, 0, 0] if sv.T_K.ndim == 3 else sv.T_K[:, 0]
        else:
            T_unique = sv.T_K
        
        # Compute moduli using Cammarano finite-strain method
        G_Pa, K_Pa, rho_cam, Vs, Vp = cammarano_elastic(
            T_unique, P_GPa_scalar, X_Fe=X_Fe, composition=composition)
        
        # Broadcast back to full grid shape
        shape = sv.T_K.shape
        if len(shape) == 3:
            Gu = np.broadcast_to(G_Pa[:, np.newaxis, np.newaxis], shape).copy()
            Ku = np.broadcast_to(K_Pa[:, np.newaxis, np.newaxis], shape).copy()
            Vsu = np.broadcast_to(Vs[:, np.newaxis, np.newaxis], shape).copy()
            Vpu = np.broadcast_to(Vp[:, np.newaxis, np.newaxis], shape).copy()
        elif len(shape) == 2:
            Gu = np.broadcast_to(G_Pa[:, np.newaxis], shape).copy()
            Ku = np.broadcast_to(K_Pa[:, np.newaxis], shape).copy()
            Vsu = np.broadcast_to(Vs[:, np.newaxis], shape).copy()
            Vpu = np.broadcast_to(Vp[:, np.newaxis], shape).copy()
        else:
            Gu, Ku, Vsu, Vpu = G_Pa, K_Pa, Vs, Vp
        
        self.output['elastic']['cammarano2003'] = {
            'Gu': Gu,    # Pa
            'Ku': Ku,    # Pa
            'Vpu': Vpu,  # m/s
            'Vsu': Vsu,  # m/s
            'units': {'Gu': 'Pa', 'Ku': 'Pa', 'Vpu': 'm/s', 'Vsu': 'm/s'}
        }
    
    def _el_anharmonic(self):
        """
        Calculate anharmonic elastic moduli.
        
        Calculates shear (Gu) and bulk (Ku) moduli at temperature and pressure,
        then derives unrelaxed velocities.
        """
        sv = self.sv
        params = self.input['elastic']['anharmonic']
        
        # Get scaling choices
        t_scale = params['temperature_scaling']
        p_scale = params['pressure_scaling']
        ref_scale = params.get('reference_scaling', 'default')
        
        # Reference values based on reference_scaling
        if ref_scale == 'upper_mantle':
            T_K_ref = params['upper_mantle']['T_K_ref']
            P_Pa_ref = params['upper_mantle']['P_Pa_ref']
            Gu_0 = params['upper_mantle']['Gu_0'] * 1e9  # Convert GPa to Pa
            Ku_0 = params['upper_mantle']['Ku_0'] * 1e9
        else:  # 'default'
            T_K_ref = params['T_K_ref']
            P_Pa_ref = params['P_Pa_ref']
            Gu_0 = params['Gu_0_ol'] * 1e9  # Convert GPa to Pa
            Ku_0 = params['Ku_0_ol'] * 1e9
        
        # Temperature and pressure changes
        dT = sv.T_K - T_K_ref
        dP = sv.P_GPa * 1e9 - P_Pa_ref
        
        # Temperature derivatives
        if t_scale == 'isaak':
            dG_dT = params['isaak']['dG_dT']
            dK_dT = params['isaak']['dK_dT']
        elif t_scale == 'cammarano':
            dG_dT = params['cammarano']['dG_dT']
            dK_dT = params['cammarano']['dK_dT']
        elif t_scale == 'upper_mantle':
            dG_dT = params['upper_mantle']['dG_dT']
            dK_dT = params['upper_mantle']['dK_dT']
        else:
            raise ValueError(f"Unknown temperature_scaling: {t_scale}. "
                           f"Available options: {params.get('available_temperature_scaling', ['isaak', 'cammarano', 'upper_mantle'])}")
        
        # Pressure derivatives
        if p_scale == 'cammarano':
            dG_dP = params['cammarano']['dG_dP']
            dG_dP2 = params['cammarano']['dG_dP2']
            dK_dP = params['cammarano']['dK_dP']
            dK_dP2 = params['cammarano']['dK_dP2']
        elif p_scale == 'abramson':
            dG_dP = params['abramson']['dG_dP']
            dG_dP2 = params['abramson']['dG_dP2']
            dK_dP = params['abramson']['dK_dP']
            dK_dP2 = params['abramson']['dK_dP2']
        elif p_scale == 'upper_mantle':
            dG_dP = params['upper_mantle']['dG_dP']
            dG_dP2 = params['upper_mantle']['dG_dP2']
            dK_dP = params['upper_mantle']['dK_dP']
            dK_dP2 = params['upper_mantle']['dK_dP2']
        else:
            raise ValueError(f"Unknown pressure_scaling: {p_scale}. "
                           f"Available options: {params.get('available_pressure_scaling', ['cammarano', 'abramson', 'upper_mantle'])}")
        
        # Calculate moduli at T, P
        Gu_TP = Gu_0 + dG_dT * dT + dG_dP * dP + dG_dP2 * dP**2
        Ku_TP = Ku_0 + dK_dT * dT + dK_dP * dP + dK_dP2 * dP**2
        
        # Calculate velocities
        Vpu = np.sqrt((Ku_TP + 4/3 * Gu_TP) / sv.rho)
        Vsu = np.sqrt(Gu_TP / sv.rho)
        
        self.output['elastic']['anharmonic'] = {
            'Gu': Gu_TP,  # Pa
            'Ku': Ku_TP,  # Pa
            'Vpu': Vpu,   # m/s
            'Vsu': Vsu,   # m/s
            'units': {'Gu': 'Pa', 'Ku': 'Pa', 'Vpu': 'm/s', 'Vsu': 'm/s'}
        }
    
    def _el_anh_poro(self):
        """
        Calculate poro-elastic melt effect on moduli.
        
        Implements Takei 2002, JGR Solid Earth, Appendix A.
        
        Requires anharmonic or cammarano2003 to be run first.
        """
        anh = self._get_base_elastic_output()
        
        sv = self.sv
        params = self.input['elastic']['anh_poro']
        
        Gu = anh['Gu']
        Ku = anh['Ku']
        phi = sv.phi
        rho = sv.rho
        
        # Poro-elastic parameters from Takei 2002
        A = params['Melt_A']    # wetting angle factor (1:2.3, Yoshino)
        Km = params['Melt_Km']  # bulk modulus of the melt (Pa)
        nu = params['Melt_nu']  # Poisson's ratio
        
        # Calculate effective moduli using Takei 2002 formulation
        Gu_eff, Gamma_G = self._melt_shear_moduli(Gu, phi, A, nu)
        Ku_eff, Gamma_K = self._melt_bulk_moduli(Ku, phi, A, Km, nu)
        
        # Calculate velocities with poro-elastic effects
        Vpu, Vsu = self._Vp_Vs_poro(phi, Gu, Ku, Gamma_G, Gamma_K, rho, Km)
        
        self.output['elastic']['anh_poro'] = {
            'Gu': Gu_eff,
            'Ku': Ku_eff,
            'Vpu': Vpu,
            'Vsu': Vsu,
        }
    
    def _melt_shear_moduli(self, mu: np.ndarray, phi: np.ndarray, A: float, nu: float):
        """
        Calculate shear moduli with poro-elastic melt effect.
        
        Following Takei 2002, Appendix A.
        """
        # Calculate contiguity as function of melt fraction
        Psi = 1 - A * np.sqrt(phi)
        
        # Coefficients from Takei 2002, Table A1 (for shear modulus)
        b = np.array([
            [1.6122, 0.13572, 0],
            [4.5869, 3.6086, 0],
            [-7.5395, -4.8676, -4.3182]
        ])
        
        # Calculate b_vec (polynomial in nu)
        b_vec = np.array([np.sum(b[i, :] * nu**np.arange(3)) for i in range(3)])
        
        # Exponent n_mu (eqn A6, Takei 2002)
        n_mu = b_vec[0] * Psi + b_vec[1] * (1 - Psi) + b_vec[2] * Psi * (1 - Psi)**2
        
        # Normalized skeleton properties
        Gamma_Mu = (1 - phi) * (1 - (1 - Psi)**n_mu) # should be vector? size mu
        mu_sk_prime = 1 - (1 - Psi)**n_mu
        Mu_sk = (1 - phi) * mu_sk_prime * mu
        
        # Effective shear modulus with melt
        Mu_eff = Mu_sk
        
        return Mu_eff, Gamma_Mu
    
    def _melt_bulk_moduli(self, k: np.ndarray, phi: np.ndarray, A: float, Km: float, nu: float):
        """
        Calculate bulk moduli with poro-elastic melt effect.
        
        Following Takei 2002, Appendix A.
        """
        # Calculate contiguity as function of melt fraction
        Psi = 1 - A * np.sqrt(phi)
        
        # Coefficients from Takei 2002, Table A1 (for bulk modulus)
        a = np.array([
            [1.8625, 0.52594, -4.8397, 0],
            [4.5001, -6.1551, -4.3634, 0],
            [-5.6512, 6.9159, 29.595, -58.96]
        ])
        
        # Calculate a_vec (polynomial in nu)
        a_vec = np.array([np.sum(a[i, :] * nu**np.arange(4)) for i in range(3)])
        
        # Exponent n_k (eqns A5, A6, Takei 2002)
        n_k = a_vec[0] * Psi + a_vec[1] * (1 - Psi) + a_vec[2] * Psi * (1 - Psi)**1.5
        
        # Normalized skeleton properties
        Gamma_k = (1 - phi) * (1 - (1 - Psi)**n_k) # should be vector? size k
        k_sk_prime = 1 - (1 - Psi)**n_k
        K_sk = k_sk_prime * k
        
        # Effective bulk modulus with melt
        delta = 1e-20  # Avoid division by zero at phi=0
        top = (1 - K_sk / k)**2
        bot = (1 - phi - K_sk / k + phi * k / Km) + delta * (phi == 0)
        Kb_eff_prime = K_sk / k + top / bot
        Kb_eff = Kb_eff_prime * k
        
        return Kb_eff, Gamma_k
    
    def _Vp_Vs_poro(self, phi: np.ndarray, Gu: np.ndarray, Ku: np.ndarray, 
                    Gamma_G: np.ndarray, Gamma_K: np.ndarray, rho: np.ndarray, Km: float):
        """
        Calculate Vp and Vs accounting for poro-elastic effects.
        
        Reduces to pure phase calculation when phi = 0.
        """
        # Poro-elastic factors
        K1 = (1 - Gamma_K)**2
        K2 = 1 - phi - Gamma_K + phi * Ku / Km # equals 0 if phi = 0 becuase gamma_K would be 1
        
        # Effective bulk and shear modulus
        delta = 1e-20  # Avoid division by zero
        bulk_mod = Ku * (Gamma_K + K1 / (K2 + delta)) 
        shear_mod = Gu * Gamma_G
        
        # Calculate Vp, Vs
        Vp = np.sqrt((bulk_mod + 4/3 * shear_mod) / rho)
        Vs = np.sqrt(shear_mod / rho)
        
        return Vp, Vs
    
    @staticmethod
    def _sr_water_fugacity(H2O_PPM, H2O_o, P_Pa, T_K):
        """Calculate water fugacity following Hirth & Kohlstedt 2003 eq 6."""
        E = 40e3     # activation energy [J/mol]
        V = 10e-6    # activation volume [m^3/mol]
        R = 8.314    # gas constant
        A_o = 26.0   # pre-exponential [PPM/MPa]
        fH2O = np.where(H2O_PPM >= H2O_o,
                        (H2O_PPM / A_o) * np.exp((E + P_Pa * V) / (R * T_K)),
                        0.0)
        return fH2O

    @staticmethod
    def _sr_melt_enhancement(phi, alpha, x_phi_c, phi_c):
        """Strain rate melt enhancement factor (Holtzman 2016)."""
        a = np.where(x_phi_c > 0, np.log(np.where(x_phi_c > 0, x_phi_c, 1.0)), 0.0)
        ratefac = np.where(phi_c > 0, 1.0 / np.where(phi_c > 0, phi_c, 1.0), 0.0)
        step = a * erf(phi * ratefac)
        slope = alpha * phi
        return np.exp(slope + step)

    def _visc_HK2003(self):
        """Calculate viscosity using Hirth & Kohlstedt 2003 flow law.

        Matches MATLAB sr_visc_calc_HK2003.m exactly.
        """
        sv = self.sv
        params = self.input['viscous']['HK2003']

        T_K = sv.T_K
        P_Pa = sv.P_GPa * 1e9
        sig = sv.sig_MPa
        d = sv.dg_um
        phi = sv.phi
        Ch2o = sv.Ch2o if sv.Ch2o is not None else np.zeros_like(T_K)
        ch2o_o = params['ch2o_o']
        R = params['R']

        # Pressure dependence
        if params['P_dep_calc'] == 'yes':
            P_eff = P_Pa
        else:
            P_eff = np.zeros_like(P_Pa)

        # Water fugacity
        fH2O = self._sr_water_fugacity(Ch2o, ch2o_o, P_eff, T_K)

        # Melt enhancement flag
        melt_enh = self.global_params.melt_enhancement

        sr_tot = np.zeros_like(T_K, dtype=np.float64)
        result = {}

        for mech in ['diff', 'disl', 'gbs']:
            if mech not in params:
                continue
            mp = params[mech]

            # Select wet/dry or T-split parameters (prep_constants equivalent)
            if mech in ('diff', 'disl'):
                is_wet = fH2O > 0
                A   = np.where(is_wet, mp.get('A_wet', mp['A']),   mp['A'])
                Q   = np.where(is_wet, mp.get('Q_wet', mp['Q']),   mp['Q'])
                V   = np.where(is_wet, mp.get('V_wet', mp['V']),   mp['V'])
                p   = np.where(is_wet, mp.get('p_wet', mp['p']),   mp['p'])
                n   = np.where(is_wet, mp.get('n_wet', mp['n']),   mp['n'])
                r   = np.where(is_wet, mp.get('r_wet', mp['r']),   mp['r'])
                alf = np.where(is_wet, mp.get('alf_wet', mp['alf']), mp['alf'])
                phi_c_m = np.where(is_wet, mp.get('phi_c_wet', mp['phi_c']), mp['phi_c'])
                x_phi_c_m = np.where(is_wet, mp.get('x_phi_c_wet', mp['x_phi_c']), mp['x_phi_c'])
            elif mech == 'gbs':
                # Temperature-dependent parameters
                T_C = T_K - 273.0
                hi = T_C >= 1250
                A   = np.where(hi, mp['A_gt1250'],   mp['A_lt1250'])
                Q   = np.where(hi, mp['Q_gt1250'],   mp['Q_lt1250'])
                V   = np.where(hi, mp['V_gt1250'],   mp['V_lt1250'])
                p   = np.where(hi, mp['p_gt1250'],   mp['p_lt1250'])
                n   = np.where(hi, mp['n_gt1250'],   mp['n_lt1250'])
                r   = np.where(hi, mp['r_gt1250'],   mp['r_lt1250'])
                alf = np.where(hi, mp['alf_gt1250'], mp['alf_lt1250'])
                phi_c_m = np.where(hi, mp['phi_c_gt1250'], mp['phi_c_lt1250'])
                x_phi_c_m = np.where(hi, mp['x_phi_c_gt1250'], mp['x_phi_c_lt1250'])

            # Override x_phi_c when melt_enhancement is off
            if melt_enh == 0:
                x_phi_c_m = np.ones_like(T_K)

            # Flow law: sr = A * sig^n * d^(-p) * exp(-(Q+P*V)/(R*T)) * fH2O^r
            sr = (A * (sig ** n) * (d ** (-p))
                  * np.exp(-(Q + P_eff * V) / (R * T_K))
                  * np.where(fH2O > 0, fH2O ** r, np.ones_like(fH2O)))

            # Correct to truly melt-free
            sr = sr / x_phi_c_m

            # Apply melt enhancement
            enhance = self._sr_melt_enhancement(phi, alf, x_phi_c_m, phi_c_m)
            sr = sr * enhance

            sr_tot = sr_tot + sr
            result[mech] = {
                'sr': sr,
                'eta': sig * 1e6 / sr,
            }

        result['sr_tot'] = sr_tot
        result['eta_total'] = sig * 1e6 / sr_tot
        result['units'] = {'sr': '1/s', 'eta': 'Pa*s'}
        self.output['viscous']['HK2003'] = result

    def _visc_xfit_premelt(self):
        """
        Calculate viscosity using Yamauchi & Takei 2016 near-solidus scaling.
        """
        sv = self.sv
        params = self.input['viscous']['xfit_premelt']
        
        # Need solidus
        if sv.Tsolidus_K is None:
            sv.Tsolidus_K = calculate_solidus_K(
                sv.P_GPa,
                method=params.get('solidus_method', 'hirschmann'),
                density_model=params.get('density_model', 'constant'),
                density_rho=params.get('density_rho', 3400.0),
                density_file=params.get('density_file'),
            )
        
        Tn = sv.T_K / sv.Tsolidus_K  # Normalized temperature
        
        # Calculate melt-free viscosity (YT2016 exact form)
        eta_meltfree = self._YT2016_melt_free_viscosity(params)
        
        # Calculate near-solidus activation energy factor A_n and melt effects
        A_n = self._calc_A_n(Tn, sv.phi, params)
        
        # Full viscosity
        eta = A_n * eta_meltfree
        
        self.output['viscous']['xfit_premelt'] = {
            'diff': {
                'eta': eta,
                'eta_meltfree': eta_meltfree,
            },
            'units': {'eta': 'Pa*s'}
        }
    
    def _YT2016_melt_free_viscosity(self, params: Dict) -> np.ndarray:
        """Calculate melt-free diffusion creep viscosity from YT2016."""
        sv = self.sv
        
        Tr = params['Tr_K']
        Pr = params['Pr_Pa']
        eta_r = params['eta_r']
        dr = params['dg_um_r']
        H = params['H']
        V = params['V']
        R = params['R']
        m = params['m']
        
        P = sv.P_GPa * 1e9
        
        eta = eta_r * (sv.dg_um / dr)**m * \
              np.exp(V / R * (P / sv.T_K - Pr / Tr)) * \
              np.exp(H / R * (1 / sv.T_K - 1 / Tr))
        
        return eta
    
    def _calc_A_n(self, Tn: np.ndarray, phi: np.ndarray, params: Dict) -> np.ndarray:
        """Calculate near-solidus pre-exponential factor and melt effects from YT2016."""
        T_eta = params['T_eta']
        gamma = params['gamma']
        lambd = params['alpha']  # lambda in YT2016
        B = params['B']
        
        A_n = np.ones_like(Tn)
        
        # Below T_eta: A_n = 1
        mask1 = Tn < T_eta
        A_n[mask1] = 1.0
        
        # T_eta <= Tn < 1
        mask2 = (Tn >= T_eta) & (Tn < 1)
        A_n[mask2] = np.exp(-(Tn[mask2] - T_eta) / (Tn[mask2] - Tn[mask2] * T_eta) * np.log(gamma))
        
        # Tn >= 1 (above solidus)
        mask3 = Tn >= 1
        A_n[mask3] = np.exp(-lambd * phi[mask3]) / gamma / B
        
        return A_n
    
    def _Q_eburgers_psp(self):
        """
        Calculate extended Burgers anelastic properties.
        
        Dispatches to PointWise (Q_eBurgers_f.m) or FastBurger (Q_eFastBurgers.m)
        based on params['method']. Default is 'PointWise' to match MATLAB workflow.
        
        FastBurger only works with the high-temp background (no dissipation peak).
        Use PointWise if using a peak (DeltaP > 0).
        
        References
        ----------
        Jackson & Faul, 2010, Phys. Earth Planet. Inter.
        """
        params = self.input['anelastic']['eburgers_psp']
        method = params.get('method', 'PointWise').lower()
        
        if method == 'pointwise':
            self._Q_eburgers_pointwise()
        elif method == 'fastburger':
            self._Q_eburgers_fastburger()
        else:
            raise ValueError(f"Unknown eburgers method: '{params.get('method')}'. Must be 'PointWise' or 'FastBurger'.")
    
    def _Q_eburgers_fastburger(self):
        """
        Calculate extended Burgers anelastic properties using FastBurger algorithm. (Q_eFastBurgers.m in Matlab)
        
        FastBurger pre-computes cumulative integrals over a global tau grid and then
        interpolates per thermodynamic state. Only supports the high-temp background
        (no dissipation peak).
        
        References
        ----------
        Jackson & Faul, 2010, Phys. Earth Planet. Inter.
        """
        sv = self.sv
        params = self.input['anelastic']['eburgers_psp']
        
        # Get unrelaxed modulus
        Gu = self._get_unrelaxed_Gu()
        
        Ju = 1.0 / Gu
        rho = sv.rho
        f_vec = sv.f
        w_vec = 2 * np.pi * f_vec # period
        
        # Get Burgers parameters
        bType = params['eBurgerFit']
        bp = params[bType]
        
        alf = bp['alf']
        DeltaB = bp['DeltaB'] # relaxation strength of the background
        
        # Calculate Maxwell times
        tau = self._eBurgers_maxwell_times(params, Gu)
        
        # Build global tau vector for integration
        n_glob = params['nTauGlob']
        min_tau_L = np.min(tau['L'])
        max_tau_H = np.max(tau['H'])
        
        Tau_glob = np.logspace(np.log10(min_tau_L), np.log10(max_tau_H), n_glob)
        
        # Add exact tau boundaries
        Tau_glob = np.unique(np.concatenate([Tau_glob, tau['L'].flatten(), tau['H'].flatten()]))
        Tau_glob = np.sort(Tau_glob)
        
        # Pre-compute cumulative integrals for each frequency using trapezoidal rule
        # (matches MATLAB's cumtrapz(Tau_glob_vec, y) exactly)
        n_freq = len(f_vec)
        integrals_J1 = []
        integrals_J2 = []
        
        for w in w_vec:
            y_J1 = Tau_glob**(alf - 1) / (1 + w**2 * Tau_glob**2)
            y_J2 = Tau_glob**alf / (1 + w**2 * Tau_glob**2)
            dx = np.diff(Tau_glob)
            
            # Cumulative trapezoidal: (y[i] + y[i+1]) / 2 * dx[i]
            int_J1 = np.concatenate([[0], np.cumsum((y_J1[:-1] + y_J1[1:]) / 2 * dx)])
            int_J2 = np.concatenate([[0], np.cumsum((y_J2[:-1] + y_J2[1:]) / 2 * dx)])
            
            integrals_J1.append(int_J1)
            integrals_J2.append(int_J2)
        
        # Output arrays
        sv_shape = sv.shape
        n_sv = np.prod(sv_shape)
        
        J1 = np.zeros((n_sv, n_freq))
        J2 = np.zeros((n_sv, n_freq))
        M = np.zeros((n_sv, n_freq))
        V = np.zeros((n_sv, n_freq))
        
        # Flatten for iteration
        tau_M_flat = tau['maxwell'].flatten()
        tau_L_flat = tau['L'].flatten()
        tau_H_flat = tau['H'].flatten()
        Ju_flat = Ju.flatten()
        rho_flat = rho.flatten()
        
        for i_th in range(n_sv):
            Tau_M = tau_M_flat[i_th]
            Tau_L = tau_L_flat[i_th]
            Tau_H = tau_H_flat[i_th]
            Ju_i = Ju_flat[i_th]
            rho_i = rho_flat[i_th]
            
            # Find integration bounds using exact equality match
            # (matching MATLAB: iLow = find(Tau_glob_vec==Tau_L0))
            i_low_arr = np.where(Tau_glob == Tau_L)[0]
            i_high_arr = np.where(Tau_glob == Tau_H)[0]
            
            if len(i_low_arr) > 0:
                i_low = i_low_arr[0]
            else:
                i_low = np.searchsorted(Tau_glob, Tau_L)
                i_low = max(0, min(i_low, len(Tau_glob) - 1))
            
            if len(i_high_arr) > 0:
                i_high = i_high_arr[0]
            else:
                i_high = np.searchsorted(Tau_glob, Tau_H)
                i_high = max(0, min(i_high, len(Tau_glob) - 1))
            
            for iw in range(n_freq):
                w = w_vec[iw]
                
                int_J1 = integrals_J1[iw]
                int_J2 = integrals_J2[iw]
                
                # Integration contribution
                denom = Tau_H**alf - Tau_L**alf
                if denom > 0:
                    d_int_J1 = alf * (int_J1[i_high] - int_J1[i_low]) / denom
                    d_int_J2 = alf * (int_J2[i_high] - int_J2[i_low]) / denom
                else:
                    d_int_J1 = 0
                    d_int_J2 = 0
                
                J1_val = Ju_i * (1 + DeltaB * d_int_J1)
                J2_val = Ju_i * (w * DeltaB * d_int_J2 + 1 / (w * Tau_M))
                
                J1[i_th, iw] = J1_val
                J2[i_th, iw] = J2_val
                
                M[i_th, iw] = 1.0 / np.sqrt(J1_val**2 + J2_val**2)
                V[i_th, iw] = np.sqrt(M[i_th, iw] / rho_i)
        
        # Reshape to include frequency dimension
        out_shape = sv_shape + (n_freq,)
        J1 = J1.reshape(out_shape)
        J2 = J2.reshape(out_shape)
        M = M.reshape(out_shape)
        V = V.reshape(out_shape)
        
        # Calculate Q
        Qinv = J2 / J1
        Q = 1.0 / Qinv
        
        # Average velocity over frequency
        Vave = np.mean(V, axis=-1)
        
        self.output['anelastic']['eburgers_psp'] = {
            'J1': J1,
            'J2': J2,
            'Q': Q,
            'Qinv': Qinv,
            'M': M,
            'V': V,
            'Vave': Vave,
            'units': {
                'J1': '1/Pa', 'J2': '1/Pa', 'Q': '', 'Qinv': '',
                'M': 'Pa', 'V': 'm/s', 'Vave': 'm/s'
            }
        }
    
    def _Q_eburgers_pointwise(self):
        """
        Calculate extended Burgers anelastic properties using PointWise method.
        (Q_eBurgers_f.m in Matlab)
        
        Loops over each thermodynamic state and frequency, performing numerical
        integration at each point. Supports both high-temp background and
        optional dissipation peak (DeltaP > 0).
        
        References
        ----------
        Jackson & Faul, 2010, Phys. Earth Planet. Inter.
        """
        sv = self.sv
        params = self.input['anelastic']['eburgers_psp']
        
        # Get unrelaxed modulus
        Gu = self._get_unrelaxed_Gu()
        
        Ju = 1.0 / Gu
        rho = sv.rho
        f_vec = sv.f
        w_vec = 2 * np.pi * f_vec
        
        # Get Burgers parameters
        bType = params['eBurgerFit']
        bp = params[bType]
        
        alf = bp['alf']
        DeltaB = bp['DeltaB']
        DeltaP = bp.get('DeltaP', 0.0)
        sig = bp.get('sig', 0.0)
        HTB_int_meth = params.get('integration_method', 0)
        ntau = params.get('tau_integration_points', 500)
        
        # Calculate Maxwell times
        tau = self._eBurgers_maxwell_times(params, Gu)
        
        # Output arrays
        sv_shape = sv.shape
        n_sv = np.prod(sv_shape)
        n_freq = len(f_vec)
        
        J1 = np.zeros((n_sv, n_freq))
        J2 = np.zeros((n_sv, n_freq))
        M = np.zeros((n_sv, n_freq))
        V = np.zeros((n_sv, n_freq))
        
        # Flatten for iteration
        tau_M_flat = tau['maxwell'].flatten()
        tau_L_flat = tau['L'].flatten()
        tau_H_flat = tau['H'].flatten()
        tau_P_flat = tau['P'].flatten()
        Ju_flat = Ju.flatten()
        rho_flat = rho.flatten()
        
        for i_th in range(n_sv):
            Ju_i = Ju_flat[i_th]
            rho_i = rho_flat[i_th]
            Tau_M = tau_M_flat[i_th]
            Tau_L = tau_L_flat[i_th]
            Tau_H = tau_H_flat[i_th]
            Tau_P = tau_P_flat[i_th]
            
            # Build tau vector for trapezoidal integration
            if HTB_int_meth == 0:
                Tau_X_vec = np.logspace(np.log10(Tau_L), np.log10(Tau_H), ntau)
            
            for iw in range(n_freq):
                w = w_vec[iw]
                
                if HTB_int_meth == 0:
                    # Trapezoidal integration (matches MATLAB HTB_int_meth==0)
                    D_vec = (alf * Tau_X_vec**(alf - 1)) / (Tau_H**alf - Tau_L**alf)
                    
                    int_J1 = np.trapz(D_vec / (1 + w**2 * Tau_X_vec**2), Tau_X_vec)
                    J1_val = 1 + DeltaB * int_J1
                    
                    int_J2 = np.trapz((Tau_X_vec * D_vec) / (1 + w**2 * Tau_X_vec**2), Tau_X_vec)
                    J2_val = w * DeltaB * int_J2 + 1 / (w * Tau_M)
                    
                else:
                    # Quadrature integration (matches MATLAB HTB_int_meth==1 or 2)
                    from scipy.integrate import quad
                    
                    Tau_fac = alf * DeltaB / (Tau_H**alf - Tau_L**alf)
                    
                    int1, _ = quad(lambda x: x**(alf - 1) / (1 + (w * x)**2), Tau_L, Tau_H)
                    int2, _ = quad(lambda x: x**alf / (1 + (w * x)**2), Tau_L, Tau_H)
                    
                    J1_val = 1 + Tau_fac * int1
                    J2_val = w * Tau_fac * int2 + 1 / (w * Tau_M)
                
                # Add dissipation peak if DeltaP > 0
                if DeltaP > 0 and sig > 0:
                    from scipy.integrate import quad
                    
                    def peak_J2_integrand(x):
                        return np.exp(-(np.log(x / Tau_P) / sig)**2 / 2) / (1 + (w * x)**2)
                    
                    def peak_J1_integrand(x):
                        return (1.0 / x) * np.exp(-(np.log(x / Tau_P) / sig)**2 / 2) / (1 + (w * x)**2)
                    
                    int2a, _ = quad(peak_J2_integrand, 0, np.inf, limit=200)
                    J2_val += DeltaP * w * int2a / (sig * np.sqrt(2 * np.pi))
                    
                    int1a, _ = quad(peak_J1_integrand, 0, np.inf, limit=200)
                    J1_val += DeltaP * int1a / (sig * np.sqrt(2 * np.pi))
                
                # Multiply on the unrelaxed compliance
                J1_val = Ju_i * J1_val
                J2_val = Ju_i * J2_val
                
                J1[i_th, iw] = J1_val
                J2[i_th, iw] = J2_val
                M[i_th, iw] = (J1_val**2 + J2_val**2)**(-0.5)
                V[i_th, iw] = np.sqrt(M[i_th, iw] / rho_i)
        
        # Reshape to include frequency dimension
        out_shape = sv_shape + (n_freq,)
        J1 = J1.reshape(out_shape)
        J2 = J2.reshape(out_shape)
        M = M.reshape(out_shape)
        V = V.reshape(out_shape)
        
        # Calculate Q
        Qinv = J2 / J1
        Q = 1.0 / Qinv
        
        # Average velocity over frequency
        Vave = np.mean(V, axis=-1)
        
        self.output['anelastic']['eburgers_psp'] = {
            'J1': J1,
            'J2': J2,
            'Q': Q,
            'Qinv': Qinv,
            'M': M,
            'V': V,
            'Vave': Vave,
            'tau_M': tau['maxwell'],
            'units': {
                'J1': '1/Pa', 'J2': '1/Pa', 'Q': '', 'Qinv': '',
                'M': 'Pa', 'V': 'm/s', 'Vave': 'm/s', 'tau_M': 's'
            }
        }
    
    def _eBurgers_maxwell_times(self, params: Dict, Gu: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate Maxwell times for extended Burgers model.
        """
        sv = self.sv
        bType = params['eBurgerFit']
        bp = params[bType]
        
        T_K = sv.T_K
        P_Pa = sv.P_GPa * 1e9
        d = sv.dg_um
        phi = sv.phi
        
        # Reference values
        TR = bp['TR']
        PR = bp['PR'] * 1e9
        dR = bp['dR']
        E = bp['E']
        R = params['R']
        Vstar = bp['Vstar']
        m_a = bp['m_a']
        m_v = bp['m_v']
        
        # Scaling for maxwell time
        scale = (d / dR)**m_v * \
                np.exp((E / R) * (1 / T_K - 1 / TR)) * \
                np.exp((Vstar / R) * (P_Pa / T_K - PR / TR))
        
        # Melt effects
        scale = self._add_melt_effects(phi, scale, params)
        
        Tau_MR = bp['Tau_MR']
        tau_maxwell = Tau_MR * scale    # steady state viscous Maxwell time
        
        # Integration limits
        LHP = (d / dR)**m_a * \
              np.exp((E / R) * (1 / T_K - 1 / TR)) * \
              np.exp((Vstar / R) * (P_Pa / T_K - PR / TR))
        LHP = self._add_melt_effects(phi, LHP, params)
        
        tau_L = bp['Tau_LR'] * LHP
        tau_H = bp['Tau_HR'] * LHP
        tau_P = bp.get('Tau_PR', 0) * LHP
        
        return {
            'maxwell': tau_maxwell,
            'L': tau_L,
            'H': tau_H,
            'P': tau_P,
        }
    
    def _add_melt_effects(self, phi: np.ndarray, scale: np.ndarray, params: Dict) -> np.ndarray:
        """
        Add melt enhancement effects following Holtzman 2016.
        
        Implements sr_melt_enhancement from MATLAB VBR:
        - a = log(x_phi_c)
        - step = a * erf(phi / phi_c)  
        - slope = alpha * phi
        - ln_SR_phi_enh = slope + step
        - SR_phi_enh = exp(ln_SR_phi_enh)
        
        Parameters
        ----------
        phi : array
            Melt fraction
        scale : array
            Maxwell time scale factor
        params : dict
            Method parameters with melt_alpha, phi_c, x_phi_c
            
        Returns
        -------
        array
            Scale factor adjusted for melt effects
        """
        alpha = params.get('melt_alpha', 25.0)  # post-critical melt fraction dependence (HK2003.diff.alf)
        phi_c = params.get('phi_c', 0.0001)     # Critical melt fraction; Default to small non-zero value
        x_phi_c = params.get('x_phi_c', 1.0)    # melt enhancement factor

        # Apply x_phi_c adjustment first (nominally melt-free to truly melt-free)
        # x_phi_c will be 1 if melt_enhancement is turned off (set to 0)
        scale = scale * x_phi_c
        
        # Calculate melt enhancement following Holtzman 2016 / sr_melt_enhancement.m
        if np.any(phi > 0) and phi_c > 0:
            a = np.log(x_phi_c) if x_phi_c > 0 else 0
            ratefac = 1.0 / phi_c
            step = a * erf(phi * ratefac)
            slope = alpha * phi
            ln_SR_phi_enh = slope + step
            SR_phi_enh = np.exp(ln_SR_phi_enh)
            
            # Divide scale by enhancement factor (enhancement increases strain rate,
            # which decreases maxwell time / integration limits)
            scale = scale / SR_phi_enh
        
        return scale
    
    def _Q_xfit_premelt(self):
        """
        Calculate anelastic properties using Yamauchi & Takei 2016 pre-melting scaling.
        
        Requires solidus temperature in state variables.
        
        When include_direct_melt_effect == 1 (YT2024), all poroelastic effects
        are applied internally to J1, so always use anharmonic Gu.
        When include_direct_melt_effect == 0 (YT2016), use anh_poro Gu if
        available (standard Q_get_state_vars behavior), otherwise anharmonic.
        """
        sv = self.sv
        params = self.input['anelastic']['xfit_premelt']
        
        # Need solidus
        if sv.Tsolidus_K is None:
            sv.Tsolidus_K = calculate_solidus_K(
                sv.P_GPa,
                method=params.get('solidus_method', 'hirschmann'),
                density_model=params.get('density_model', 'constant'),
                density_rho=params.get('density_rho', 3400.0),
                density_file=params.get('density_file'),
            )
        
        # Get unrelaxed modulus - follows MATLAB Q_xfit_premelt.m logic:
        # if include_direct_melt_effect == 1 (YT2024): always use base elastic
        #   (poroelastic effects applied internally to J1)
        # if include_direct_melt_effect == 0 (YT2016): use anh_poro if available
        #   (standard Q_get_state_vars behavior)
        if params['include_direct_melt_effect'] == 1:
            Gu = self._get_base_elastic_output()['Gu']
        else:
            Gu = self._get_unrelaxed_Gu()
        Ju = 1.0 / Gu
        rho = sv.rho
        phi = sv.phi
        
        Tn = sv.T_K / sv.Tsolidus_K
        
        # Calculate viscosity if not done
        if 'xfit_premelt' not in self.output['viscous']:
            self.viscous_methods.append('xfit_premelt')
            self.input['viscous']['xfit_premelt'] = Params_Viscous('xfit_premelt')
            self._visc_xfit_premelt()
        
        eta_diff = self.output['viscous']['xfit_premelt']['diff']['eta']
        tau_m = eta_diff / Gu
        
        # Parameters
        alpha_B = params['alpha_B']
        A_B = params['A_B']
        tau_pp = params['tau_pp']
        include_melt = params['include_direct_melt_effect']
        
        # Calculate A_p and sig_p
        A_p, sig_p = self._calc_Ap_sigp(Tn, phi, params)
        
        if include_melt:
            Beta_B = params['Beta_B'] * phi
            poro_factor = params['poro_Lambda'] * phi
        else:
            Beta_B = np.zeros_like(phi)
            poro_factor = np.zeros_like(phi)
        
        A_B_plus_Beta_B = A_B + Beta_B
        
        # Frequency loop
        n_freq = len(sv.f)
        sv_shape = sv.shape
        n_sv = np.prod(sv_shape)
        
        J1 = np.zeros((n_sv, n_freq))
        J2 = np.zeros((n_sv, n_freq))
        
        Ju_flat = Ju.flatten()
        tau_m_flat = tau_m.flatten()
        A_B_BB_flat = A_B_plus_Beta_B.flatten()
        A_p_flat = A_p.flatten()
        sig_p_flat = sig_p.flatten()
        poro_flat = poro_factor.flatten()
        
        pifac = np.sqrt(2 * np.pi) / 2
        
        for ifreq, f in enumerate(sv.f):
            period = 1.0 / f
            
            for i in range(n_sv):
                tau_m_i = tau_m_flat[i]
                p_p = period / (2 * np.pi * tau_m_i)
                
                ABppa = A_B_BB_flat[i] * p_p**alpha_B
                lntaupp = np.log(tau_pp / p_p)
                sig_p_i = sig_p_flat[i]
                A_p_i = A_p_flat[i]
                
                # J1 calculation (Eq. 13 in YT2016)
                J1[i, ifreq] = Ju_flat[i] * (1 + poro_flat[i] + ABppa / alpha_B +
                                             pifac * A_p_i * sig_p_i * 
                                             (1 - erf(lntaupp / (np.sqrt(2) * sig_p_i))) if sig_p_i > 0 else 0)
                
                # J2 calculation
                peak_term = A_p_i * np.exp(-lntaupp**2 / (2 * sig_p_i**2)) if sig_p_i > 0 else 0
                J2[i, ifreq] = Ju_flat[i] * np.pi / 2 * (ABppa + peak_term) + Ju_flat[i] * p_p
        
        # Reshape
        out_shape = sv_shape + (n_freq,)
        J1 = J1.reshape(out_shape)
        J2 = J2.reshape(out_shape)
        
        # Calculate derived quantities
        rho_f = np.broadcast_to(rho[..., np.newaxis], out_shape)
        V = np.sqrt(1.0 / (J1 * rho_f))
        M = 1.0 / np.sqrt(J1**2 + J2**2)
        Qinv = J2 / J1
        Q = 1.0 / Qinv
        Vave = np.mean(V, axis=-1)
        
        self.output['anelastic']['xfit_premelt'] = {
            'J1': J1,
            'J2': J2,
            'Q': Q,
            'Qinv': Qinv,
            'M': M,
            'V': V,
            'Vave': Vave,
            'units': {
                'J1': '1/Pa', 'J2': '1/Pa', 'Q': '', 'Qinv': '',
                'M': 'Pa', 'V': 'm/s', 'Vave': 'm/s'
            }
        }
    
    def _calc_Ap_sigp(self, Tn: np.ndarray, phi: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Tn-dependent A_p and sig_p coefficients."""
        Ap_pts = params['Ap_Tn_pts']
        sig_pts = params['sig_p_Tn_pts']
        
        include_melt = params['include_direct_melt_effect']
        Beta_p = params['Beta'] if include_melt else 0
        
        A_p = np.zeros_like(Tn)
        A_p[Tn >= Ap_pts[2]] = params['Ap_fac_3'] + Beta_p * phi[Tn >= Ap_pts[2]] # beta_p(phi) not * phi?
        A_p[(Tn >= Ap_pts[1]) & (Tn < Ap_pts[2])] = params['Ap_fac_3']
        
        mask = (Tn >= Ap_pts[0]) & (Tn < Ap_pts[1])
        A_p[mask] = params['Ap_fac_1'] + params['Ap_fac_2'] * (Tn[mask] - Ap_pts[0])
        A_p[Tn < Ap_pts[0]] = params['Ap_fac_1']
        
        sig_p = np.zeros_like(Tn)
        sig_p[Tn < sig_pts[0]] = params['sig_p_fac_1']
        
        mask = (Tn >= sig_pts[0]) & (Tn < sig_pts[1])
        sig_p[mask] = params['sig_p_fac_1'] + params['sig_p_fac_2'] * (Tn[mask] - sig_pts[0])
        sig_p[Tn >= sig_pts[1]] = params['sig_p_fac_3']
        
        return A_p, sig_p
    
    def _Q_andrade_psp(self):
        """
        Calculate Andrade pseudo-period scaling anelastic properties.
        
        References
        ----------
        Jackson and Faul, 2010, PEPI https://doi.org/10.1016/j.pepi.2010.09.005
        Bellis and Holtzman, 2014, JGR http://dx.doi.org/10.1002/2013JB010831
        """
        sv = self.sv
        params = self.input['anelastic']['andrade_psp']
        
        # Get unrelaxed modulus (use anh_poro if available for melt-corrected moduli)
        Gu = self._get_unrelaxed_Gu()
        Ju = 1.0 / Gu
        rho = sv.rho
        
        # Parameters
        n = params['n']
        Beta = params['Beta']
        Tau_MR = params['Tau_MR']
        E = params['E']
        R = params['R']
        TR = params['TR']
        PR = params['PR'] * 1e9
        dR = params['dR']
        Vstar = params['Vstar']
        m = params['m']
        
        T_K = sv.T_K
        P_Pa = sv.P_GPa * 1e9
        d = sv.dg_um
        phi = sv.phi
        
        # Calculate Xtilde - pseudo-period master variable (independent of frequency)
        # Following MATLAB: Xtilde = ((d/dR)^-m) * exp((-E/R)*(1/T - 1/TR)) * exp(-(Vstar/R)*(P/T - PR/TR))
        # NOTE: grain size exponent is NEGATIVE (-m) in MATLAB!
        Xtilde = (d / dR)**(-m) * \
                 np.exp((-E / R) * (1 / T_K - 1 / TR)) * \
                 np.exp((-Vstar / R) * (P_Pa / T_K - PR / TR))
        
        # Apply melt effects to Xtilde (following calculateXtilde in MATLAB)
        alpha = params.get('melt_alpha', 25.0)  # HK2003.diff.alf
        phi_c = params.get('phi_c', 1e-5)
        x_phi_c = params.get('x_phi_c', 1.0)
        
        # When melt_enhancement is off, x_phi_c = 1 but melt effects still apply
        if x_phi_c != 1.0:
            Xtilde = Xtilde / x_phi_c
        
        # Apply melt enhancement: Xtilde_prime = sr_melt_enhancement(phi, alpha, x_phi_c, phi_c)
        # sr_melt_enhancement returns exp(alpha*phi + log(x_phi_c)*erf(phi/phi_c))
        if phi_c > 0:
            a = np.log(x_phi_c) if x_phi_c > 0 else 0
            Xtilde_prime = np.exp(alpha * phi + a * erf(phi / phi_c))
            Xtilde = Xtilde_prime * Xtilde
        
        # Frequency calculations
        n_freq = len(sv.f)
        sv_shape = sv.shape
        n_sv = np.prod(sv_shape)
        
        J1 = np.zeros((n_sv, n_freq))
        J2 = np.zeros((n_sv, n_freq))
        
        Ju_flat = Ju.flatten()
        Xtilde_flat = Xtilde.flatten()
        
        from scipy.special import gamma as gamma_func
        Gamma_n = gamma_func(1 + n)
        
        # Pre-compute constants
        param1 = Beta * Gamma_n * np.cos(n * np.pi / 2)
        param2 = Beta * Gamma_n * np.sin(n * np.pi / 2)
        
        for ifreq, f in enumerate(sv.f):
            T0 = 1.0 / f  # period
            
            for i in range(n_sv):
                # wX_mat = 2*pi / (T0 * Xtilde) - this is the dimensionless frequency
                wX = 2 * np.pi / (T0 * Xtilde_flat[i])
                
                # Andrade compliance (following MATLAB exactly)
                J1[i, ifreq] = Ju_flat[i] * (1 + param1 * wX**(-n))
                J2[i, ifreq] = Ju_flat[i] * (param2 * wX**(-n) + 1 / (Tau_MR * wX))
        
        # Reshape and calculate derived quantities
        out_shape = sv_shape + (n_freq,)
        J1 = J1.reshape(out_shape)
        J2 = J2.reshape(out_shape)
        
        rho_f = np.broadcast_to(rho[..., np.newaxis], out_shape)
        M = 1.0 / np.sqrt(J1**2 + J2**2)
        V = np.sqrt(M / rho_f)
        Qinv = J2 / J1
        Q = 1.0 / Qinv
        Vave = np.mean(V, axis=-1)
        
        self.output['anelastic']['andrade_psp'] = {
            'J1': J1, 'J2': J2, 'Q': Q, 'Qinv': Qinv,
            'M': M, 'V': V, 'Vave': Vave,
        }
    
    @staticmethod
    def _xfit_mxw_xfunc(tau_norm_vec, params):
        """Relaxation spectrum function X(tau_norm) for xfit_mxw.
        
        Matches MATLAB Q_xfit_mxw_xfunc.m exactly.
        """
        Beta = params['beta1'] * np.ones_like(tau_norm_vec)
        Alpha = (params['Alpha_a']
                 - params['Alpha_b'] / (1 + params['Alpha_c'] * tau_norm_vec**params['Alpha_taun']))
        
        mask = tau_norm_vec < params['tau_cutoff']
        Beta[mask] = params['beta2']
        Alpha[mask] = params['alpha2']
        
        X_tau = Beta * tau_norm_vec**Alpha
        return X_tau

    def _Q_xfit_mxw(self):
        """Calculate McCarthy et al. 2011 master curve Maxwell scaling.
        
        Follows MATLAB Q_xfit_mxw.m exactly:
        - J1 via numerical integration of X(tau)/tau over logspace (eq 18 of [1])
        - J2 = Ju * (pi/2 * X(tau_norm) + tau_norm)
        - V = sqrt(1 / (J1 * rho))
        """
        sv = self.sv
        params = self.input['anelastic']['xfit_mxw']
        
        # Get unrelaxed modulus - use anh_poro if available (matches Q_get_state_vars)
        Gu = self._get_unrelaxed_Gu()
        Ju = 1.0 / Gu
        rho = sv.rho
        
        # Need viscosity for Maxwell time
        # MATLAB uses the first viscous method: visc_method=VBR.in.viscous.methods_list{1}
        visc_methods = self.input['viscous'].get('methods_list', [])
        if visc_methods:
            visc_method = visc_methods[0]
        else:
            visc_method = 'xfit_premelt'  # fallback
        
        if visc_method not in self.output['viscous']:
            if visc_method not in self.input['viscous']:
                self.input['viscous'][visc_method] = Params_Viscous(visc_method)
            if visc_method == 'xfit_premelt':
                self._visc_xfit_premelt()
            elif visc_method == 'HK2003':
                self._visc_HK2003()
        
        eta = self.output['viscous'][visc_method]['diff']['eta']
        tau_maxwell = eta / Gu
        
        # Frequency
        f_vec = sv.f
        n_freq = len(f_vec)
        sv_shape = sv.shape
        n_sv = np.prod(sv_shape)
        
        J1 = np.zeros((n_sv, n_freq))
        J2 = np.zeros((n_sv, n_freq))
        
        Ju_flat = Ju.flatten()
        tau_m_flat = tau_maxwell.flatten()
        rho_flat = rho.flatten()
        
        for i in range(n_sv):
            tau_mxw = tau_m_flat[i]
            Ju_i = Ju_flat[i]
            
            for ifreq in range(n_freq):
                freq = f_vec[ifreq]
                f_norm = tau_mxw * freq  # normalized frequency
                max_tau_norm = 1.0 / (2.0 * np.pi * f_norm)  # maximum normalized tau
                
                # Build local logspace grid for integration (matching MATLAB)
                tau_norm_vec_local = np.logspace(-30, np.log10(max_tau_norm), 100)
                
                # Evaluate relaxation spectrum X at all 100 points
                X_tau = self._xfit_mxw_xfunc(tau_norm_vec_local, params)
                
                # J1: numerical integration of X(tau)/tau dtau (eq 18 of [1])
                int1 = np.trapz(X_tau / tau_norm_vec_local, tau_norm_vec_local)
                J1[i, ifreq] = Ju_i * (1.0 + int1)
                
                # J2: pi/2 * X(max_tau_norm) + viscous term (eq 18 of [1])
                tau_norm_ifreq = 1.0 / (2.0 * np.pi * freq * tau_mxw)  # same as max_tau_norm
                J2[i, ifreq] = Ju_i * (np.pi / 2.0 * X_tau[-1] + tau_norm_ifreq)
        
        # Reshape to state variable shape + frequency dimension
        out_shape = sv_shape + (n_freq,)
        J1 = J1.reshape(out_shape)
        J2 = J2.reshape(out_shape)
        
        rho_f = np.broadcast_to(rho[..., np.newaxis], out_shape)
        M = 1.0 / np.sqrt(J1**2 + J2**2)
        V = np.sqrt(1.0 / (J1 * rho_f))  # MATLAB: V = sqrt(1/(J1*rho))
        Qinv = J2 / J1
        Q = 1.0 / Qinv
        Vave = np.mean(V, axis=-1)
        
        self.output['anelastic']['xfit_mxw'] = {
            'J1': J1, 'J2': J2, 'Q': Q, 'Qinv': Qinv,
            'M': M, 'V': V, 'Vave': Vave,
        }
