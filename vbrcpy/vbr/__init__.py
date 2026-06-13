"""
VBR - Very Broadband Rheology

Python translation of the MATLAB VBR calculator for calculating
material properties (velocity, attenuation) from thermodynamic state variables.

This module provides:
- VBR class for running calculations
- Parameter classes for elastic, anelastic, and viscous methods
- Thermal property functions (solidus calculations)
- Parameter sweep generation for Bayesian inversion

References
----------
Jackson & Faul, 2010, Phys. Earth Planet. Inter.
Yamauchi & Takei, 2016, J. Geophys. Res.
McCarthy, Takei, Hiraga, 2011, JGR

Usage
-----
For sweep generation, import directly from the submodule:
    from vbrcpy.vbr.generate_sweep import generate_parameter_sweep, save_sweep, load_sweep
Or use the CLI:
    python -m vbrcpy.vbr.generate_sweep --config sweep_config.yaml
"""

from .core import VBR
from .params import (
    Params_Elastic,
    Params_Anelastic,
    Params_Viscous,
    Params_Global,
)
from .thermal import solidus

# Note: generate_parameter_sweep, save_sweep, load_sweep are not imported here to avoid
# warnings when running as `python -m vbrcpy.vbr.generate_sweep`.
# Import them directly: from vbrcpy.vbr.generate_sweep import generate_parameter_sweep

__all__ = [
    'VBR',
    'Params_Elastic',
    'Params_Anelastic', 
    'Params_Viscous',
    'Params_Global',
    'solidus',
]
