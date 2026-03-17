"""
Validation tests for the VBR Bayesian inversion package.

Two validation cases:
1. **PREM validation**: Invert PREM reference velocities through a sweep
   to recover a temperature profile. Compare with the solidus.
2. **Round-trip validation**: Start from a known adiabat, melt fraction,
   and grain size profile → forward-model Vs and Q → invert → compare
   recovered parameters against the known inputs.
"""
