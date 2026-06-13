"""
Golden-file tests for VBR anelastic methods:
eburgers_psp (FastBurger + PointWise), andrade_psp, xfit_premelt, xfit_mxw.
"""

import numpy as np
import pytest

from tests.conftest import assert_or_store_golden

from vbrcpy.vbr.core import VBR, StateVariables

pytestmark = pytest.mark.fast


class TestEburgersFastBurger:
    def test_golden(self, standard_state_variables, golden_dir, regenerate):
        """eburgers_psp (FastBurger) V, Q, J1, J2 should match golden."""
        vbr = VBR(
            standard_state_variables,
            anelastic_methods=['eburgers_psp'],
            viscous_methods=['HK2003'],
        )
        vbr.run()
        out = vbr.output['anelastic']['eburgers_psp']

        golden = assert_or_store_golden(
            golden_dir, regenerate, "vbr_anelastic_eburgers_fastburger.npz",
            V=out['V'], Q=out['Q'], J1=out['J1'], J2=out['J2'],
        )
        if golden is not None:
            np.testing.assert_allclose(out['V'], golden['V'], rtol=1e-10)
            np.testing.assert_allclose(out['Q'], golden['Q'], rtol=1e-8)
            np.testing.assert_allclose(out['J1'], golden['J1'], rtol=1e-8)
            np.testing.assert_allclose(out['J2'], golden['J2'], rtol=1e-8)

    def test_q_physical_range(self, standard_state_variables):
        """Q should be positive and in reasonable range (10-10000)."""
        vbr = VBR(
            standard_state_variables,
            anelastic_methods=['eburgers_psp'],
            viscous_methods=['HK2003'],
        )
        vbr.run()
        Q = vbr.output['anelastic']['eburgers_psp']['Q']
        assert np.all(Q > 0)
        assert np.all(Q < 1e5)


class TestEburgersPointWise:
    def test_golden(self, golden_dir, regenerate):
        """eburgers_psp (PointWise) on tiny grid should match golden."""
        # Small grid to keep PointWise fast (2x1x1, 2 freqs)
        sv = StateVariables(
            T_K=np.array([[[1500.0]], [[1700.0]]]),
            P_GPa=np.full((2, 1, 1), 3.0),
            rho=np.full((2, 1, 1), 3300.0),
            dg_um=np.full((2, 1, 1), 5000.0),
            phi=np.full((2, 1, 1), 0.0),
            sig_MPa=np.full((2, 1, 1), 0.1),
            f=np.logspace(-2.0, -1.5, 2),
        )
        vbr = VBR(sv, anelastic_methods=['eburgers_psp'], viscous_methods=['HK2003'])
        vbr.input['anelastic']['eburgers_psp']['method'] = 'PointWise'
        vbr.run()
        out = vbr.output['anelastic']['eburgers_psp']

        golden = assert_or_store_golden(
            golden_dir, regenerate, "vbr_anelastic_eburgers_pointwise.npz",
            V=out['V'], Q=out['Q'],
        )
        if golden is not None:
            np.testing.assert_allclose(out['V'], golden['V'], rtol=1e-10)
            np.testing.assert_allclose(out['Q'], golden['Q'], rtol=1e-8)


class TestAndradePsp:
    def test_golden(self, standard_state_variables, golden_dir, regenerate):
        """andrade_psp V, Q should match golden."""
        vbr = VBR(
            standard_state_variables,
            anelastic_methods=['andrade_psp'],
            viscous_methods=['HK2003'],
        )
        vbr.run()
        out = vbr.output['anelastic']['andrade_psp']

        golden = assert_or_store_golden(
            golden_dir, regenerate, "vbr_anelastic_andrade.npz",
            V=out['V'], Q=out['Q'], J1=out['J1'], J2=out['J2'],
        )
        if golden is not None:
            np.testing.assert_allclose(out['V'], golden['V'], rtol=1e-10)
            np.testing.assert_allclose(out['Q'], golden['Q'], rtol=1e-8)


class TestXfitPremelt:
    def test_golden(self, standard_state_variables, golden_dir, regenerate):
        """xfit_premelt V, Q should match golden."""
        vbr = VBR(
            standard_state_variables,
            anelastic_methods=['xfit_premelt'],
        )
        vbr.run()
        out = vbr.output['anelastic']['xfit_premelt']

        golden = assert_or_store_golden(
            golden_dir, regenerate, "vbr_anelastic_xfit_premelt.npz",
            V=out['V'], Q=out['Q'], J1=out['J1'], J2=out['J2'],
        )
        if golden is not None:
            np.testing.assert_allclose(out['V'], golden['V'], rtol=1e-10)
            np.testing.assert_allclose(out['Q'], golden['Q'], rtol=1e-8)


class TestXfitMxw:
    def test_golden(self, standard_state_variables, golden_dir, regenerate):
        """xfit_mxw V, Q should match golden (requires viscosity first)."""
        vbr = VBR(
            standard_state_variables,
            anelastic_methods=['xfit_mxw'],
            viscous_methods=['HK2003'],
        )
        vbr.run()
        out = vbr.output['anelastic']['xfit_mxw']

        golden = assert_or_store_golden(
            golden_dir, regenerate, "vbr_anelastic_xfit_mxw.npz",
            V=out['V'], Q=out['Q'], J1=out['J1'], J2=out['J2'],
        )
        if golden is not None:
            np.testing.assert_allclose(out['V'], golden['V'], rtol=1e-10)
            np.testing.assert_allclose(out['Q'], golden['Q'], rtol=1e-8)


class TestAnelasticPhysics:
    def test_melt_reduces_q(self):
        """Non-zero phi should reduce Q via poroelastic modulus reduction."""
        sv_dry = StateVariables(
            T_K=np.array([1600.0]),
            P_GPa=np.array([3.0]),
            rho=np.array([3300.0]),
            dg_um=np.array([5000.0]),
            phi=np.array([0.0]),
            sig_MPa=np.array([0.1]),
            f=np.array([0.01]),
        )
        sv_melt = StateVariables(
            T_K=np.array([1600.0]),
            P_GPa=np.array([3.0]),
            rho=np.array([3300.0]),
            dg_um=np.array([5000.0]),
            phi=np.array([0.02]),
            sig_MPa=np.array([0.1]),
            f=np.array([0.01]),
        )
        # eburgers with anh_poro shows clear melt effect on Q
        vbr_dry = VBR(
            sv_dry,
            elastic_methods=['anharmonic', 'anh_poro'],
            anelastic_methods=['eburgers_psp'],
            viscous_methods=['HK2003'],
        )
        vbr_dry.run()
        vbr_melt = VBR(
            sv_melt,
            elastic_methods=['anharmonic', 'anh_poro'],
            anelastic_methods=['eburgers_psp'],
            viscous_methods=['HK2003'],
        )
        vbr_melt.run()

        Q_dry = vbr_dry.output['anelastic']['eburgers_psp']['Q']
        Q_melt = vbr_melt.output['anelastic']['eburgers_psp']['Q']
        assert np.all(Q_melt < Q_dry)

    def test_higher_temp_reduces_q(self, standard_state_variables):
        """Higher temperature should reduce Q."""
        vbr = VBR(
            standard_state_variables,
            anelastic_methods=['eburgers_psp'],
            viscous_methods=['HK2003'],
        )
        vbr.run()
        Q = vbr.output['anelastic']['eburgers_psp']['Q']
        # Average Q over frequencies; T increases along axis 0
        Q_avg = Q.mean(axis=-1)
        # Higher T → lower Q at same phi, gs
        assert np.all(Q_avg[0, :, :] > Q_avg[2, :, :])
