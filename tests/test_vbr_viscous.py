"""
Golden-file tests for VBR viscous methods: HK2003 and xfit_premelt.
"""

import numpy as np
import pytest

from tests.conftest import assert_or_store_golden

from vbrcpy.vbr.core import VBR, StateVariables

pytestmark = pytest.mark.fast


class TestHK2003Dry:
    def test_golden(self, standard_state_variables, golden_dir, regenerate):
        """HK2003 viscosity should match golden reference (dry)."""
        vbr = VBR(standard_state_variables, viscous_methods=['HK2003'])
        vbr.run()
        out = vbr.output['viscous']['HK2003']

        golden = assert_or_store_golden(
            golden_dir, regenerate, "vbr_viscous_HK2003.npz",
            eta_total=out['eta_total'],
            sr_tot=out['sr_tot'],
            eta_diff=out['diff']['eta'],
            eta_disl=out['disl']['eta'],
            eta_gbs=out['gbs']['eta'],
        )
        if golden is not None:
            np.testing.assert_allclose(
                out['eta_total'], golden['eta_total'], rtol=1e-10
            )
            np.testing.assert_allclose(
                out['sr_tot'], golden['sr_tot'], rtol=1e-10
            )
            np.testing.assert_allclose(
                out['diff']['eta'], golden['eta_diff'], rtol=1e-10
            )
            np.testing.assert_allclose(
                out['disl']['eta'], golden['eta_disl'], rtol=1e-10
            )
            np.testing.assert_allclose(
                out['gbs']['eta'], golden['eta_gbs'], rtol=1e-10
            )

    def test_mechanisms_present(self, standard_state_variables):
        """All three mechanisms should be computed."""
        vbr = VBR(standard_state_variables, viscous_methods=['HK2003'])
        vbr.run()
        out = vbr.output['viscous']['HK2003']
        assert 'diff' in out
        assert 'disl' in out
        assert 'gbs' in out
        assert 'eta_total' in out

    def test_eta_total_physical_range(self, standard_state_variables):
        """Total viscosity should be in 1e17-1e25 Pa.s range."""
        vbr = VBR(standard_state_variables, viscous_methods=['HK2003'])
        vbr.run()
        eta = vbr.output['viscous']['HK2003']['eta_total']
        assert np.all(eta > 1e17)
        assert np.all(eta < 1e25)


class TestHK2003Wet:
    def test_water_reduces_viscosity(self):
        """Adding water (Ch2o > 0) should reduce viscosity."""
        shape = (2,)
        sv_dry = StateVariables(
            T_K=np.full(shape, 1600.0),
            P_GPa=np.full(shape, 3.0),
            rho=np.full(shape, 3300.0),
            dg_um=np.full(shape, 5000.0),
            phi=np.full(shape, 0.0),
            sig_MPa=np.full(shape, 0.1),
            f=np.array([0.01]),
            Ch2o=np.full(shape, 0.0),
        )
        sv_wet = StateVariables(
            T_K=np.full(shape, 1600.0),
            P_GPa=np.full(shape, 3.0),
            rho=np.full(shape, 3300.0),
            dg_um=np.full(shape, 5000.0),
            phi=np.full(shape, 0.0),
            sig_MPa=np.full(shape, 0.1),
            f=np.array([0.01]),
            Ch2o=np.full(shape, 1000.0),  # 1000 ppm water
        )
        vbr_dry = VBR(sv_dry, viscous_methods=['HK2003'])
        vbr_dry.run()
        vbr_wet = VBR(sv_wet, viscous_methods=['HK2003'])
        vbr_wet.run()

        eta_dry = vbr_dry.output['viscous']['HK2003']['eta_total']
        eta_wet = vbr_wet.output['viscous']['HK2003']['eta_total']
        assert np.all(eta_wet < eta_dry)


class TestHK2003Temperature:
    def test_viscosity_decreases_with_temperature(self, standard_state_variables):
        """Viscosity should decrease with increasing temperature."""
        vbr = VBR(standard_state_variables, viscous_methods=['HK2003'])
        vbr.run()
        eta = vbr.output['viscous']['HK2003']['eta_total']
        # T varies along axis 0 (increases): eta should decrease
        assert np.all(eta[0, :, :] > eta[2, :, :])


class TestXfitPremeltViscous:
    def test_golden(self, standard_state_variables, golden_dir, regenerate):
        """xfit_premelt viscosity should match golden reference."""
        vbr = VBR(standard_state_variables, viscous_methods=['xfit_premelt'])
        vbr.run()
        out = vbr.output['viscous']['xfit_premelt']

        golden = assert_or_store_golden(
            golden_dir, regenerate, "vbr_viscous_xfit_premelt.npz",
            eta=out['diff']['eta'],
            eta_meltfree=out['diff']['eta_meltfree'],
        )
        if golden is not None:
            np.testing.assert_allclose(
                out['diff']['eta'], golden['eta'], rtol=1e-10
            )
            np.testing.assert_allclose(
                out['diff']['eta_meltfree'], golden['eta_meltfree'], rtol=1e-10
            )

    def test_melt_reduces_viscosity(self):
        """Non-zero phi above solidus should reduce viscosity vs melt-free."""
        # T must exceed solidus for melt effect (A_n != 1)
        # Solidus at 1 GPa is ~1400 C → ~1673 K; use T well above
        sv = StateVariables(
            T_K=np.array([1900.0]),
            P_GPa=np.array([1.0]),
            rho=np.array([3300.0]),
            dg_um=np.array([5000.0]),
            phi=np.array([0.02]),
            sig_MPa=np.array([0.1]),
            f=np.array([0.01]),
        )
        vbr = VBR(sv, viscous_methods=['xfit_premelt'])
        vbr.run()
        out = vbr.output['viscous']['xfit_premelt']
        assert out['diff']['eta'][0] < out['diff']['eta_meltfree'][0]
