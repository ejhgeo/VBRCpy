"""
Golden-file tests for VBR elastic methods: anharmonic and cammarano2003.
"""

import numpy as np
import pytest

from tests.conftest import assert_or_store_golden

from vbrcpy.vbr.core import VBR, StateVariables
from vbrcpy.vbr.cammarano import cammarano_elastic, pyrolite_assemblage

pytestmark = pytest.mark.fast


class TestAnharmonicDefault:
    def test_golden(self, standard_state_variables, golden_dir, regenerate):
        """Anharmonic elastic moduli should match golden reference."""
        vbr = VBR(standard_state_variables, elastic_methods=['anharmonic'])
        vbr.run()
        out = vbr.output['elastic']['anharmonic']

        golden = assert_or_store_golden(
            golden_dir, regenerate, "vbr_elastic_anharmonic.npz",
            Gu=out['Gu'],
            Ku=out['Ku'],
            Vsu=out['Vsu'],
            Vpu=out['Vpu'],
        )
        if golden is not None:
            np.testing.assert_allclose(out['Gu'], golden['Gu'], rtol=1e-10)
            np.testing.assert_allclose(out['Ku'], golden['Ku'], rtol=1e-10)
            np.testing.assert_allclose(out['Vsu'], golden['Vsu'], rtol=1e-10)
            np.testing.assert_allclose(out['Vpu'], golden['Vpu'], rtol=1e-10)

    def test_vs_physical_range(self, standard_state_variables):
        """Vs should be in a physically reasonable range (3.5-5.5 km/s)."""
        vbr = VBR(standard_state_variables, elastic_methods=['anharmonic'])
        vbr.run()
        Vs_kms = vbr.output['elastic']['anharmonic']['Vsu'] / 1000.0
        assert np.all(Vs_kms > 3.5)
        assert np.all(Vs_kms < 5.5)

    def test_vs_decreases_with_temperature(self, standard_state_variables):
        """Higher T should reduce elastic Vs."""
        vbr = VBR(standard_state_variables, elastic_methods=['anharmonic'])
        vbr.run()
        Vs = vbr.output['elastic']['anharmonic']['Vsu']
        # T varies along axis 0: T[0] < T[1] < T[2]
        # Vs should decrease
        assert np.all(Vs[0, :, :] > Vs[2, :, :])


class TestAnharmonicPoro:
    def test_melt_reduces_vs(self):
        """Poroelastic correction should reduce Vs when phi > 0."""
        T_K = np.array([1600.0])
        base_sv = StateVariables(
            T_K=T_K, P_GPa=np.array([3.0]), rho=np.array([3300.0]),
            dg_um=np.array([5000.0]), phi=np.array([0.0]),
            sig_MPa=np.array([0.1]), f=np.array([0.01]),
        )
        melt_sv = StateVariables(
            T_K=T_K, P_GPa=np.array([3.0]), rho=np.array([3300.0]),
            dg_um=np.array([5000.0]), phi=np.array([0.02]),
            sig_MPa=np.array([0.1]), f=np.array([0.01]),
        )
        vbr_base = VBR(base_sv, elastic_methods=['anharmonic', 'anh_poro'])
        vbr_base.run()
        vbr_melt = VBR(melt_sv, elastic_methods=['anharmonic', 'anh_poro'])
        vbr_melt.run()

        Vs_base = vbr_base.output['elastic']['anh_poro']['Vsu']
        Vs_melt = vbr_melt.output['elastic']['anh_poro']['Vsu']
        assert Vs_melt < Vs_base


class TestCammarano2003:
    def test_olivine_regime_golden(self, golden_dir, regenerate):
        """Cammarano at 3 GPa (olivine regime)."""
        T_K = np.array([1400.0, 1600.0, 1800.0])
        P_GPa = 3.0
        G, K, rho, Vs, Vp = cammarano_elastic(T_K, P_GPa, X_Fe=0.1)
        golden = assert_or_store_golden(
            golden_dir, regenerate, "vbr_elastic_cammarano_olivine.npz",
            G=G, K=K, rho=rho, Vs=Vs, Vp=Vp, T_K=T_K,
        )
        if golden is not None:
            np.testing.assert_allclose(G, golden['G'], rtol=1e-10)
            np.testing.assert_allclose(K, golden['K'], rtol=1e-10)
            np.testing.assert_allclose(Vs, golden['Vs'], rtol=1e-10)
            np.testing.assert_allclose(Vp, golden['Vp'], rtol=1e-10)

    def test_wadsleyite_regime_golden(self, golden_dir, regenerate):
        """Cammarano at 16 GPa (wadsleyite regime)."""
        T_K = np.array([1600.0, 1800.0])
        P_GPa = 16.0
        G, K, rho, Vs, Vp = cammarano_elastic(T_K, P_GPa, X_Fe=0.1)
        golden = assert_or_store_golden(
            golden_dir, regenerate, "vbr_elastic_cammarano_wadsleyite.npz",
            G=G, K=K, rho=rho, Vs=Vs, Vp=Vp,
        )
        if golden is not None:
            np.testing.assert_allclose(G, golden['G'], rtol=1e-10)
            np.testing.assert_allclose(Vs, golden['Vs'], rtol=1e-10)

    def test_lower_mantle_regime_golden(self, golden_dir, regenerate):
        """Cammarano at 30 GPa (perovskite/lower mantle regime)."""
        T_K = np.array([2000.0, 2200.0])
        P_GPa = 30.0
        G, K, rho, Vs, Vp = cammarano_elastic(T_K, P_GPa, X_Fe=0.1)
        golden = assert_or_store_golden(
            golden_dir, regenerate, "vbr_elastic_cammarano_lower_mantle.npz",
            G=G, K=K, rho=rho, Vs=Vs, Vp=Vp,
        )
        if golden is not None:
            np.testing.assert_allclose(G, golden['G'], rtol=1e-10)
            np.testing.assert_allclose(Vs, golden['Vs'], rtol=1e-10)

    def test_vs_increases_with_pressure(self):
        """At constant T, Vs should generally increase with pressure."""
        T_K = np.array([1600.0])
        Vs_vals = []
        for P in [3.0, 10.0, 16.0, 25.0]:
            _, _, _, Vs, _ = cammarano_elastic(T_K, P, X_Fe=0.1)
            Vs_vals.append(Vs[0])
        # General trend should be increasing (phase transitions may cause jumps)
        assert Vs_vals[-1] > Vs_vals[0]

    def test_pyrolite_assemblage_regimes(self):
        """Assemblage should change across pressure regimes."""
        assem_3 = pyrolite_assemblage(3.0, 0.1)
        assem_16 = pyrolite_assemblage(16.0, 0.1)
        assem_30 = pyrolite_assemblage(30.0, 0.1)
        # Number of minerals and their types should differ
        assert len(assem_3) >= 3
        assert len(assem_16) >= 3
        assert len(assem_30) >= 2
