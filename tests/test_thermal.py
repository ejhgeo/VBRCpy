"""
Golden-file tests for the thermal module: solidus, volatile depressions,
and Earth model loading.
"""

import numpy as np
import pytest

from tests.conftest import assert_or_store_golden

from vbrcpy.vbr.thermal import (
    _solidus_hirschmann,
    _solidus_katz,
    _solidus_yk2001,
    _depression_katz,
    _depression_dasgupta,
    _load_earth_model,
    calculate_solidus_K,
    load_geotherm,
)

pytestmark = pytest.mark.fast

# Pressures spanning upper mantle (avoid extrapolation warnings)
P_TEST = np.array([1.0, 3.0, 5.0, 8.0, 10.0])


class TestSolidusHirschmann:
    def test_golden(self, golden_dir, regenerate):
        result = _solidus_hirschmann(P_TEST)
        golden = assert_or_store_golden(
            golden_dir, regenerate, "thermal_hirschmann.npz",
            Tsol_dry=result['Tsol_dry'],
            P_GPa=P_TEST,
        )
        if golden is not None:
            np.testing.assert_allclose(
                result['Tsol_dry'], golden['Tsol_dry'], rtol=1e-10
            )

    def test_increases_with_pressure(self):
        """Solidus should increase with pressure in valid range."""
        result = _solidus_hirschmann(P_TEST)
        assert np.all(np.diff(result['Tsol_dry']) > 0)

    def test_known_value_at_1atm(self):
        """At P=0, Tsol should equal A1 (1108.08 C)."""
        result = _solidus_hirschmann(np.array([0.0]))
        np.testing.assert_allclose(result['Tsol_dry'][0], 1108.08, rtol=1e-10)


class TestSolidusKatz:
    def test_golden(self, golden_dir, regenerate):
        result = _solidus_katz(P_TEST)
        golden = assert_or_store_golden(
            golden_dir, regenerate, "thermal_katz.npz",
            Tsol_dry=result['Tsol_dry'],
            Tlherz_dry=result['Tlherz_dry'],
            Tliq_dry=result['Tliq_dry'],
            P_GPa=P_TEST,
        )
        if golden is not None:
            np.testing.assert_allclose(
                result['Tsol_dry'], golden['Tsol_dry'], rtol=1e-10
            )
            np.testing.assert_allclose(
                result['Tlherz_dry'], golden['Tlherz_dry'], rtol=1e-10
            )
            np.testing.assert_allclose(
                result['Tliq_dry'], golden['Tliq_dry'], rtol=1e-10
            )

    def test_solidus_below_liquidus(self):
        """Solidus < lherzolite liquidus < liquidus at all pressures."""
        result = _solidus_katz(P_TEST)
        assert np.all(result['Tsol_dry'] < result['Tlherz_dry'])
        assert np.all(result['Tlherz_dry'] < result['Tliq_dry'])


class TestSolidusYK2001:
    def test_golden(self, golden_dir, regenerate):
        depth_km = np.array([100.0, 200.0, 400.0, 600.0])
        result = _solidus_yk2001(
            P_GPa=np.array([3.3, 6.6, 13.2, 19.8]),
            depth_km=depth_km,
        )
        golden = assert_or_store_golden(
            golden_dir, regenerate, "thermal_yk2001.npz",
            Tsol_dry=result['Tsol_dry'],
            depth_km=depth_km,
        )
        if golden is not None:
            np.testing.assert_allclose(
                result['Tsol_dry'], golden['Tsol_dry'], rtol=1e-10
            )

    def test_upper_mantle_formula(self):
        """Verify the upper mantle (z < 660 km) piecewise formula."""
        z = np.array([100.0])
        result = _solidus_yk2001(P_GPa=np.array([3.3]), depth_km=z)
        # Ts_K = 2100 + 1.4848*z - 5e-4*z^2, then converted to C via -273
        expected_K = 2100.0 + 1.4848 * 100.0 - 5e-4 * 100.0**2
        expected_C = expected_K - 273.0  # C2K = 273
        np.testing.assert_allclose(result['Tsol_dry'][0], expected_C, rtol=1e-6)


class TestVolatileDepressions:
    def test_h2o_depression_golden(self, golden_dir, regenerate):
        H2O = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        P = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        dT, dT_dH2O = _depression_katz(H2O, P)
        golden = assert_or_store_golden(
            golden_dir, regenerate, "thermal_depression_h2o.npz",
            dT=dT, dT_dH2O=dT_dH2O,
        )
        if golden is not None:
            np.testing.assert_allclose(dT, golden['dT'], rtol=1e-10)
            np.testing.assert_allclose(dT_dH2O, golden['dT_dH2O'], rtol=1e-10)

    def test_h2o_zero_gives_zero(self):
        """No water should produce zero depression."""
        dT, _ = _depression_katz(np.array([0.0]), np.array([3.0]))
        np.testing.assert_allclose(dT[0], 0.0, atol=1e-15)

    def test_co2_depression_golden(self, golden_dir, regenerate):
        CO2 = np.array([0.0, 5.0, 15.0, 25.0, 30.0, 40.0])
        dT = _depression_dasgupta(CO2)
        golden = assert_or_store_golden(
            golden_dir, regenerate, "thermal_depression_co2.npz",
            dT=dT, CO2=CO2,
        )
        if golden is not None:
            np.testing.assert_allclose(dT, golden['dT'], rtol=1e-10)

    def test_co2_zero_gives_zero(self):
        """No CO2 should produce zero depression."""
        dT = _depression_dasgupta(np.array([0.0]))
        np.testing.assert_allclose(dT[0], 0.0, atol=1e-10)


class TestEarthModels:
    def test_prem_loads(self):
        """PREM should load and return depth + density arrays."""
        depth_m, density = _load_earth_model('prem')
        assert len(depth_m) > 50
        assert depth_m[0] >= 0
        assert np.all(density > 1000)  # includes ocean/crust layers

    def test_stw105_loads(self):
        """STW105 should load."""
        depth_m, density = _load_earth_model('stw105')
        assert len(depth_m) > 100

    def test_prem_with_fields(self):
        """PREM should return Vs and Qmu when requested."""
        depth_m, density, Vs, Qmu = _load_earth_model('prem', fields=['Vs', 'Qmu'])
        assert len(Vs) == len(depth_m)
        assert np.all(Vs >= 0)
        assert np.all(Qmu >= 0)

    def test_prem_density_at_100km(self):
        """PREM density near 100 km should be ~3.4 g/cm3."""
        depth_m, density = _load_earth_model('prem')
        idx_100km = np.argmin(np.abs(depth_m - 100e3))
        assert 3200 < density[idx_100km] < 3600


class TestCalculateSolidusK:
    def test_returns_kelvin(self):
        """Output should be in Kelvin (> 273)."""
        Tsol_K = calculate_solidus_K(np.array([3.0]), method='hirschmann')
        assert Tsol_K[0] > 1000  # reasonable mantle solidus

    def test_methods_consistent(self):
        """All methods should give solidus in same ballpark at 3 GPa."""
        P = np.array([3.0])
        T_hirsch = calculate_solidus_K(P, method='hirschmann')[0]
        T_katz = calculate_solidus_K(P, method='katz')[0]
        T_yk = calculate_solidus_K(P, method='yk2001', depth_km=np.array([100.0]))[0]
        # All should be between 1500-2500 K
        for T in [T_hirsch, T_katz, T_yk]:
            assert 1500 < T < 2500


class TestGeotherm:
    def test_sc2006_loads(self):
        """Built-in SC2006 geotherm should load and interpolate."""
        depths_km, T_C = load_geotherm('sc2006', depth_km=np.array([100.0, 200.0]))
        assert len(T_C) == 2
        assert T_C[0] < T_C[1]  # temperature increases with depth
        assert 500 < T_C[0] < 1500
