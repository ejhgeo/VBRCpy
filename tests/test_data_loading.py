"""
Tests for data loading utilities: CSV, Earth model, format detection.
"""

import numpy as np
import pytest
from pathlib import Path

from vbrcpy.data_processing import (
    load_seismic_model_from_csv,
    load_seismic_model_from_earth_model,
    load_seismic_model_universal,
    SeismicModelData,
)

pytestmark = pytest.mark.fast

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestLoadCSV:
    def test_loads_fixture(self):
        """Should load tiny_obs.csv and return correct values."""
        data = load_seismic_model_from_csv(
            str(FIXTURES_DIR / "tiny_obs.csv"),
            default_vs_error=0.1,
            default_q_error=15.0,
        )
        assert len(data) == 3
        assert data.has_vs()
        assert data.has_q()

    def test_location_values(self):
        """Loaded locations should match CSV content."""
        data = load_seismic_model_from_csv(
            str(FIXTURES_DIR / "tiny_obs.csv"),
            default_vs_error=0.1,
            default_q_error=15.0,
        )
        # First row: lon=-120, lat=35
        lats = [loc[0] for loc in data.locations]
        lons = [loc[1] for loc in data.locations]
        assert 35.0 in lats
        assert -120.0 in lons

    def test_vs_values(self):
        """Loaded Vs values should match CSV."""
        data = load_seismic_model_from_csv(
            str(FIXTURES_DIR / "tiny_obs.csv"),
            default_vs_error=0.1,
            default_q_error=15.0,
        )
        np.testing.assert_allclose(data.Vs[0], 4.3, rtol=1e-10)
        np.testing.assert_allclose(data.Vs[1], 4.2, rtol=1e-10)
        np.testing.assert_allclose(data.Vs[2], 4.1, rtol=1e-10)

    def test_uses_file_errors(self):
        """Should use error columns from file when present."""
        data = load_seismic_model_from_csv(
            str(FIXTURES_DIR / "tiny_obs.csv"),
            default_vs_error=999.0,  # should NOT be used
            default_q_error=999.0,
        )
        # File has Vs_error=0.05, Q_error=10
        np.testing.assert_allclose(data.Vs_error[0], 0.05, rtol=1e-10)
        np.testing.assert_allclose(data.Q_error[0], 10.0, rtol=1e-10)


class TestLoadEarthModel:
    def test_prem(self):
        """Should load PREM as SeismicModelData."""
        data = load_seismic_model_from_earth_model('prem')
        assert isinstance(data, SeismicModelData)
        assert data.has_vs()
        assert len(data) > 10

    def test_stw105(self):
        """Should load STW105 as SeismicModelData."""
        data = load_seismic_model_from_earth_model('stw105')
        assert isinstance(data, SeismicModelData)
        assert data.has_vs()

    def test_prem_vs_reasonable(self):
        """PREM Vs values should be in km/s range."""
        data = load_seismic_model_from_earth_model('prem')
        # Vs in km/s (upper mantle ~4-5 km/s, lower mantle up to ~7)
        assert np.all(data.Vs >= 0)
        assert np.all(data.Vs < 8.0)

    def test_depth_filter(self):
        """z_range filter should limit depths."""
        data = load_seismic_model_from_earth_model(
            'prem', z_range=(100, 400)
        )
        for zr in data.z_ranges:
            mid = (zr[0] + zr[1]) / 2
            assert 50 <= mid <= 450  # some tolerance for bin edges


class TestUniversalLoader:
    def test_csv_dispatch(self):
        """Universal loader should detect CSV and dispatch."""
        data = load_seismic_model_universal(
            str(FIXTURES_DIR / "tiny_obs.csv"),
            default_vs_error=0.1,
            default_q_error=15.0,
        )
        assert isinstance(data, SeismicModelData)
        assert len(data) == 3

    def test_earth_model_dispatch(self):
        """Universal loader should detect built-in model names."""
        data = load_seismic_model_universal(
            'prem',
            default_vs_error=0.1,
            default_q_error=50.0,
        )
        assert isinstance(data, SeismicModelData)
        assert len(data) > 10
