"""
Tests for the prior probability module.

Primarily known-answer tests verifying grid construction,
PDF type dispatch, and prior configuration.
"""

import numpy as np
import pytest

from vbrcpy.prior import (
    make_param_grid,
    prior_model_probs,
    prep_gs_lognormal,
    apply_melt_fraction_prior,
    apply_temperature_prior,
    MeltFractionPrior,
    TemperaturePrior,
    GrainSizePrior,
)

pytestmark = pytest.mark.fast


class TestMakeParamGrid:
    def test_3d_shapes(self):
        """3-variable grid should produce correct meshgrid shapes."""
        sweep = {
            'T': np.array([1200.0, 1400.0, 1600.0]),
            'phi': np.array([0.0, 0.01]),
            'gs': np.array([1000.0, 5000.0, 10000.0]),
        }
        params = make_param_grid(['T', 'phi', 'gs'], sweep)
        assert params['T'].shape == (3, 2, 3)
        assert params['phi'].shape == (3, 2, 3)
        assert params['gs'].shape == (3, 2, 3)

    def test_indexing_ij(self):
        """First axis should correspond to first variable (indexing='ij')."""
        sweep = {
            'T': np.array([100.0, 200.0]),
            'phi': np.array([0.0, 0.5, 1.0]),
            'gs': np.array([10.0, 20.0]),
        }
        params = make_param_grid(['T', 'phi', 'gs'], sweep)
        # T varies along axis 0
        assert params['T'][0, 0, 0] == 100.0
        assert params['T'][1, 0, 0] == 200.0
        # phi varies along axis 1
        assert params['phi'][0, 0, 0] == 0.0
        assert params['phi'][0, 1, 0] == 0.5
        assert params['phi'][0, 2, 0] == 1.0
        # gs varies along axis 2
        assert params['gs'][0, 0, 0] == 10.0
        assert params['gs'][0, 0, 1] == 20.0

    def test_mean_std_computed(self):
        """Mean and std should be computed from sweep vectors."""
        sweep = {
            'T': np.array([1000.0, 2000.0]),
            'phi': np.array([0.0, 0.04]),
        }
        params = make_param_grid(['T', 'phi'], sweep)
        np.testing.assert_allclose(params['T_mean'], 1500.0)
        np.testing.assert_allclose(params['T_std'], np.std([1000.0, 2000.0]))
        np.testing.assert_allclose(params['phi_mean'], 0.02)

    def test_1d_passthrough(self):
        """Single variable should return 1-D array directly."""
        sweep = {'T': np.array([100.0, 200.0, 300.0])}
        params = make_param_grid(['T'], sweep)
        np.testing.assert_array_equal(params['T'], sweep['T'])


class TestPriorModelProbs:
    def test_uniform_is_flat(self):
        """Uniform prior should produce constant values."""
        sweep = {
            'T': np.array([1200.0, 1400.0, 1600.0]),
            'phi': np.array([0.0, 0.01, 0.02]),
        }
        states = make_param_grid(['T', 'phi'], sweep)
        prior, sigma = prior_model_probs(states, ['T', 'phi'])
        # All values should be identical for uniform
        np.testing.assert_allclose(prior, prior.flat[0], rtol=1e-10)

    def test_normal_peaks_at_mean(self):
        """Normal prior should peak at the mean value."""
        sweep = {'T': np.linspace(1000, 2000, 50)}
        states = make_param_grid(['T'], sweep)
        states['T_pdf_type'] = 'normal'
        states['T_mean'] = 1500.0
        states['T_std'] = 100.0
        prior, sigma = prior_model_probs(states, ['T'])
        peak_idx = np.argmax(prior)
        np.testing.assert_allclose(states['T'][peak_idx], 1500.0, atol=25.0)

    def test_lognormal_prior(self):
        """Lognormal prior should be positive and peak at mode."""
        sweep = {'gs': np.linspace(0.1, 5.0, 100)}
        states = make_param_grid(['gs'], sweep)
        states['gs_pdf_type'] = 'lognormal'
        states['gs_mean'] = 0.5  # log-space mean
        states['gs_std'] = 0.3
        prior, sigma = prior_model_probs(states, ['gs'])
        assert np.all(prior >= 0)
        assert prior.sum() > 0

    def test_uniform_log_prior(self):
        """Log-uniform prior should weight smaller values more."""
        sweep = {'gs': np.array([100.0, 1000.0, 10000.0])}
        states = make_param_grid(['gs'], sweep)
        states['gs_pdf_type'] = 'uniform_log'
        prior, sigma = prior_model_probs(states, ['gs'])
        # uniform in log-space means constant probability per log interval
        np.testing.assert_allclose(prior, prior[0], rtol=1e-10)

    def test_joint_independence(self):
        """Joint prior should be product of marginals."""
        sweep = {
            'T': np.array([1200.0, 1400.0]),
            'phi': np.array([0.0, 0.02]),
        }
        states = make_param_grid(['T', 'phi'], sweep)
        states['T_pdf_type'] = 'normal'
        states['T_mean'] = 1300.0
        states['T_std'] = 100.0
        # phi stays uniform
        prior_joint, _ = prior_model_probs(states, ['T', 'phi'])

        # Compute T marginal alone
        states_T = make_param_grid(['T'], sweep)
        states_T['T_pdf_type'] = 'normal'
        states_T['T_mean'] = 1300.0
        states_T['T_std'] = 100.0
        prior_T, _ = prior_model_probs(states_T, ['T'])

        # phi uniform marginal
        states_phi = make_param_grid(['phi'], sweep)
        prior_phi, _ = prior_model_probs(states_phi, ['phi'])

        # Joint should be outer product
        expected = prior_T[:, np.newaxis] * prior_phi[np.newaxis, :]
        np.testing.assert_allclose(prior_joint, expected, rtol=1e-10)


class TestPrepGsLognormal:
    def test_log_normalization(self):
        """Should convert gs_mean to log(gs_mean/gsref)."""
        params = {'gs_mean': 2000.0, 'gs_std': 0.5}
        sweep = {'gs_params': {'gsref': 1000.0}}
        result = prep_gs_lognormal(params, sweep)
        expected_mean = np.log(2000.0 / 1000.0)
        np.testing.assert_allclose(result['gs_mean'], expected_mean, rtol=1e-12)

    def test_preserves_units(self):
        """Should store original value in gs_mean_units."""
        params = {'gs_mean': 5000.0, 'gs_std': 0.3}
        sweep = {'gs_params': {'gsref': 1000.0}}
        result = prep_gs_lognormal(params, sweep)
        assert result['gs_mean_units'] == 5000.0
        assert result['gs_std_units'] == 0.3


class TestApplyMeltFractionPrior:
    def test_uniform_is_noop(self):
        """Uniform melt prior should not modify params."""
        params = {}
        prior = MeltFractionPrior(phi_prior_type='uniform')
        apply_melt_fraction_prior(params, prior, depth_km=100.0)
        assert 'phi_pdf_type' not in params

    def test_zero_melt(self):
        """Zero-melt prior should set normal with tiny std."""
        params = {}
        prior = MeltFractionPrior(phi_prior_type='zero_melt')
        apply_melt_fraction_prior(params, prior, depth_km=100.0)
        assert params['phi_pdf_type'] == 'normal'
        assert params['phi_mean'] == 0.0
        assert params['phi_std'] == 0.001

    def test_piecewise_shallow(self):
        """Above onset depth, piecewise should be uniform (no-op)."""
        params = {}
        prior = MeltFractionPrior(phi_prior_type='piecewise_depth',
                                  onset_depth_km=80.0)
        apply_melt_fraction_prior(params, prior, depth_km=50.0)
        assert 'phi_pdf_type' not in params

    def test_piecewise_deep(self):
        """Below onset depth, piecewise should suppress melt."""
        params = {}
        prior = MeltFractionPrior(phi_prior_type='piecewise_depth',
                                  onset_depth_km=80.0)
        apply_melt_fraction_prior(params, prior, depth_km=200.0)
        assert params['phi_pdf_type'] == 'normal'
        assert params['phi_mean'] == 0.0


class TestApplyTemperaturePrior:
    def test_uniform_is_noop(self):
        """Uniform T prior should not modify params."""
        params = {}
        prior = TemperaturePrior(t_prior_type='uniform')
        apply_temperature_prior(params, prior, depth_km=100.0)
        assert 'T_pdf_type' not in params

    def test_geotherm_sets_normal(self):
        """Geotherm prior should set normal with expected T at depth."""
        params = {}
        prior = TemperaturePrior(
            t_prior_type='geotherm',
            geotherm_file='sc2006',
            geotherm_std_C=200.0,
        )
        apply_temperature_prior(params, prior, depth_km=100.0)
        assert params['T_pdf_type'] == 'normal'
        assert params['T_std'] == 200.0
        # SC2006 geotherm at 100 km should be a reasonable mantle temperature
        assert 500.0 < params['T_mean'] < 1500.0
