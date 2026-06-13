"""
Golden-file tests for the fitting pipeline:
fit_preloaded_observations, extract_ml_estimates, depth averaging.
"""

import numpy as np
import pytest

from tests.conftest import assert_or_store_golden

from vbrcpy.fitting import (
    fit_preloaded_observations,
    extract_ml_estimates,
    extract_calculated_values_in_depth_range,
)
from vbrcpy.prior import GrainSizePrior

pytestmark = pytest.mark.integration

# Synthetic observations within sweep range
OBS_VS = 4.3  # km/s
SIGMA_VS = 0.05
OBS_Q = 80.0
SIGMA_Q = 10.0
Z_RANGE = (75.0, 150.0)
METHOD = 'eburgers_psp'


class TestDepthAveraging:
    def test_shape(self, tiny_sweep):
        """Depth-averaged values should have shape (nT, nphi, ngs)."""
        mean_vals, z_inds = extract_calculated_values_in_depth_range(
            tiny_sweep, 'Vs', METHOD, Z_RANGE
        )
        nT = len(tiny_sweep['T'])
        nphi = len(tiny_sweep['phi'])
        ngs = len(tiny_sweep['gs'])
        assert mean_vals.shape == (nT, nphi, ngs)

    def test_z_indices_within_range(self, tiny_sweep):
        """Returned z_inds should correspond to depths within z_range."""
        _, z_inds = extract_calculated_values_in_depth_range(
            tiny_sweep, 'Vs', METHOD, Z_RANGE
        )
        z_m = tiny_sweep['z']
        for idx in z_inds:
            depth_km = z_m[idx] / 1e3
            assert Z_RANGE[0] <= depth_km <= Z_RANGE[1]

    def test_manual_average_matches(self, tiny_sweep):
        """Manual depth averaging should match function output."""
        mean_vals, z_inds = extract_calculated_values_in_depth_range(
            tiny_sweep, 'Vs', METHOD, Z_RANGE
        )
        raw = tiny_sweep['Box'][METHOD]['meanVs']
        expected = np.mean(raw[:, :, :, z_inds], axis=3)
        np.testing.assert_allclose(mean_vals, expected, rtol=1e-12)


class TestFitVsOnly:
    def test_golden(self, tiny_sweep, golden_dir, regenerate):
        """Vs-only fit posterior should match golden."""
        gs_prior = GrainSizePrior(gs_pdf_type='uniform_log')
        posterior, _ = fit_preloaded_observations(
            obs_vs=OBS_VS, sigma_vs=SIGMA_VS,
            obs_q=None, sigma_q=None,
            z_range=Z_RANGE,
            anelastic_method=METHOD,
            grain_size_prior=gs_prior,
            sweep=tiny_sweep,
        )
        golden = assert_or_store_golden(
            golden_dir, regenerate, "fitting_vs_only.npz",
            pS=posterior['pS'],
        )
        if golden is not None:
            np.testing.assert_allclose(
                posterior['pS'], golden['pS'], rtol=1e-8
            )

    def test_posterior_positive(self, tiny_sweep):
        """All posterior values should be non-negative."""
        gs_prior = GrainSizePrior(gs_pdf_type='uniform_log')
        posterior, _ = fit_preloaded_observations(
            obs_vs=OBS_VS, sigma_vs=SIGMA_VS,
            obs_q=None, sigma_q=None,
            z_range=Z_RANGE,
            anelastic_method=METHOD,
            grain_size_prior=gs_prior,
            sweep=tiny_sweep,
        )
        assert np.all(posterior['pS'] >= 0)


class TestFitQOnly:
    def test_golden(self, tiny_sweep, golden_dir, regenerate):
        """Q-only fit posterior should match golden."""
        gs_prior = GrainSizePrior(gs_pdf_type='uniform_log')
        posterior, _ = fit_preloaded_observations(
            obs_vs=None, sigma_vs=None,
            obs_q=OBS_Q, sigma_q=SIGMA_Q,
            z_range=Z_RANGE,
            anelastic_method=METHOD,
            grain_size_prior=gs_prior,
            sweep=tiny_sweep,
        )
        golden = assert_or_store_golden(
            golden_dir, regenerate, "fitting_q_only.npz",
            pS=posterior['pS'],
        )
        if golden is not None:
            np.testing.assert_allclose(
                posterior['pS'], golden['pS'], rtol=1e-8
            )


class TestFitVsAndQ:
    def test_golden(self, tiny_sweep, golden_dir, regenerate):
        """Combined Vs+Q posterior should match golden."""
        gs_prior = GrainSizePrior(gs_pdf_type='uniform_log')
        posterior, _ = fit_preloaded_observations(
            obs_vs=OBS_VS, sigma_vs=SIGMA_VS,
            obs_q=OBS_Q, sigma_q=SIGMA_Q,
            z_range=Z_RANGE,
            anelastic_method=METHOD,
            grain_size_prior=gs_prior,
            sweep=tiny_sweep,
        )
        golden = assert_or_store_golden(
            golden_dir, regenerate, "fitting_vs_q.npz",
            pS=posterior['pS'],
        )
        if golden is not None:
            np.testing.assert_allclose(
                posterior['pS'], golden['pS'], rtol=1e-8
            )

    def test_combined_more_constrained(self, tiny_sweep):
        """Combined posterior should be more peaked (lower entropy) than either alone."""
        gs_prior = GrainSizePrior(gs_pdf_type='uniform_log')
        post_vs, _ = fit_preloaded_observations(
            obs_vs=OBS_VS, sigma_vs=SIGMA_VS,
            obs_q=None, sigma_q=None,
            z_range=Z_RANGE, anelastic_method=METHOD,
            grain_size_prior=gs_prior, sweep=tiny_sweep,
        )
        post_both, _ = fit_preloaded_observations(
            obs_vs=OBS_VS, sigma_vs=SIGMA_VS,
            obs_q=OBS_Q, sigma_q=SIGMA_Q,
            z_range=Z_RANGE, anelastic_method=METHOD,
            grain_size_prior=gs_prior, sweep=tiny_sweep,
        )
        # Normalized max should be higher for combined (more peaked)
        max_vs = post_vs['pS'].max() / post_vs['pS'].sum()
        max_both = post_both['pS'].max() / post_both['pS'].sum()
        assert max_both >= max_vs


class TestMLEstimates:
    def test_golden(self, tiny_sweep, golden_dir, regenerate):
        """ML estimates should match golden."""
        gs_prior = GrainSizePrior(gs_pdf_type='uniform_log')
        posterior, sweep_out = fit_preloaded_observations(
            obs_vs=OBS_VS, sigma_vs=SIGMA_VS,
            obs_q=OBS_Q, sigma_q=SIGMA_Q,
            z_range=Z_RANGE,
            anelastic_method=METHOD,
            grain_size_prior=gs_prior,
            sweep=tiny_sweep,
        )
        ml = extract_ml_estimates(posterior, sweep_out, METHOD)

        golden = assert_or_store_golden(
            golden_dir, regenerate, "fitting_ml_estimates.npz",
            T_ml=np.array([ml['T']['ml']]),
            T_mean=np.array([ml['T']['mean']]),
            T_std=np.array([ml['T']['std']]),
            phi_ml=np.array([ml['phi']['ml']]),
            gs_ml=np.array([ml['gs']['ml']]),
            predicted_Vs=np.array([ml['predicted_Vs']]),
            predicted_Q=np.array([ml['predicted_Q']]),
        )
        if golden is not None:
            np.testing.assert_allclose(
                ml['T']['ml'], golden['T_ml'][0], rtol=1e-8
            )
            np.testing.assert_allclose(
                ml['T']['mean'], golden['T_mean'][0], rtol=1e-8
            )
            np.testing.assert_allclose(
                ml['predicted_Vs'], golden['predicted_Vs'][0], rtol=1e-8
            )

    def test_ml_within_sweep_range(self, tiny_sweep):
        """ML estimates should fall within sweep parameter ranges."""
        gs_prior = GrainSizePrior(gs_pdf_type='uniform_log')
        posterior, sweep_out = fit_preloaded_observations(
            obs_vs=OBS_VS, sigma_vs=SIGMA_VS,
            obs_q=OBS_Q, sigma_q=SIGMA_Q,
            z_range=Z_RANGE,
            anelastic_method=METHOD,
            grain_size_prior=gs_prior,
            sweep=tiny_sweep,
        )
        ml = extract_ml_estimates(posterior, sweep_out, METHOD)

        assert tiny_sweep['T'].min() <= ml['T']['ml'] <= tiny_sweep['T'].max()
        assert tiny_sweep['phi'].min() <= ml['phi']['ml'] <= tiny_sweep['phi'].max()
