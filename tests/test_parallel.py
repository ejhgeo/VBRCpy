"""
Parallel dispatch smoke test: verify parallel results match sequential.
"""

import numpy as np
import pytest

from vbrcpy.parallel import (
    precompute_depth_averaged_sweep,
    run_locations_parallel,
)
from vbrcpy.prior import GrainSizePrior, MeltFractionPrior
from vbrcpy.data_processing import SeismicModelData
from vbrcpy.run_bayes import InversionConfig

pytestmark = pytest.mark.parallel

Z_RANGE = (75.0, 150.0)
METHOD = 'eburgers_psp'


class TestPrecomputeDepthAveraged:
    def test_shape(self, tiny_sweep):
        """Precomputed depth-averaged sweep should have shape (nT, nphi, ngs)."""
        avg = precompute_depth_averaged_sweep(tiny_sweep, METHOD, Z_RANGE)
        expected_shape = (
            len(tiny_sweep['T']),
            len(tiny_sweep['phi']),
            len(tiny_sweep['gs']),
        )
        assert avg['meanVs'].shape == expected_shape
        assert avg['meanQ'].shape == expected_shape

    def test_matches_manual_average(self, tiny_sweep):
        """Should match manual averaging over depth indices."""
        avg = precompute_depth_averaged_sweep(tiny_sweep, METHOD, Z_RANGE)
        z_m = tiny_sweep['z']
        z_inds = np.where(
            (z_m >= Z_RANGE[0] * 1e3) & (z_m <= Z_RANGE[1] * 1e3)
        )[0]
        raw_vs = tiny_sweep['Box'][METHOD]['meanVs']
        expected_vs = np.mean(raw_vs[:, :, :, z_inds], axis=3)
        np.testing.assert_allclose(avg['meanVs'], expected_vs, rtol=1e-12)


class TestParallelMatchesSequential:
    def test_2_locations_match(self, tiny_sweep):
        """2 locations via parallel (n_workers=2) should match sequential."""
        locations = [(35.0, -120.0), (36.0, -119.0)]
        names = ['loc1', 'loc2']
        z_ranges = [Z_RANGE, Z_RANGE]

        # Build SeismicModelData with synthetic observations
        vs_obs = np.array([4.3, 4.2])
        vs_err = np.array([0.05, 0.05])
        q_obs = np.array([80.0, 75.0])
        q_err = np.array([10.0, 10.0])
        depths = np.array([112.5, 112.5])  # midpoints

        seismic_data = SeismicModelData(
            locations=locations,
            names=names,
            z_ranges=z_ranges,
            depths=depths,
            Vs=vs_obs,
            Vs_error=vs_err,
            Q=q_obs,
            Q_error=q_err,
        )

        gs_prior = GrainSizePrior(gs_pdf_type='uniform_log')
        melt_prior = MeltFractionPrior(phi_prior_type='uniform')
        config = InversionConfig(
            save_ml_csv=False,
            default_vs_error=0.05,
            default_q_error=10.0,
            q_error_mode='absolute',
        )

        # Run sequential (1 worker)
        results_seq = run_locations_parallel(
            locations=locations,
            names=names,
            z_ranges=z_ranges,
            seismic_model_data=seismic_data,
            sweep=tiny_sweep,
            anelastic_method=METHOD,
            grain_size_prior=gs_prior,
            config=config,
            n_workers=1,
            use_vs=True,
            use_q=True,
            melt_fraction_prior=melt_prior,
            temperature_prior=None,
            lightweight_results=True,
        )

        # Run parallel (2 workers)
        results_par = run_locations_parallel(
            locations=locations,
            names=names,
            z_ranges=z_ranges,
            seismic_model_data=seismic_data,
            sweep=tiny_sweep,
            anelastic_method=METHOD,
            grain_size_prior=gs_prior,
            config=config,
            n_workers=2,
            use_vs=True,
            use_q=True,
            melt_fraction_prior=melt_prior,
            temperature_prior=None,
            lightweight_results=True,
        )

        # Compare ML estimates
        assert len(results_seq) == len(results_par) == 2

        for i in range(2):
            seq_ml = results_seq[i]['ml_est']
            par_ml = results_par[i]['ml_est']
            np.testing.assert_allclose(
                seq_ml['T']['ml'], par_ml['T']['ml'], rtol=1e-12,
                err_msg=f"Location {i}: T_ml mismatch"
            )
            np.testing.assert_allclose(
                seq_ml['phi']['ml'], par_ml['phi']['ml'], rtol=1e-12,
                err_msg=f"Location {i}: phi_ml mismatch"
            )
            np.testing.assert_allclose(
                seq_ml['gs']['ml'], par_ml['gs']['ml'], rtol=1e-12,
                err_msg=f"Location {i}: gs_ml mismatch"
            )
            np.testing.assert_allclose(
                seq_ml['predicted_Vs'], par_ml['predicted_Vs'], rtol=1e-12,
                err_msg=f"Location {i}: predicted_Vs mismatch"
            )
