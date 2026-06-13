"""
Golden-file tests for parameter sweep generation.

Uses the session-scoped tiny_sweep fixture (5T x 3phi x 4gs x 5z, all 4 methods).
"""

import numpy as np
import pytest

from tests.conftest import assert_or_store_golden

pytestmark = pytest.mark.integration


class TestSweepStructure:
    def test_required_keys(self, tiny_sweep):
        """Sweep should contain all required top-level keys."""
        required = ['T', 'phi', 'gs', 'z', 'P_GPa', 'Box', 'state_names']
        for key in required:
            assert key in tiny_sweep, f"Missing key: {key}"

    def test_state_names(self, tiny_sweep):
        """state_names should be ['T', 'phi', 'gs']."""
        assert tiny_sweep['state_names'] == ['T', 'phi', 'gs']

    def test_all_methods_present(self, tiny_sweep):
        """Box should contain all 4 anelastic methods."""
        methods = ['eburgers_psp', 'andrade_psp', 'xfit_mxw', 'xfit_premelt']
        for method in methods:
            assert method in tiny_sweep['Box'], f"Missing method: {method}"

    def test_box_keys_per_method(self, tiny_sweep):
        """Each method should have meanVs and meanQ."""
        for method in tiny_sweep['Box']:
            assert 'meanVs' in tiny_sweep['Box'][method]
            assert 'meanQ' in tiny_sweep['Box'][method]


class TestSweepShapes:
    def test_axis_lengths(self, tiny_sweep):
        """Sweep axes should have correct lengths."""
        assert len(tiny_sweep['T']) == 5
        assert len(tiny_sweep['phi']) == 3
        assert len(tiny_sweep['gs']) == 4
        assert len(tiny_sweep['z']) == 5

    def test_box_array_shapes(self, tiny_sweep):
        """Box arrays should have shape (5, 3, 4, 5) = (nT, nphi, ngs, nz)."""
        expected_shape = (5, 3, 4, 5)
        for method in tiny_sweep['Box']:
            assert tiny_sweep['Box'][method]['meanVs'].shape == expected_shape, \
                f"{method} meanVs shape mismatch"
            assert tiny_sweep['Box'][method]['meanQ'].shape == expected_shape, \
                f"{method} meanQ shape mismatch"


class TestSweepValues:
    def test_meanVs_golden(self, tiny_sweep, golden_dir, regenerate):
        """meanVs values for all methods should match golden."""
        arrays = {}
        for method in tiny_sweep['Box']:
            arrays[f'{method}_meanVs'] = tiny_sweep['Box'][method]['meanVs']
        golden = assert_or_store_golden(
            golden_dir, regenerate, "sweep_meanVs.npz", **arrays
        )
        if golden is not None:
            for method in tiny_sweep['Box']:
                key = f'{method}_meanVs'
                np.testing.assert_allclose(
                    tiny_sweep['Box'][method]['meanVs'],
                    golden[key],
                    rtol=1e-8,
                    err_msg=f"{method} meanVs mismatch",
                )

    def test_meanQ_golden(self, tiny_sweep, golden_dir, regenerate):
        """meanQ values for all methods should match golden."""
        arrays = {}
        for method in tiny_sweep['Box']:
            arrays[f'{method}_meanQ'] = tiny_sweep['Box'][method]['meanQ']
        golden = assert_or_store_golden(
            golden_dir, regenerate, "sweep_meanQ.npz", **arrays
        )
        if golden is not None:
            for method in tiny_sweep['Box']:
                key = f'{method}_meanQ'
                np.testing.assert_allclose(
                    tiny_sweep['Box'][method]['meanQ'],
                    golden[key],
                    rtol=1e-6,
                    err_msg=f"{method} meanQ mismatch",
                )

    def test_meanEta_golden(self, tiny_sweep, golden_dir, regenerate):
        """meanEta values (if present) should match golden."""
        arrays = {}
        has_eta = False
        for method in tiny_sweep['Box']:
            if 'meanEta' in tiny_sweep['Box'][method]:
                arrays[f'{method}_meanEta'] = tiny_sweep['Box'][method]['meanEta']
                has_eta = True
        if not has_eta:
            pytest.skip("No meanEta in sweep")
        golden = assert_or_store_golden(
            golden_dir, regenerate, "sweep_meanEta.npz", **arrays
        )
        if golden is not None:
            for key in arrays:
                np.testing.assert_allclose(
                    arrays[key], golden[key], rtol=1e-8,
                    err_msg=f"{key} mismatch",
                )


class TestSweepMethodDifferences:
    def test_methods_produce_distinct_vs(self, tiny_sweep):
        """Each method should produce distinct meanVs (not accidentally swapped)."""
        methods = list(tiny_sweep['Box'].keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                vs_i = tiny_sweep['Box'][methods[i]]['meanVs']
                vs_j = tiny_sweep['Box'][methods[j]]['meanVs']
                assert not np.allclose(vs_i, vs_j, rtol=1e-4), \
                    f"{methods[i]} and {methods[j]} have nearly identical meanVs"

    def test_methods_produce_distinct_q(self, tiny_sweep):
        """Each method should produce distinct meanQ."""
        methods = list(tiny_sweep['Box'].keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                q_i = tiny_sweep['Box'][methods[i]]['meanQ']
                q_j = tiny_sweep['Box'][methods[j]]['meanQ']
                assert not np.allclose(q_i, q_j, rtol=1e-4), \
                    f"{methods[i]} and {methods[j]} have nearly identical meanQ"

    def test_vs_physical_range(self, tiny_sweep):
        """All meanVs should be in physically reasonable range."""
        for method in tiny_sweep['Box']:
            vs = tiny_sweep['Box'][method]['meanVs']
            # Vs in km/s should be between 3.0 and 5.5
            assert np.all(vs > 3.0), f"{method}: Vs below 3.0 km/s"
            assert np.all(vs < 5.5), f"{method}: Vs above 5.5 km/s"

    def test_q_positive(self, tiny_sweep):
        """All meanQ values should be positive."""
        for method in tiny_sweep['Box']:
            q = tiny_sweep['Box'][method]['meanQ']
            assert np.all(q > 0), f"{method}: Q has non-positive values"
