"""
Shared pytest configuration and fixtures for regression tests.

Adds --regenerate-golden CLI flag. When set, tests write outputs
to golden files instead of comparing against them.
"""

import pytest
import numpy as np
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "golden"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-golden",
        action="store_true",
        default=False,
        help="Regenerate golden reference files instead of comparing.",
    )


@pytest.fixture(scope="session")
def regenerate(request):
    """Return True when --regenerate-golden is passed."""
    return request.config.getoption("--regenerate-golden")


@pytest.fixture(scope="session")
def golden_dir():
    """Return Path to the golden/ directory, creating it if needed."""
    GOLDEN_DIR.mkdir(exist_ok=True)
    return GOLDEN_DIR


def assert_or_store_golden(golden_dir, regenerate, filename, **arrays):
    """
    If regenerate=True, save arrays to golden_dir/filename.
    Otherwise, load golden file and return it for comparison.

    Parameters
    ----------
    golden_dir : Path
        Directory containing golden .npz files
    regenerate : bool
        If True, write arrays and skip the test
    filename : str
        Name of the .npz file (e.g., 'vbr_anelastic.npz')
    **arrays : dict
        Named arrays to save/compare

    Returns
    -------
    dict or None
        Loaded golden data (numpy NpzFile) if not regenerating, else None
    """
    path = golden_dir / filename
    if regenerate:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **arrays)
        pytest.skip(f"Golden file regenerated: {path.name}")
    else:
        if not path.exists():
            pytest.fail(
                f"Golden file missing: {path.name}. "
                f"Run with --regenerate-golden to create it."
            )
        return np.load(str(path), allow_pickle=True)


# ─── Shared expensive fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="session")
def standard_state_variables():
    """
    A small StateVariables instance (3T x 2phi x 2gs = 12 points, 4 freqs)
    suitable for exercising all VBR methods quickly.
    """
    from vbrcpy.vbr.core import StateVariables

    T_C = np.array([1200.0, 1400.0, 1600.0])
    phi = np.array([0.0, 0.005])
    gs = np.array([1000.0, 10000.0])  # micrometers
    T_grid, phi_grid, gs_grid = np.meshgrid(T_C + 273.15, phi, gs, indexing='ij')
    shape = T_grid.shape

    return StateVariables(
        T_K=T_grid,
        P_GPa=np.full(shape, 3.0),
        rho=np.full(shape, 3300.0),
        dg_um=gs_grid,
        phi=phi_grid,
        sig_MPa=np.full(shape, 0.1),
        f=np.logspace(-2.2, -1.3, 4),
    )


@pytest.fixture(scope="session")
def tiny_sweep():
    """
    Generate a tiny sweep (5T x 3phi x 4gs x 5z) with all 4 anelastic methods.
    Used by fitting and parallel tests.
    """
    from vbrcpy.vbr.generate_sweep import (
        generate_parameter_sweep, SweepParams
    )

    params = SweepParams(
        T=np.linspace(1200, 1600, 5),
        phi=np.array([0.0, 0.005, 0.02]),
        gs=np.logspace(np.log10(500), np.log10(10000), 4),
        z_min=75.0,
        z_max=150.0,
        n_z=5,
        n_freq=4,
        freq_log_min=-2.2,
        freq_log_max=-1.3,
        anelastic_methods=['eburgers_psp', 'andrade_psp', 'xfit_mxw', 'xfit_premelt'],
        eburgers_method='FastBurger',
        viscous_method='HK2003',
        density_model='constant',
        rho=3300.0,
    )
    sweep = generate_parameter_sweep(params, verbose=False)
    return sweep
