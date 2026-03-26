#!/usr/bin/env python3
"""
Benchmark validation: Python VBRc vs original MATLAB VBRc.

Generates a parameter sweep with the same settings used in the original
MATLAB code (vbr/test.mat), compares them numerically, plots lookup
tables with side-by-side MATLAB comparison, and runs the Bayesian
inversion on 3 manual test locations with all 4 anelastic methods.

Usage
-----
From the workspace root::

    python -m vbrc_V2Tpy.validation.benchmarkTest_vsMatlab

Steps
-----
1. Generate a parameter sweep from sweep_config.yaml (with caching).
2. Point-by-point comparison of Python sweep vs MATLAB test.mat
   (akin to compare_sweeps.py — results printed to terminal).
3. Plot LUT comparison slices (T vs gs, gs vs phi, T vs phi) for each
   method at 2 GPa.
4. Run the Bayesian inversion (manual locations, all 4 methods).
"""

import os
import sys
import json
import hashlib
import subprocess
import numpy as np
import scipy.io as sio
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BENCHMARK_DIR, '..', '..', '..'))

CONFIG_FILE = os.path.join(BENCHMARK_DIR, 'config.yaml')
MATLAB_SWEEP = os.path.join(REPO_ROOT, 'vbr', 'test.mat')

# All output paths derive from the single output_dir in config.yaml.
with open(CONFIG_FILE, 'r') as _f:
    _cfg = yaml.safe_load(_f)
_output_dir_rel = _cfg.get('output_dir', 'validation_tests/benchmarkTest_vsMatlab')
OUTPUT_DIR = os.path.join(REPO_ROOT, _output_dir_rel)

SWEEP_FILE        = os.path.join(OUTPUT_DIR, 'sweep.npz')
SWEEP_FINGERPRINT = os.path.join(OUTPUT_DIR, 'sweep_fingerprint.json')
INVERSION_DIR     = os.path.join(OUTPUT_DIR, 'inversion_results')

METHODS = ['andrade_psp', 'eburgers_psp', 'xfit_mxw', 'xfit_premelt']

# Fixed pressures at which to always plot comparison LUTs
LUT_P_GPA_FIXED = [2.0]


# ===================================================================
# Sweep config fingerprinting
# ===================================================================
def _config_fingerprint(config_path):
    """SHA-256 hash of only the sweep_generation section of the config.

    Ignores inversion-only parameters so that changing priors or output
    settings does not trigger unnecessary sweep regeneration.
    """
    import json as _json
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    section = cfg.get('sweep_generation', {})
    return hashlib.sha256(_json.dumps(section, sort_keys=True).encode()).hexdigest()


def _sweep_needs_regeneration():
    if not os.path.isfile(SWEEP_FILE):
        return True
    if not os.path.isfile(SWEEP_FINGERPRINT):
        return True
    current = _config_fingerprint(CONFIG_FILE)
    with open(SWEEP_FINGERPRINT, 'r') as f:
        saved = json.load(f).get('hash')
    return current != saved


def _save_sweep_fingerprint():
    fp = _config_fingerprint(CONFIG_FILE)
    with open(SWEEP_FINGERPRINT, 'w') as f:
        json.dump({'hash': fp, 'config': CONFIG_FILE}, f)


# ===================================================================
# Step 2: Point-by-point comparison (mirrors compare_sweeps.py)
# ===================================================================
def _compare_sweeps_numerically(python_sweep_file, matlab_sweep_file):
    """Compare Python and MATLAB sweeps at representative grid points.

    Prints per-method Vs/Q differences and melt-effect comparisons to
    the terminal, in the same style as compare_sweeps.py.
    """
    # --- Load Python sweep (.npz) via the project loader ---
    sys.path.insert(0, REPO_ROOT)
    from vbrc_V2Tpy.bayesian_fitting_py.fitting import load_sweep_data
    sweep_py = load_sweep_data(python_sweep_file)

    # --- Load MATLAB sweep (raw struct, same as compare_sweeps.py) ---
    mat = sio.loadmat(matlab_sweep_file)
    Box_mat = mat['sweep']['Box'][0, 0]
    T_mat = mat['sweep']['T'][0, 0].flatten()
    gs_mat = mat['sweep']['gs'][0, 0].flatten()
    phi_mat = mat['sweep']['phi'][0, 0].flatten()
    z_mat = mat['sweep']['z'][0, 0].flatten()

    T_py = sweep_py['T']
    phi_py = sweep_py['phi']
    gs_py = sweep_py['gs']
    z_py = sweep_py['z']

    # Representative grid point (same indices as compare_sweeps.py)
    i_T = 12
    i_gs = 10
    i_z = 0

    print(f"=== Comparing Python vs MATLAB at grid point ===")
    print(f"T = {T_mat[i_T]:.0f} C,  gs = {gs_mat[i_gs]:.0f} um,  "
          f"z = {z_mat[i_z]/1000:.0f} km")
    print()

    all_ok = True
    phi_indices = [0, 8]  # phi=0, phi≈0.02

    for method in METHODS:
        print(f"=== {method} comparison ===")
        for i_phi in phi_indices:
            # Python values
            vs_py = float(sweep_py['Box'][method]['meanVs'][i_T, i_phi, i_gs, i_z])
            q_py = float(sweep_py['Box'][method]['meanQ'][i_T, i_phi, i_gs, i_z])

            # MATLAB values (raw loadmat struct access)
            e_mat = Box_mat[i_T, i_phi, i_gs][method][0, 0]
            vs_mat = float(e_mat['meanVs'].flatten()[i_z])
            q_mat = float(e_mat['meanQ'].flatten()[i_z])

            dVs = vs_py - vs_mat
            dQ = q_py - q_mat
            pctVs = dVs / vs_mat * 100 if vs_mat != 0 else 0
            pctQ = dQ / q_mat * 100 if q_mat != 0 else 0

            if abs(pctVs) >= 0.01 or abs(pctQ) >= 0.1:
                all_ok = False

            print(f"phi={phi_mat[i_phi]:.4f}:")
            print(f"  Vs: MATLAB={vs_mat:.6f}, Python={vs_py:.6f}, "
                  f"diff={(dVs)*1000:+.2f} m/s ({pctVs:+.4f}%)")
            print(f"  Q:  MATLAB={q_mat:.4f}, Python={q_py:.4f}, "
                  f"diff={dQ:+.4f} ({pctQ:+.4f}%)")
            print()

        # --- Melt effect (phi≈0.02 vs phi=0) ---
        vs_py_0 = float(sweep_py['Box'][method]['meanVs'][i_T, 0, i_gs, i_z])
        vs_py_m = float(sweep_py['Box'][method]['meanVs'][i_T, 8, i_gs, i_z])
        e_mat_0 = Box_mat[i_T, 0, i_gs][method][0, 0]
        e_mat_m = Box_mat[i_T, 8, i_gs][method][0, 0]
        vs_mat_0 = float(e_mat_0['meanVs'].flatten()[i_z])
        vs_mat_m = float(e_mat_m['meanVs'].flatten()[i_z])

        print(f"=== Melt effect (phi=0.02 vs phi=0) for {method} ===")
        print(f"MATLAB: dVs = {(vs_mat_m - vs_mat_0)*1000:.2f} m/s "
              f"({(vs_mat_m - vs_mat_0)/vs_mat_0*100:.3f}%)")
        print(f"Python: dVs = {(vs_py_m - vs_py_0)*1000:.2f} m/s "
              f"({(vs_py_m - vs_py_0)/vs_py_0*100:.3f}%)")
        print(f"Difference: "
              f"{((vs_py_m - vs_py_0) - (vs_mat_m - vs_mat_0))*1000:.2f} m/s")
        print()

    # --- Full-grid summary (phi=0 slice) ---
    print("=" * 60)
    print("Full-grid summary (phi=0 slice, all T / gs / z)")
    print("=" * 60)
    nT = len(T_py)
    ngs = len(gs_py)
    nz = len(z_py)
    for method in METHODS:
        vs_py_arr = sweep_py['Box'][method]['meanVs'][:, 0, :, :]
        q_py_arr = sweep_py['Box'][method]['meanQ'][:, 0, :, :]
        vs_mat_arr = np.zeros((nT, ngs, nz))
        q_mat_arr = np.zeros((nT, ngs, nz))
        for iT in range(nT):
            for igs in range(ngs):
                e = Box_mat[iT, 0, igs][method][0, 0]
                vs_mat_arr[iT, igs, :] = e['meanVs'].flatten()
                q_mat_arr[iT, igs, :] = e['meanQ'].flatten()

        pctVs = 100 * (vs_py_arr - vs_mat_arr) / vs_mat_arr
        valid_Q = q_mat_arr != 0
        pctQ = np.zeros_like(q_mat_arr)
        pctQ[valid_Q] = 100 * (q_py_arr[valid_Q] - q_mat_arr[valid_Q]) / q_mat_arr[valid_Q]

        abs_pctVs = np.abs(pctVs)
        abs_pctQ = np.abs(pctQ)

        # Max and where it occurs
        idx_vs_max = np.unravel_index(np.argmax(abs_pctVs), abs_pctVs.shape)
        idx_q_max = np.unravel_index(np.argmax(abs_pctQ), abs_pctQ.shape)
        z_vs_max_km = z_py[idx_vs_max[2]] / 1e3
        z_q_max_km = z_py[idx_q_max[2]] / 1e3

        # Median
        med_pctVs = np.median(abs_pctVs)
        med_pctQ = np.median(abs_pctQ[valid_Q]) if np.any(valid_Q) else 0.0

        print(f"  {method:18s}")
        print(f"    Vs: median|%diff| = {med_pctVs:.6f}%,  "
              f"max|%diff| = {np.max(abs_pctVs):.6f}% "
              f"(at z = {z_vs_max_km:.1f} km, "
              f"T = {T_py[idx_vs_max[0]]:.0f} C, "
              f"gs = {gs_py[idx_vs_max[1]]:.0f} um)")
        print(f"    Q:  median|%diff| = {med_pctQ:.6f}%,  "
              f"max|%diff| = {np.max(abs_pctQ[valid_Q]):.6f}% "
              f"(at z = {z_q_max_km:.1f} km, "
              f"T = {T_py[idx_q_max[0]]:.0f} C, "
              f"gs = {gs_py[idx_q_max[1]]:.0f} um)")

    # Collect the P_GPa values where the worst Vs diff occurs per method
    P_GPa_py = sweep_py['P_GPa']
    max_diff_pressures = {}
    for method in METHODS:
        vs_py_arr = sweep_py['Box'][method]['meanVs'][:, 0, :, :]
        q_py_arr = sweep_py['Box'][method]['meanQ'][:, 0, :, :]
        vs_mat_arr = np.zeros((nT, ngs, nz))
        for iT in range(nT):
            for igs in range(ngs):
                e = Box_mat[iT, 0, igs][method][0, 0]
                vs_mat_arr[iT, igs, :] = e['meanVs'].flatten()
        abs_pct = np.abs(100 * (vs_py_arr - vs_mat_arr) / vs_mat_arr)
        idx = np.unravel_index(np.argmax(abs_pct), abs_pct.shape)
        max_diff_pressures[method] = float(P_GPa_py[idx[2]])

    print()
    if all_ok:
        print("PASS — All point differences within tolerance.")
    else:
        print("NOTE — Some differences exceed tight tolerances (see above).")
    return all_ok, max_diff_pressures


# ===================================================================
# Step 3: LUT comparison plots
# ===================================================================
def _plot_lut_comparisons(python_sweep_file, matlab_sweep_file,
                         max_diff_pressures=None):
    """Generate LUT comparison figures for all methods at multiple depths.

    Plots at the fixed pressures in LUT_P_GPA_FIXED plus the first, middle,
    and last pressure in the sweep, and (per method) the pressure where the
    maximum Vs difference was found.

    Produces three comparison plot types per method per pressure:
      - T vs gs   (compare_lut_slices_T_gs)
      - gs vs phi (compare_lut_slices_gs_phi)
      - T vs phi  (compare_lut_slices_T_phi)
    Plus standalone Python-only LUT slices.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from vbrc_V2Tpy.bayesian_fitting_py.vbr.plot_lut import (
        _load_sweep_file,
        compare_lut_slices_T_gs,
        compare_lut_slices_gs_phi,
        compare_lut_slices_T_phi,
    )

    sweep_py = _load_sweep_file(python_sweep_file)
    sweep_mat = _load_sweep_file(matlab_sweep_file)

    P_arr = np.atleast_1d(
        sweep_py['P_GPa'] if isinstance(sweep_py, dict) else sweep_py.P_GPa)

    # Build the set of pressures to plot: fixed + first/mid/last of sweep
    base_pressures = sorted(set(
        LUT_P_GPA_FIXED + [float(P_arr[0]),
                           float(P_arr[len(P_arr)//2]),
                           float(P_arr[-1])]))

    if max_diff_pressures is None:
        max_diff_pressures = {}

    lut_dir = os.path.join(OUTPUT_DIR, 'lut_comparison')
    os.makedirs(lut_dir, exist_ok=True)

    # Suppress per-figure "Saved figure to ..." prints from plot_lut
    import io
    import contextlib

    n_figs = 0
    for method in METHODS:
        method_dir = os.path.join(lut_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # Pressures for this method = base + max-diff pressure (if unique)
        p_maxdiff = max_diff_pressures.get(method)
        pressures = list(base_pressures)
        if p_maxdiff is not None and p_maxdiff not in pressures:
            pressures.append(p_maxdiff)
            pressures.sort()

        for P in pressures:
            tag = f" (max-diff depth)" if p_maxdiff is not None and P == p_maxdiff else ""
            print(f"    {method} @ {P:.2f} GPa{tag}")

            with contextlib.redirect_stdout(io.StringIO()):
                fig = compare_lut_slices_T_gs(
                    sweep_mat, sweep_py, method=method,
                    P_GPa=P, save_path=method_dir)
                plt.close(fig)

                fig = compare_lut_slices_gs_phi(
                    sweep_mat, sweep_py, method=method,
                    P_GPa=P, save_path=method_dir)
                plt.close(fig)

                fig = compare_lut_slices_T_phi(
                    sweep_mat, sweep_py, method=method,
                    P_GPa=P, save_path=method_dir)
                plt.close(fig)

            n_figs += 3

    print(f"  Saved {n_figs} LUT figures to {lut_dir}/")


# ===================================================================
# Main
# ===================================================================
def main():
    os.chdir(REPO_ROOT)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    python = sys.executable

    # Verify MATLAB reference exists
    if not os.path.isfile(MATLAB_SWEEP):
        print(f"ERROR: MATLAB reference sweep not found: {MATLAB_SWEEP}")
        print("       This file is required for the benchmark comparison.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: Generate the parameter sweep
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Generate parameter sweep (look-up table)")
    print("=" * 70, flush=True)
    if _sweep_needs_regeneration():
        if os.path.isfile(SWEEP_FILE):
            print("  Sweep config changed — regenerating ...")
        from vbrc_V2Tpy.bayesian_fitting_py.vbr.generate_sweep import (
            load_sweep_params_from_yaml as _lspy,
            generate_parameter_sweep as _gps,
            save_sweep as _ss,
        )
        _params = _lspy(CONFIG_FILE)
        _params.output_file = SWEEP_FILE
        _sweep = _gps(_params)
        _ss(_sweep, _params.output_file)
        _save_sweep_fingerprint()
    else:
        print(f"  Sweep up-to-date at {SWEEP_FILE} — skipping generation.")

    # ------------------------------------------------------------------
    # Step 2: Numerical comparison against MATLAB
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 2: Numerical comparison — Python vs MATLAB (cf. compare_sweeps.py)")
    print("=" * 70, flush=True)
    _, max_diff_pressures = _compare_sweeps_numerically(SWEEP_FILE, MATLAB_SWEEP)

    # ------------------------------------------------------------------
    # Step 3: LUT comparison plots
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 3: LUT comparison plots — Python vs MATLAB")
    print("        (multiple depths incl. max-diff depth per method)")
    print("=" * 70, flush=True)
    _plot_lut_comparisons(SWEEP_FILE, MATLAB_SWEEP,
                          max_diff_pressures=max_diff_pressures)

    # ------------------------------------------------------------------
    # Step 4: Bayesian inversion
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STEP 4: Bayesian inversion (3 manual locations, 4 methods)")
    print("=" * 70, flush=True)
    cmd = [
        python, '-m',
        'vbrc_V2Tpy.bayesian_fitting_py',
        '--config', CONFIG_FILE,
        '--sweep-file', SWEEP_FILE,
        '--output-dir', INVERSION_DIR,
    ]
    print(f"  Running: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Benchmark validation complete.  Outputs in:")
    print(f"  {OUTPUT_DIR}")
    print()
    print("Contents:")
    print("  sweep.npz                       — Python-generated sweep")
    print("  lut_comparison/<method>/         — Comparison plots (Python vs MATLAB)")
    print("    lut_comparison_T_vs_gs_*.png")
    print("    lut_comparison_phi_vs_gs_*.png")
    print("    lut_comparison_T_vs_phi_*.png")
    print("  lut_comparison/<method>_python/  — Standalone Python LUT slices")
    print("  inversion_results/               — Inversion posteriors & ML estimates")
    print("=" * 70)


if __name__ == '__main__':
    main()
