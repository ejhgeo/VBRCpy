"""Orchestration utilities for inversion workflow scripts.

Provides reusable components for the multi-step workflow used by
example scripts (Patagonia_Test, validation cases, etc.):

- **Sweep fingerprinting**: detect when the sweep needs regeneration
- **Sweep generation**: generate/cache a parameter sweep from YAML config
- **LUT replotting**: regenerate look-up-table plots from an existing sweep
- **Inversion running**: call the Bayesian inversion via subprocess
"""

import hashlib
import json
import os
import subprocess
import sys
from typing import Optional


# ===================================================================
# Sweep fingerprinting
# ===================================================================

def config_fingerprint(config_path: str) -> str:
    """SHA-256 hash of only the ``sweep_generation`` section of a YAML config.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    str
        Hex digest of the sweep_generation section.
    """
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    section = cfg.get('sweep_generation', {})
    return hashlib.sha256(json.dumps(section, sort_keys=True).encode()).hexdigest()


def sweep_needs_regeneration(
    sweep_file: str,
    fingerprint_file: str,
    config_path: str,
) -> bool:
    """Check whether the sweep needs to be regenerated.

    Returns True if:
    - The sweep file does not exist, or
    - The fingerprint file does not exist, or
    - The current config hash differs from the saved hash.

    Parameters
    ----------
    sweep_file : str
        Path to the sweep .npz file.
    fingerprint_file : str
        Path to the fingerprint JSON file.
    config_path : str
        Path to the YAML configuration file.
    """
    if not os.path.isfile(sweep_file):
        return True
    if not os.path.isfile(fingerprint_file):
        return True
    current = config_fingerprint(config_path)
    with open(fingerprint_file, 'r') as f:
        saved = json.load(f).get('hash')
    return current != saved


def save_sweep_fingerprint(
    fingerprint_file: str,
    config_path: str,
) -> None:
    """Save the current sweep config fingerprint to disk.

    Parameters
    ----------
    fingerprint_file : str
        Path to write the fingerprint JSON file.
    config_path : str
        Path to the YAML configuration file that was used.
    """
    fp = config_fingerprint(config_path)
    with open(fingerprint_file, 'w') as f:
        json.dump({'hash': fp, 'config': config_path}, f)


# ===================================================================
# Sweep generation step
# ===================================================================

def run_sweep_step(
    config_path: str,
    sweep_file: str,
    output_dir: str,
    fingerprint_file: Optional[str] = None,
) -> bool:
    """Generate a parameter sweep if the config has changed.

    Loads sweep parameters from the YAML config, generates the sweep,
    saves it, optionally generates LUT plots, and saves the fingerprint.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    sweep_file : str
        Path to write the sweep .npz file.
    output_dir : str
        Parent output directory (used for LUT plot subdirectory).
    fingerprint_file : str, optional
        Path to the fingerprint JSON.  Defaults to
        ``{output_dir}/sweep_fingerprint.json``.

    Returns
    -------
    bool
        True if the sweep was (re)generated, False if skipped.
    """
    from .vbr.generate_sweep import (
        load_sweep_params_from_yaml,
        generate_parameter_sweep,
        save_sweep,
    )

    if fingerprint_file is None:
        fingerprint_file = os.path.join(output_dir, 'sweep_fingerprint.json')

    if not sweep_needs_regeneration(sweep_file, fingerprint_file, config_path):
        print(f"  Sweep up-to-date at {sweep_file} — skipping generation.")
        return False

    if os.path.isfile(sweep_file):
        print("  Sweep config changed — regenerating ...")

    params = load_sweep_params_from_yaml(config_path)
    params.output_file = sweep_file
    if params.plot_lut:
        params.plot_lut_dir = os.path.join(output_dir, 'lut_plots')

    sweep = generate_parameter_sweep(params)
    save_sweep(sweep, params.output_file)

    if params.plot_lut:
        from .vbr.plot_lut import generate_sweep_lut_plots
        print(f"\nGenerating LUT plots (every {params.plot_lut_every_n} depths)...")
        generate_sweep_lut_plots(sweep, params.plot_lut_dir,
                                 every_n=params.plot_lut_every_n)

    save_sweep_fingerprint(fingerprint_file, config_path)
    return True


# ===================================================================
# LUT replotting
# ===================================================================

def replot_lut(
    config_path: str,
    sweep_file: str,
    output_dir: str,
) -> int:
    """Regenerate LUT diagnostic plots from an existing sweep.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    sweep_file : str
        Path to the sweep .npz file.
    output_dir : str
        Parent output directory (LUT plots go in ``lut_plots/`` subdirectory).

    Returns
    -------
    int
        Number of figures saved.
    """
    from .fitting import load_sweep_data
    from .vbr.plot_lut import generate_sweep_lut_plots
    from .vbr.generate_sweep import load_sweep_params_from_yaml

    print(f"Loading sweep from {sweep_file} ...")
    sweep = load_sweep_data(sweep_file)

    params = load_sweep_params_from_yaml(config_path)
    params.output_file = sweep_file
    if params.plot_lut:
        params.plot_lut_dir = os.path.join(output_dir, 'lut_plots')
    plot_dir = params.plot_lut_dir
    every_n = params.plot_lut_every_n

    print(f"Generating LUT plots (every {every_n} depths) in {plot_dir} ...")
    n_saved = generate_sweep_lut_plots(sweep, plot_dir, every_n=every_n)
    print(f"Done — {n_saved} figures saved.")
    return n_saved


# ===================================================================
# Inversion runner
# ===================================================================

def run_inversion_step(
    config_path: str,
    sweep_file: str,
    output_dir: str,
    vs_file: Optional[str] = None,
    q_file: Optional[str] = None,
    parallel: int = 1,
) -> None:
    """Run the Bayesian inversion as a subprocess.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    sweep_file : str
        Path to the sweep .npz file.
    output_dir : str
        Output directory for inversion results.
    vs_file : str, optional
        Path or built-in name for Vs observations.
        If None, uses whatever is in the config.
    q_file : str, optional
        Path or built-in name for Q observations.
        If None, uses whatever is in the config.
    parallel : int
        Number of worker processes (0 = auto, 1 = sequential).
    """
    python = sys.executable
    cmd = [
        python, '-m',
        'vbrcpy',
        '--config', config_path,
        '--sweep-file', sweep_file,
        '--output-dir', output_dir,
    ]
    if vs_file is not None:
        cmd.extend(['--vs-file', vs_file])
    if q_file is not None:
        cmd.extend(['--q-file', q_file])
    if parallel != 1:
        cmd.extend(['--parallel', str(parallel)])

    print(f"  Running: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)
