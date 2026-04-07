"""I/O utilities for ML-estimate CSV output (split-file format).

Instead of writing one wide ``ml_estimates.csv`` file, results are split
into variable-grouped CSVs inside a ``ml_estimates/`` directory.  Each
file repeats the key columns (lat, lon, z, anelastic_method) so it can
be loaded independently.

The loader (:func:`load_ml_estimates`) transparently supports both the
new split-directory format and the legacy single-CSV format.
"""

import csv
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional


# Key columns included in every split file for self-contained loading.
KEY_COLS = ['lat', 'lon', 'z', 'anelastic_method']

# Split-file definitions: filename stem -> list of data columns.
# Key columns are always prepended.  Files are only written if at least
# one of their data columns is present in the records.
SPLIT_FILE_GROUPS = OrderedDict([
    ('coordinates', ['name', 'z_min', 'z_max']),
    ('temperature', ['T_ml', 'T_std', 'T_mean']),
    ('melt', ['phi_ml', 'phi_std', 'phi_mean']),
    ('grainsize', ['gs_ml_mm', 'gs_std_mm', 'gs_mean_mm']),
    ('viscosity', ['log10_eta_ml', 'log10_eta_std', 'log10_eta_mean']),
    ('fit_quality', ['Vs_obs', 'Vs_pred', 'Vs_misfit', 'Vs_chi2',
                     'Q_obs', 'Q_pred', 'Q_misfit', 'Q_chi2', 'chi2_total']),
])


def _format_value(v):
    """Format a value for CSV output."""
    if isinstance(v, float):
        if abs(v) < 0.001 and v != 0:
            return f'{v:.3e}'
        return f'{v:.3f}'
    return v


def write_split_ml_csv(
    records: List[Dict[str, Any]],
    output_dir: str,
) -> List[str]:
    """Write ML-estimate records into split CSV files.

    Creates a ``ml_estimates/`` subdirectory inside *output_dir* containing
    one CSV per variable group (temperature.csv, melt.csv, etc.).  Each
    file includes lat/lon/z/anelastic_method as key columns so it can be
    loaded independently.

    Parameters
    ----------
    records : list of dict
        The flat list of per-location ML estimate records.
    output_dir : str
        Parent directory.  A ``ml_estimates/`` subdirectory will be created.

    Returns
    -------
    list of str
        Paths to the files that were written.
    """
    if not records:
        return []

    est_dir = os.path.join(output_dir, 'ml_estimates')
    os.makedirs(est_dir, exist_ok=True)

    # Determine which columns actually exist in the records
    all_record_keys: set = set()
    for r in records:
        all_record_keys.update(r.keys())

    written: List[str] = []
    for group_name, data_cols in SPLIT_FILE_GROUPS.items():
        # Only include data columns that exist in the records
        present_data_cols = [c for c in data_cols if c in all_record_keys]
        if not present_data_cols:
            continue

        # Determine which key columns are present
        present_key_cols = [c for c in KEY_COLS if c in all_record_keys]
        fieldnames = present_key_cols + present_data_cols

        csv_path = os.path.join(est_dir, f'{group_name}.csv')
        formatted = [
            {k: _format_value(r.get(k, '')) for k in fieldnames}
            for r in records
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(formatted)
        written.append(csv_path)

    return written


def load_ml_estimates(
    path: str,
    method: Optional[str] = None,
):
    """Load ML estimates from either a split-file directory or single CSV.

    Parameters
    ----------
    path : str
        Path to either:
        - A ``ml_estimates/`` directory containing split CSV files, or
        - A single ``ml_estimates.csv`` file (backward compatibility).
    method : str, optional
        If given, filter to this anelastic method.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame with all columns.
    """
    import pandas as pd

    path = Path(path)

    if path.is_dir():
        # Split-file format: load and merge on key columns
        dfs: Dict[str, Any] = {}
        for csv_file in sorted(path.glob('*.csv')):
            dfs[csv_file.stem] = pd.read_csv(csv_file)

        if not dfs:
            return pd.DataFrame()

        # Start with coordinates if available, otherwise first file
        if 'coordinates' in dfs:
            merged = dfs.pop('coordinates')
        else:
            first_key = next(iter(dfs))
            merged = dfs.pop(first_key)

        # Merge remaining on shared key columns
        for _name, df in dfs.items():
            merge_cols = [
                c for c in KEY_COLS
                if c in merged.columns and c in df.columns
            ]
            if merge_cols:
                data_cols = [c for c in df.columns if c not in merge_cols]
                merged = merged.merge(
                    df[merge_cols + data_cols],
                    on=merge_cols,
                    how='outer',
                )
            else:
                merged = pd.concat([merged, df], axis=1)

    elif path.is_file():
        # Single-file format (backward compatibility)
        merged = pd.read_csv(path)
    else:
        raise FileNotFoundError(f"No ML estimates found at {path}")

    if method is not None and 'anelastic_method' in merged.columns:
        merged = merged[merged['anelastic_method'] == method].copy()

    return merged


def find_ml_estimates(search_dir: str) -> Optional[str]:
    """Auto-detect the ML estimates path (directory or CSV file).

    Looks for ``ml_estimates/`` directory first, then falls back to
    ``ml_estimates.csv``.

    Parameters
    ----------
    search_dir : str
        Directory to search in (e.g. the inversion output directory).

    Returns
    -------
    str or None
        Path to the estimates location, or None if not found.
    """
    estimates_dir = os.path.join(search_dir, 'ml_estimates')
    if os.path.isdir(estimates_dir):
        return estimates_dir

    csv_file = os.path.join(search_dir, 'ml_estimates.csv')
    if os.path.isfile(csv_file):
        return csv_file

    return None
