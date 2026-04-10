"""
Post-processing utilities for Bayesian inversion results.

Provides:

- **CSV → NetCDF conversion** (:func:`csv_to_netcdf`): converts the
  split-file ``ml_estimates/`` CSV directory into compressed 3-D NetCDF
  files with dimensions ``(z, lat, lon)``.

- **Global map plotting** (:func:`plot_global_maps`): produces
  Robinson-projection map grids of inversion results using PyGMT.

- **Data loading helpers** for reading from either NetCDF or CSV.

CLI entry points (installed via ``pip install -e .``)::

    vbrc-to-netcdf [--csv DIR] [--outdir DIR] [--groups ...]
    vbrc-plot-maps [--nc DIR | --csv DIR] --depth Z [Z ...] [--vars ...]
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import xarray as xr


# =========================================================================
# Constants
# =========================================================================

#: Variable groups matching the CSV split structure.
#: Keys = CSV stem, values = list of data column names in that file.
VARIABLE_GROUPS: Dict[str, List[str]] = {
    "temperature": ["T_ml", "T_std", "T_mean"],
    "melt":        ["phi_ml", "phi_std", "phi_mean"],
    "grainsize":   ["gs_ml_mm", "gs_std_mm", "gs_mean_mm"],
    "viscosity":   ["log10_eta_ml", "log10_eta_std", "log10_eta_mean"],
    "fit_quality": ["Vs_obs", "Vs_pred", "Vs_misfit", "Vs_chi2",
                    "Q_obs", "Q_pred", "Q_misfit", "Q_chi2", "chi2_total"],
}

#: CF-convention attributes for NetCDF variables.
VAR_ATTRS: Dict[str, Dict[str, str]] = {
    "T_ml":            {"long_name": "Temperature (MAP estimate)",          "units": "degC"},
    "T_std":           {"long_name": "Temperature (posterior std dev)",     "units": "degC"},
    "T_mean":          {"long_name": "Temperature (posterior mean)",        "units": "degC"},
    "phi_ml":          {"long_name": "Melt fraction (MAP estimate)",       "units": "1"},
    "phi_std":         {"long_name": "Melt fraction (posterior std dev)",   "units": "1"},
    "phi_mean":        {"long_name": "Melt fraction (posterior mean)",      "units": "1"},
    "gs_ml_mm":        {"long_name": "Grain size (MAP estimate)",          "units": "mm"},
    "gs_std_mm":       {"long_name": "Grain size (posterior std dev)",      "units": "mm"},
    "gs_mean_mm":      {"long_name": "Grain size (posterior mean)",         "units": "mm"},
    "log10_eta_ml":    {"long_name": "log10 Viscosity (MAP estimate)",     "units": "log10(Pa s)"},
    "log10_eta_std":   {"long_name": "log10 Viscosity (posterior std dev)","units": "log10(Pa s)"},
    "log10_eta_mean":  {"long_name": "log10 Viscosity (posterior mean)",   "units": "log10(Pa s)"},
    "Vs_obs":          {"long_name": "Observed shear-wave velocity",       "units": "km/s"},
    "Vs_pred":         {"long_name": "Predicted shear-wave velocity",      "units": "km/s"},
    "Vs_misfit":       {"long_name": "Vs misfit (obs - pred)",             "units": "km/s"},
    "Vs_chi2":         {"long_name": "Vs chi-squared contribution",        "units": "1"},
    "Q_obs":           {"long_name": "Observed quality factor",            "units": "1"},
    "Q_pred":          {"long_name": "Predicted quality factor",           "units": "1"},
    "Q_misfit":        {"long_name": "Q misfit (obs - pred)",              "units": "1"},
    "Q_chi2":          {"long_name": "Q chi-squared contribution",         "units": "1"},
    "chi2_total":      {"long_name": "Total chi-squared",                  "units": "1"},
}

#: Default colour-map / series / label styles for PyGMT plotting.
VAR_STYLE: Dict[str, Dict[str, Any]] = {
    # Temperature
    "T_mean":  {"cmap": "roma",  "series": [800, 2000, 10],   "unit": "Temperature °C",             "reverse": True,  "label": "T (post. mean)"},
    "T_ml":    {"cmap": "roma",  "series": [800, 2000, 10],   "unit": "Temperature °C",             "reverse": True,  "label": "T (MAP)"},
    "T_std":   {"cmap": "roma",  "series": [0,   200,  10],   "unit": "Temperature °C",             "reverse": True,  "label": "T std"},
    # Viscosity
    "log10_eta_mean": {"cmap": "haxby",  "series": [18, 25, 0.1],   "unit": "log10 Viscosity (Pa·s)", "reverse": True,  "label": "Viscosity (post. mean)"},
    "log10_eta_ml":   {"cmap": "haxby",  "series": [18, 25, 0.1],   "unit": "log10 Viscosity (Pa·s)", "reverse": True,  "label": "Viscosity (MAP)"},
    # Melt fraction
    "phi_mean": {"cmap": "bilbao", "series": [0, 0.05, 0.001], "unit": "Melt Fraction", "reverse": True,  "label": "Melt Fraction (post. mean)"},
    "phi_ml":   {"cmap": "bilbao", "series": [0, 0.05, 0.001], "unit": "Melt Fraction", "reverse": True,  "label": "Melt Fraction (MAP)"},
    # Grain size
    "gs_mean_mm": {"cmap": "davos", "series": [0.1, 5.0, 0.1], "unit": "Grain Size mm",            "reverse": True,  "label": "Grain Size (post. mean)"},
    "gs_ml_mm":   {"cmap": "davos", "series": [0.1, 5.0, 0.1], "unit": "Grain Size mm",            "reverse": True,  "label": "Grain Size (MAP)"},
    # Vs
    "Vs_obs":  {"cmap": "roma",  "series": [4.0, 4.7, 0.01],  "unit": "Vs km/s",           "reverse": False, "label": "Vs obs"},
    "Vs_pred": {"cmap": "roma",  "series": [4.0, 4.7, 0.01],  "unit": "Vs km/s",           "reverse": False, "label": "Vs pred"},
}

#: Default variables to plot when none are specified.
DEFAULT_PLOT_VARS = ["T_mean", "log10_eta_mean", "phi_mean", "gs_mean_mm"]


# =========================================================================
# CSV → NetCDF conversion
# =========================================================================

def _build_grid(csv_path: str):
    """Read a single CSV and determine the (lat, lon, z) grid vectors.

    Returns
    -------
    lat, lon, z : numpy.ndarray
        Sorted 1-D coordinate arrays.
    """
    import pandas as pd

    print(f"  Scanning grid structure from {os.path.basename(csv_path)} ...")
    df = pd.read_csv(csv_path, usecols=["lat", "lon", "z"])
    lat = np.sort(df["lat"].unique())
    lon = np.sort(df["lon"].unique())
    z = np.sort(df["z"].unique())
    print(f"  Grid: {len(lat)} lat × {len(lon)} lon × {len(z)} depths")
    return lat, lon, z


def _convert_one_group(
    csv_path: str,
    data_cols: List[str],
    lat: np.ndarray,
    lon: np.ndarray,
    z: np.ndarray,
    outdir: str,
    group_name: str,
) -> Optional[str]:
    """Read one CSV, reshape to 3-D, write a compressed NetCDF.

    Returns the path to the written file, or *None* if skipped.
    """
    import pandas as pd

    if not os.path.isfile(csv_path):
        print(f"  Skipping {group_name}: {csv_path} not found")
        return None

    t0 = time.time()
    stem = os.path.basename(csv_path).replace(".csv", "")
    print(f"  Loading {stem}.csv ...")

    usecols = ["lat", "lon", "z"] + data_cols
    try:
        df = pd.read_csv(csv_path, usecols=usecols)
    except ValueError:
        present = pd.read_csv(csv_path, nrows=0).columns.tolist()
        usecols = ["lat", "lon", "z"] + [c for c in data_cols if c in present]
        data_cols = [c for c in data_cols if c in present]
        if not data_cols:
            print(f"  Skipping {group_name}: no matching data columns")
            return None
        df = pd.read_csv(csv_path, usecols=usecols)

    print(f"    {len(df):,} rows loaded ({time.time()-t0:.1f}s)")

    lat_idx = {v: i for i, v in enumerate(lat)}
    lon_idx = {v: i for i, v in enumerate(lon)}
    z_idx = {v: i for i, v in enumerate(z)}
    nz, nlat, nlon = len(z), len(lat), len(lon)

    ds = xr.Dataset(
        coords={
            "z":   ("z",   z,   {"long_name": "Depth", "units": "km"}),
            "lat": ("lat", lat, {"long_name": "Latitude", "units": "degrees_north"}),
            "lon": ("lon", lon, {"long_name": "Longitude", "units": "degrees_east"}),
        },
    )

    zi = df["z"].map(z_idx).values
    li = df["lat"].map(lat_idx).values
    oi = df["lon"].map(lon_idx).values

    for col in data_cols:
        print(f"    Reshaping {col} ...", end="", flush=True)
        arr = np.full((nz, nlat, nlon), np.nan, dtype=np.float32)
        arr[zi, li, oi] = df[col].values.astype(np.float32)
        attrs = VAR_ATTRS.get(col, {"long_name": col})
        ds[col] = xr.DataArray(arr, dims=["z", "lat", "lon"], attrs=attrs)
        print(" done")

    del df

    nc_path = os.path.join(outdir, f"{group_name}.nc")
    encoding = {
        v: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for v in data_cols
    }
    print(f"    Writing {nc_path} ...")
    ds.to_netcdf(nc_path, encoding=encoding)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(nc_path) / 1e6
    print(f"    {nc_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")
    return nc_path


def csv_to_netcdf(
    csv_dir: str,
    outdir: Optional[str] = None,
    groups: Optional[List[str]] = None,
) -> List[str]:
    """Convert split-file ML-estimate CSVs to 3-D NetCDF files.

    Parameters
    ----------
    csv_dir : str
        Path to the ``ml_estimates/`` directory containing the split CSVs.
    outdir : str, optional
        Output directory for NetCDF files.  Defaults to
        ``ml_estimates_nc/`` as a sibling of *csv_dir*.
    groups : list of str, optional
        Variable groups to convert (keys of :data:`VARIABLE_GROUPS`).
        Defaults to all groups.

    Returns
    -------
    list of str
        Paths to the NetCDF files that were written.
    """
    if os.path.isfile(csv_dir):
        csv_dir = os.path.dirname(csv_dir)

    if outdir is None:
        outdir = os.path.join(os.path.dirname(csv_dir), "ml_estimates_nc")
    os.makedirs(outdir, exist_ok=True)

    print(f"Input:  {csv_dir}")
    print(f"Output: {outdir}")

    # Determine grid from the first available CSV
    grid_csv = None
    for name in ["temperature", "coordinates", "melt"]:
        candidate = os.path.join(csv_dir, f"{name}.csv")
        if os.path.isfile(candidate):
            grid_csv = candidate
            break
    if grid_csv is None:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    lat, lon, z = _build_grid(grid_csv)

    groups = groups or list(VARIABLE_GROUPS.keys())
    written: List[str] = []
    t_total = time.time()

    for group_name in groups:
        if group_name not in VARIABLE_GROUPS:
            print(f"  Unknown group '{group_name}' — skipping")
            continue
        csv_path = os.path.join(csv_dir, f"{group_name}.csv")
        nc_path = _convert_one_group(
            csv_path, VARIABLE_GROUPS[group_name],
            lat, lon, z, outdir, group_name,
        )
        if nc_path:
            written.append(nc_path)

    elapsed = time.time() - t_total
    print(f"\nDone: {len(written)} NetCDF files in {elapsed:.0f}s")
    for p in written:
        print(f"  {p}")
    return written


# =========================================================================
# Data loading helpers
# =========================================================================

def load_from_netcdf(
    nc_dir: str,
    variables: Sequence[str],
) -> xr.Dataset:
    """Load variables from per-group NetCDF files into a single Dataset.

    Parameters
    ----------
    nc_dir : str
        Directory containing NetCDF files (temperature.nc, melt.nc, etc.).
    variables : sequence of str
        Variable names to load.

    Returns
    -------
    xr.Dataset
        Combined dataset with dimensions ``(z, lat, lon)``.
    """
    ds_parts = []
    loaded_vars: set = set()
    for nc_file in sorted(Path(nc_dir).glob("*.nc")):
        ds_file = xr.open_dataset(nc_file)
        needed = [v for v in variables if v in ds_file and v not in loaded_vars]
        if needed:
            ds_parts.append(ds_file[needed])
            loaded_vars.update(needed)
    if not ds_parts:
        raise FileNotFoundError(
            f"No variables {list(variables)} found in NetCDF files under {nc_dir}"
        )
    return xr.merge(ds_parts)


def load_from_csv(
    csv_path: str,
    method: Optional[str] = None,
) -> "pd.DataFrame":
    """Load ML estimates from split-file CSV directory or single CSV.

    Thin wrapper around :func:`~bayesian_fitting_py.io.load_ml_estimates`.
    """
    from .io import load_ml_estimates
    return load_ml_estimates(csv_path, method=method)


def build_3d_dataset(
    df: "pd.DataFrame",
    variables: Sequence[str],
) -> xr.Dataset:
    """Reshape a flat DataFrame into a 3-D xarray Dataset ``(z, lat, lon)``."""
    keep = ["lat", "lon", "z"] + [v for v in variables if v in df.columns]
    sub = df[keep].copy()
    ds = sub.set_index(["z", "lat", "lon"]).to_xarray()
    ds = ds.sortby(["z", "lat", "lon"])
    return ds


def find_data_source(
    search_dirs: Optional[Sequence[str]] = None,
    prefer_nc: bool = True,
):
    """Auto-detect NetCDF or CSV output from inversion directories.

    Parameters
    ----------
    search_dirs : sequence of str, optional
        Directories to search.  If *None*, searches the current working
        directory and then standard ``output/inversion_*/`` locations.
    prefer_nc : bool
        If *True* (default), prefer ``ml_estimates_nc/`` over CSVs.

    Returns
    -------
    (path, format) : tuple
        *path* is a directory path, *format* is ``'nc'`` or ``'csv'``.
        Both are *None* if nothing was found.
    """
    from .io import find_ml_estimates

    if search_dirs is None:
        search_dirs = [os.getcwd()]

    for d in search_dirs:
        d = Path(d)
        if prefer_nc:
            nc_dir = d / "ml_estimates_nc"
            if nc_dir.is_dir() and any(nc_dir.glob("*.nc")):
                return str(nc_dir), "nc"
        csv_found = find_ml_estimates(str(d))
        if csv_found:
            return csv_found, "csv"

    return None, None


# =========================================================================
# Global map plotting
# =========================================================================

def _get_style(var: str) -> Dict[str, Any]:
    """Return colour-map style dict; fall back to sensible defaults."""
    if var in VAR_STYLE:
        return VAR_STYLE[var]
    return {"cmap": "viridis", "series": None, "unit": "", "reverse": False, "label": var}


def plot_global_maps(
    ds: xr.Dataset,
    depths: Sequence[float],
    variables: Sequence[str],
    figure_width: float = 16.0,
    center_lon: float = 0.0,
    output_path: Optional[str] = None,
) -> Any:
    """Plot a grid of global Robinson-projection maps.

    Uses ``pygmt.Figure.subplot`` for layout management.

    Parameters
    ----------
    ds : xr.Dataset
        3-D dataset with coordinates ``(z, lat, lon)``.
    depths : sequence of float
        Depth slices to plot (one row each).
    variables : sequence of str
        Variable names to plot (one column each).
    figure_width : float
        Total figure width in inches (default 16).
    center_lon : float
        Centre longitude for the Robinson projection (default 0).
    output_path : str, optional
        If given, save the figure to this path (PNG at 300 DPI).

    Returns
    -------
    pygmt.Figure
    """
    try:
        import pygmt as gmt
    except ImportError:
        raise ImportError(
            "PyGMT is required for map plots.  Install with:\n"
            "    conda install -c conda-forge pygmt"
        )

    n_rows = len(depths)
    n_cols = len(variables)

    region = [-180, 180, -90, 90]
    proj_panel = f"N{center_lon}/?"

    # Robinson aspect ratio is roughly 2:1
    map_height = (figure_width / n_cols) * 0.6 * n_rows
    colorbar_width = (figure_width / n_cols) * 0.6

    fig = gmt.Figure()

    with fig.subplot(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(f"{figure_width}i", f"{map_height}i"),
        margins=["0.2c", "0.2c"],
        sharex="b",
        sharey="l",
        frame=["af"],
        autolabel=False,
    ):
        for ir, z in enumerate(depths):
            z_actual = float(ds.z.sel(z=z, method="nearest"))

            for ic, var in enumerate(variables):
                if var not in ds:
                    print(f"Warning: variable '{var}' not in dataset — skipping")
                    continue

                style = _get_style(var)
                grid = ds[var].sel(z=z_actual)

                with fig.set_panel(panel=[ir, ic]):
                    # Colour palette
                    kw_cpt = {"cmap": style["cmap"]}
                    if style["series"] is not None:
                        kw_cpt["series"] = style["series"]
                    if style["reverse"]:
                        kw_cpt["reverse"] = True
                    gmt.makecpt(**kw_cpt)

                    # Map content
                    with gmt.config(
                        FONT_ANNOT_PRIMARY="8p,Helvetica",
                        FONT_LABEL="8p,Helvetica-Bold",
                    ):
                        fig.basemap(
                            region=region, projection=proj_panel,
                            frame=["WSne", "af"],
                        )
                        fig.grdimage(grid, region=region, projection=proj_panel)
                        fig.coast(
                            region=region,
                            projection=proj_panel,
                            shorelines="0.3p,gray30",
                            area_thresh=10000,
                        )

                    # Top row: variable label
                    if ir == 0:
                        fig.text(
                            position="TC",
                            text=style["label"],
                            font="16p,Helvetica-Bold",
                            offset="0/0.8c",
                            no_clip=True,
                        )

                    # Left column: depth annotation
                    if ic == 0:
                        fig.text(
                            position="ML",
                            text=f"{z_actual:.0f} km",
                            font="16p,Helvetica-Bold",
                            angle=90,
                            offset="-1.0c/-1.0c",
                            no_clip=True,
                        )

                    # Colour bar below each panel (bottom row only)
                    with gmt.config(
                        FONT_ANNOT_PRIMARY="16p,Helvetica",
                        FONT_LABEL="16p,Helvetica",
                    ):
                        if ir == n_rows - 1:
                            cb_label = style["unit"] if style["unit"] else style["label"]
                            fig.colorbar(
                                position=f"JBC+o0c/0.3i+h+w{colorbar_width}i/0.15i",
                                frame=[f"x+l{cb_label}"],
                            )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        print(f"Figure saved to {output_path}")

    return fig


# =========================================================================
# CLI entry points
# =========================================================================

def _cli_to_netcdf():
    """CLI: convert ML-estimate CSVs to 3-D NetCDF files."""
    import argparse

    from .io import find_ml_estimates

    parser = argparse.ArgumentParser(
        description="Convert ML-estimate CSVs to 3-D NetCDF files",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to ml_estimates/ directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory for NetCDF files (default: ml_estimates_nc/ "
             "next to the CSV directory)",
    )
    parser.add_argument(
        "--groups", nargs="+", default=None,
        help="Variable groups to convert (default: all). "
             "Choices: " + ", ".join(VARIABLE_GROUPS.keys()),
    )
    args = parser.parse_args()

    est_path = args.csv
    if est_path is None:
        est_path = find_ml_estimates(os.getcwd())
    if est_path is None:
        parser.error(
            "Could not find ml_estimates/ directory.  Specify --csv explicitly."
        )

    csv_to_netcdf(est_path, outdir=args.outdir, groups=args.groups)


def _cli_plot_maps():
    """CLI: plot global Robinson-projection maps of inversion results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot global Robinson-projection maps of Bayesian "
                    "inversion results",
    )
    parser.add_argument(
        "--nc", default=None,
        help="Path to ml_estimates_nc/ directory with NetCDF files",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to ml_estimates/ directory or single CSV file",
    )
    parser.add_argument(
        "--depth", "-z", nargs="+", type=float, default=[100.0],
        help="Depth(s) to plot (km); one row per depth",
    )
    parser.add_argument(
        "--vars", nargs="+", default=DEFAULT_PLOT_VARS,
        help="Variable names to plot (columns)",
    )
    parser.add_argument(
        "--method", default=None,
        help="Anelastic method to filter (only used with CSV input)",
    )
    parser.add_argument(
        "--width", type=float, default=16.0,
        help="Total figure width in inches (default: 16)",
    )
    parser.add_argument(
        "--center-lon", type=float, default=0.0,
        help="Centre longitude for Robinson projection (default: 0)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output figure path (default: auto-named in current directory)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display the figure interactively",
    )
    args = parser.parse_args()

    # Resolve data source
    if args.nc:
        data_path, fmt = args.nc, "nc"
    elif args.csv:
        data_path, fmt = args.csv, "csv"
    else:
        data_path, fmt = find_data_source(prefer_nc=True)
    if data_path is None:
        parser.error(
            "Could not find ml_estimates_nc/ or ml_estimates/.  "
            "Specify --nc or --csv explicitly."
        )
    print(f"Loading results from {data_path} (format: {fmt})")

    if fmt == "nc":
        ds = load_from_netcdf(data_path, args.vars)
    else:
        df = load_from_csv(data_path, method=args.method)
        if df.empty:
            print("No data after filtering — nothing to plot.")
            return
        ds = build_3d_dataset(df, args.vars)
        del df

    avail_z = sorted(float(v) for v in ds.z.values)
    if len(avail_z) > 8:
        print(f"Available depths ({len(avail_z)}): "
              f"{avail_z[:5]} ... {avail_z[-3:]}")
    else:
        print(f"Available depths ({len(avail_z)}): {avail_z}")

    output_path = args.output
    if output_path is None:
        depth_tag = "_".join(f"{d:.0f}" for d in args.depth)
        output_path = f"global_maps_z{depth_tag}km.png"

    fig = plot_global_maps(
        ds,
        depths=args.depth,
        variables=args.vars,
        figure_width=args.width,
        center_lon=args.center_lon,
        output_path=output_path,
    )

    if args.show:
        fig.show()
