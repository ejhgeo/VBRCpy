"""
Data processing functions for seismic observations.

Handles loading and processing of seismic velocity (Vs), 
attenuation (Q), and LAB depth observations.

Translated from MATLAB Projects/bayesian_fitting/functions/process_SeismicModels.m
"""

import numpy as np
from scipy.io import loadmat
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field


@dataclass
class Location:
    """Location specification for extracting seismic observations."""
    lat: float  # latitude [degrees North]
    lon: float  # longitude [degrees East] (positive values)
    z_min: float  # minimum depth for observation range [km]
    z_max: float  # maximum depth for observation range [km]
    smooth_rad: float = 0.5  # radius to smooth over observations [degrees]


@dataclass
class SeismicModelData:
    """
    Container for seismic model data loaded from a file.
    
    Used when locations and seismic observations are read together
    from a single data source (CSV, MAT, or NetCDF file).
    """
    # Location information
    locations: List[Tuple[float, float]]  # List of (lat, lon) tuples
    names: List[str]  # Location names
    z_ranges: List[Tuple[float, float]]  # (z_min, z_max) for each point
    depths: List[float]  # Exact depth at each point
    
    # Seismic observations (same length as locations)
    Vs: Optional[np.ndarray] = None  # Shear velocity [km/s]
    Vs_error: Optional[np.ndarray] = None  # Vs uncertainty
    Q: Optional[np.ndarray] = None  # Quality factor
    Q_error: Optional[np.ndarray] = None  # Q uncertainty
    
    def __len__(self) -> int:
        return len(self.locations)
    
    def has_vs(self) -> bool:
        return self.Vs is not None and len(self.Vs) > 0
    
    def has_q(self) -> bool:
        return self.Q is not None and len(self.Q) > 0


def load_seismic_model(model_file: str) -> Dict[str, Any]:
    """
    Load a seismic model from a .mat file.

    Parameters
    ----------
    model_file : str
        Path to the .mat file containing the seismic model

    Returns
    -------
    dict
        Model dictionary with Latitude, Longitude, Depth (if available),
        observation values, and Error (if available)
    """
    mat_data = loadmat(model_file, squeeze_me=True, struct_as_record=False)
    
    # Find the model variable (ignore metadata keys)
    model_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if len(model_keys) == 0:
        raise ValueError(f"No model data found in {model_file}")
    
    model_obj = mat_data[model_keys[0]]
    
    # Convert MATLAB struct to dictionary
    model = {}
    for field in dir(model_obj):
        if not field.startswith('_'):
            value = getattr(model_obj, field)
            if isinstance(value, np.ndarray):
                model[field] = value
            elif np.isscalar(value):
                model[field] = np.array([value])
            else:
                model[field] = value
    
    return model


def check_errors(model: Dict[str, Any], obs_name: str) -> Dict[str, Any]:
    """
    Check if an 'Error' field exists in the model. If not, create one
    with a default constant value.

    Parameters
    ----------
    model : dict
        Seismic model dictionary
    obs_name : str
        Observation name ('Vs', 'Q', 'LAB_Depth')

    Returns
    -------
    dict
        Model with 'Error' field populated
    """
    if 'Error' in model:
        return model

    # Default error values
    default_errors = {
        'Vs': 0.05,
        'LAB_Depth': 5.0,
        'Q': 10.0,
    }
    
    constant_error = default_errors.get(obs_name, 0.1)
    model['Error'] = constant_error * np.ones(model[obs_name].shape)
    
    return model


def check_overlap(model: Dict[str, Any], location: Location) -> Tuple[Dict[str, Any], bool]:
    """
    Check if the model contains the specified location.
    Also convert negative longitudes to positive.

    Parameters
    ----------
    model : dict
        Seismic model dictionary
    location : Location
        Location specification

    Returns
    -------
    tuple
        (model with corrected longitudes, success flag)
    """
    # Convert negative longitudes to positive
    model['Longitude'] = np.where(
        model['Longitude'] < 0,
        model['Longitude'] + 360,
        model['Longitude']
    )
    
    # Global model limits
    lat_min, lat_max = model['Latitude'].min(), model['Latitude'].max()
    lon_min, lon_max = model['Longitude'].min(), model['Longitude'].max()
    
    # Check latitude overlap
    lat_ok = (
        location.lat + location.smooth_rad >= lat_min and
        location.lat - location.smooth_rad <= lat_max
    )
    
    # Check longitude overlap
    lon_ok = (
        location.lon + location.smooth_rad >= lon_min and
        location.lon - location.smooth_rad <= lon_max
    )
    
    if not (lat_ok and lon_ok):
        print("Your coordinates are not within the model lat/lon bounds")
        return model, False
    
    # Check depth overlap if available
    if 'Depth' in model:
        depth = np.asarray(model['Depth']).flatten()
        if not (location.z_min >= depth[0] and location.z_max <= depth[-1]):
            print(f"Your coordinates are not within the model depth bounds: {depth[[0, -1]]}")
            return model, False
    
    return model, True


def limit_by_coords(
    model: Dict[str, Any], field_name: str, location: Location
) -> Dict[str, Any]:
    """
    Extract the slice of the model defined by the location box.

    Parameters
    ----------
    model : dict
        Seismic model dictionary
    field_name : str
        Name of the observation field
    location : Location
        Location specification

    Returns
    -------
    dict
        Model limited to the specified region
    """
    d_deg = location.smooth_rad
    lat, lon = location.lat, location.lon
    
    # Create lat/lon masks
    lat_mask = (
        (model['Latitude'] >= lat - d_deg) &
        (model['Latitude'] <= lat + d_deg)
    )
    lon_mask = (
        (model['Longitude'] >= lon - d_deg) &
        (model['Longitude'] <= lon + d_deg)
    )
    
    # Apply masks
    model['Longitude'] = model['Longitude'][lon_mask]
    model['Latitude'] = model['Latitude'][lat_mask]
    
    # Handle different array dimensions
    obs_data = model[field_name]
    error_data = model['Error']
    
    if obs_data.ndim == 2:
        # 2D data (no depth dimension)
        model[field_name] = obs_data[np.ix_(lat_mask, lon_mask)]
        model['Error'] = error_data[np.ix_(lat_mask, lon_mask)]
    elif obs_data.ndim == 3:
        # 3D data with depth
        model[field_name] = obs_data[np.ix_(lat_mask, lon_mask)]
        if error_data.ndim == 3:
            model['Error'] = error_data[np.ix_(lat_mask, lon_mask)]
    
    return model


def find_median(model: Dict[str, Any], field_name: str) -> np.ndarray:
    """
    Find the median value for a field, optionally as a function of depth.

    Parameters
    ----------
    model : dict
        Seismic model dictionary
    field_name : str
        Name of the field to compute median for

    Returns
    -------
    np.ndarray
        Median value(s)
    """
    all_vals = model[field_name]
    
    if 'Depth' in model:
        depth = np.asarray(model['Depth']).flatten()
        n_z = len(depth)
        
        # Reshape to (n_spatial, n_depth) and compute median
        if all_vals.ndim >= 2:
            reshaped = all_vals.reshape(-1, n_z)
            median_val = np.nanmedian(reshaped, axis=0)
        else:
            median_val = np.nanmedian(all_vals)
    else:
        median_val = np.nanmedian(all_vals.flatten())
    
    return np.atleast_1d(median_val)


def find_lateral_error(model: Dict[str, Any], field_name: str) -> np.ndarray:
    """
    Calculate uncertainty estimate from the standard deviation
    of values within the observation box.

    Parameters
    ----------
    model : dict
        Seismic model dictionary
    field_name : str
        Name of the field

    Returns
    -------
    np.ndarray
        Lateral error (standard deviation)
    """
    all_vals = model[field_name]
    
    if 'Depth' in model:
        depth = np.asarray(model['Depth']).flatten()
        n_z = len(depth)
        
        if all_vals.ndim >= 2:
            reshaped = all_vals.reshape(-1, n_z)
            lateral_error = np.nanstd(reshaped, axis=0)
        else:
            lateral_error = np.nanstd(all_vals)
    else:
        lateral_error = np.nanstd(all_vals.flatten())
    
    return np.atleast_1d(lateral_error)


def limit_by_depth(
    obs_value: np.ndarray,
    obs_error: np.ndarray,
    depth: np.ndarray,
    location: Location,
) -> Tuple[float, float]:
    """
    Extract median values from the desired depth range.

    Parameters
    ----------
    obs_value : np.ndarray
        Observation values as a function of depth
    obs_error : np.ndarray
        Error values as a function of depth
    depth : np.ndarray
        Depth vector [km]
    location : Location
        Location specification with z_min and z_max

    Returns
    -------
    tuple
        (median obs value, median obs error) for depth range
    """
    depth = np.asarray(depth).flatten()
    depth_mask = (depth >= location.z_min) & (depth <= location.z_max)
    
    obs_val = np.nanmedian(np.asarray(obs_value).flatten()[depth_mask])
    obs_err = np.nanmedian(np.asarray(obs_error).flatten()[depth_mask])
    
    return float(obs_val), float(obs_err)


def process_seismic_models(
    obs_name: str,
    location: Location,
    model_file: str,
    ifplot: bool = False,
) -> Tuple[float, float]:
    """
    Load seismic observations and return the observed value and error
    at the specified location.

    Parameters
    ----------
    obs_name : str
        Observation name ('Vs', 'Q', 'LAB')
    location : Location
        Location specification
    model_file : str
        Path to the .mat file containing observations
    ifplot : bool, optional
        If True, plot the observation as a function of depth

    Returns
    -------
    tuple
        (obs_value, obs_error) at the specified location
    """
    # Load the model
    model = load_seismic_model(model_file)
    
    # Check for errors and add defaults if missing
    model = check_errors(model, obs_name)
    
    # Check coordinate overlap
    model, success = check_overlap(model, location)
    if not success:
        raise ValueError("Location not within model bounds")
    
    # Limit to the coordinate region
    model = limit_by_coords(model, obs_name, location)
    
    # Find median and error
    obs_value_z = find_median(model, obs_name)
    median_error_z = find_median(model, 'Error')
    lateral_error_z = find_lateral_error(model, obs_name)
    
    # Take maximum of reported error and lateral variability
    obs_error_z = np.maximum(median_error_z, lateral_error_z)
    
    # Average over depth if depth dimension exists
    if 'Depth' in model:
        depth = np.asarray(model['Depth']).flatten()
        obs_value, obs_error = limit_by_depth(
            obs_value_z, obs_error_z, depth, location
        )
        
        if ifplot:
            from .plotting import plot_seismic_obs
            plot_seismic_obs(
                obs_value_z, obs_error_z, obs_name,
                depth, location, obs_value, obs_error
            )
    else:
        obs_value = float(obs_value_z.flatten()[0])
        obs_error = float(obs_error_z.flatten()[0])
    
    return obs_value, obs_error


def extract_calculated_values_in_depth_range(
    sweep: Dict[str, Any],
    obs_name: str,
    anelastic_method: str,
    depth_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each element in the parameter sweep, find the calculated mean
    Vs and mean Q in the relevant depth range.

    Parameters
    ----------
    sweep : dict
        Parameter sweep dictionary containing:
        - z: depth vector [m]
        - Box: 3D array of calculated values (T, phi, gs dimensions)
    obs_name : str
        Observation name ('Vs' or 'Q')
    anelastic_method : str
        Anelastic method name (e.g., 'xfit_premelt')
    depth_range : tuple
        (min_depth, max_depth) in km

    Returns
    -------
    tuple
        (mean_values array, depth indices)
    """
    z = sweep['z']
    depth_range_m = np.array(depth_range) * 1e3  # Convert to meters
    
    z_inds = np.where(
        (z >= depth_range_m[0]) & (z <= depth_range_m[1])
    )[0]
    
    # Get the field name in the Box
    field_name = f'mean{obs_name}'
    
    # Box is indexed as [anelastic_method][field_name] with shape (nT, nphi, ngs, nz)
    box = sweep['Box']
    data = box[anelastic_method][field_name]
    
    # Vectorized mean over the matching depth indices
    mean_val = np.mean(data[:, :, :, z_inds], axis=3)
    
    return mean_val, z_inds


def check_file_exists(filenames: Dict[str, str], field: str) -> bool:
    """
    Check if a file field exists and the file path is valid.

    Parameters
    ----------
    filenames : dict
        Dictionary with file paths
    field : str
        Field name to check

    Returns
    -------
    bool
        True if field exists and file exists
    """
    import os
    
    if field not in filenames:
        return False
    
    if os.path.exists(filenames[field]):
        return True
    else:
        print(f"{filenames[field]} does not exist.")
        return False


# =============================================================================
# Location Loading Functions
# =============================================================================

def load_locations_from_file(
    filepath: str,
) -> Tuple[List[Tuple[float, float]], List[str], List[Tuple[float, float]]]:
    """
    Load locations from a delimited text file (CSV, TSV, or space-separated).

    File format: lon, lat, [name], z_min, z_max
    Lines starting with # are treated as comments.
    
    Supports comma, tab, or space/whitespace delimited files. The delimiter
    is auto-detected from the file content.

    Parameters
    ----------
    filepath : str
        Path to the location file

    Returns
    -------
    tuple
        (locations, names, z_ranges) where:
        - locations: list of (lat, lon) tuples
        - names: list of location names
        - z_ranges: list of (z_min, z_max) tuples
    """
    locations = []
    names = []
    z_ranges = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Detect delimiter: comma, tab, or whitespace
    first_data_line = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            first_data_line = line
            break
    
    if first_data_line is None:
        raise ValueError(f"No data found in {filepath}")
    
    # Auto-detect delimiter: try comma, then tab, then whitespace
    if ',' in first_data_line:
        delimiter = ','
    elif '\t' in first_data_line:
        delimiter = '\t'
    else:
        delimiter = None  # whitespace
    
    point_idx = 0
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if delimiter is not None:
            parts = [p.strip() for p in line.split(delimiter)]
        else:
            parts = line.split()  # split on whitespace
        
        # Determine format based on number of columns
        try:
            if len(parts) == 4:
                # lon, lat, z_min, z_max (no name)
                lon, lat, z_min, z_max = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                name = f"point_{point_idx}"
            elif len(parts) == 5:
                # lon, lat, name, z_min, z_max
                lon, lat = float(parts[0]), float(parts[1])
                name = parts[2]
                z_min, z_max = float(parts[3]), float(parts[4])
            else:
                raise ValueError(f"Line {line_num}: expected 4 or 5 columns, got {len(parts)}")
            
            locations.append((lat, lon))
            names.append(name)
            z_ranges.append((z_min, z_max))
            point_idx += 1
            
        except ValueError as e:
            print(f"Warning: skipping line {line_num}: {e}")
            continue
    
    if not locations:
        raise ValueError(f"No valid locations found in {filepath}")
    
    print(f"Loaded {len(locations)} locations from {filepath}")
    return locations, names, z_ranges


def load_locations_from_tomography(
    model_file: str,
    z_range: Optional[Tuple[float, float]],
    subsample: int = 1,
) -> Tuple[List[Tuple[float, float]], List[str], List[Tuple[float, float]]]:
    """
    Extract all grid points from a tomography model file.
    
    For 3D tomography models (with Depth field), each (lat, lon, depth) point
    is treated separately. The z_range for each point is automatically set to
    span half the depth spacing above and below, so the inversion uses the
    exact depth level from the tomography.
    
    For 2D models (horizontal slices), all points share the same z_range.

    Parameters
    ----------
    model_file : str
        Path to the tomography model .mat file
    z_range : tuple or None
        (z_min, z_max) depth range. For 3D models, this filters which depths
        to include. For 2D models, this is the depth range for all points.
        If None for 3D, all depths are used.
    subsample : int
        Use every Nth point (1 = all points, 2 = every other point, etc.)

    Returns
    -------
    tuple
        (locations, names, z_ranges) where:
        - locations: list of (lat, lon) tuples
        - names: list of location names (formatted as "lat_lon" or "lat_lon_depthkm")
        - z_ranges: list of (z_min, z_max) tuples for each point
    """
    mat_data = loadmat(model_file, squeeze_me=True, struct_as_record=False)
    
    # Find the model variable
    model_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if len(model_keys) == 0:
        raise ValueError(f"No model data found in {model_file}")
    
    model_obj = mat_data[model_keys[0]]
    
    # Extract lat/lon arrays
    if hasattr(model_obj, 'Latitude'):
        lats = np.atleast_1d(getattr(model_obj, 'Latitude'))
        lons = np.atleast_1d(getattr(model_obj, 'Longitude'))
    else:
        raise ValueError(f"Model file {model_file} does not contain Latitude/Longitude fields")
    
    # Check for depth/z array - indicates 3D tomography
    depths = None
    if hasattr(model_obj, 'Depth'):
        depths = np.atleast_1d(getattr(model_obj, 'Depth'))
    elif hasattr(model_obj, 'depth'):
        depths = np.atleast_1d(getattr(model_obj, 'depth'))
    elif hasattr(model_obj, 'Z'):
        depths = np.atleast_1d(getattr(model_obj, 'Z'))
    elif hasattr(model_obj, 'z'):
        depths = np.atleast_1d(getattr(model_obj, 'z'))
    
    # Determine if this is a 3D model with explicit depths
    is_3d_model = depths is not None and len(depths) > 1
    
    if is_3d_model:
        # This is a true 3D tomography model
        return _load_3d_tomography(
            model_file, lats, lons, depths, z_range, subsample
        )
    else:
        # This is a 2D model (horizontal slice) - use fixed z_range
        return _load_2d_tomography(
            model_file, lats, lons, z_range, subsample
        )


def _load_3d_tomography(
    model_file: str,
    lats: np.ndarray,
    lons: np.ndarray,
    depths: np.ndarray,
    z_range: Optional[Tuple[float, float]],
    subsample: int,
) -> Tuple[List[Tuple[float, float]], List[str], List[Tuple[float, float]]]:
    """
    Load locations from a 3D tomography model with explicit depths.
    
    Each grid point in the 3D model gets its own z_range based on the 
    depth spacing in the model.
    """
    # Get unique coordinates
    unique_lats = np.unique(lats)
    unique_lons = np.unique(lons)
    unique_depths = np.sort(np.unique(depths))
    
    # Calculate depth spacing for z_range
    if len(unique_depths) > 1:
        # Use half the spacing to adjacent depths as the z_range
        depth_diffs = np.diff(unique_depths)
        # Pad to get spacing for first and last points
        spacing_below = np.concatenate([[depth_diffs[0]], depth_diffs])
        spacing_above = np.concatenate([depth_diffs, [depth_diffs[-1]]])
    else:
        # Single depth - use a small range
        spacing_below = np.array([5.0])
        spacing_above = np.array([5.0])
    
    # Filter by z_range if specified
    if z_range is not None:
        z_mask = (unique_depths >= z_range[0]) & (unique_depths <= z_range[1])
        unique_depths = unique_depths[z_mask]
        spacing_below = spacing_below[z_mask]
        spacing_above = spacing_above[z_mask]
    
    if len(unique_depths) == 0:
        raise ValueError(f"No depths found in range {z_range} km. "
                        f"Model depth range: {np.unique(depths).min():.1f} to {np.unique(depths).max():.1f} km")
    
    # Subsample if requested
    if subsample > 1:
        unique_lats = unique_lats[::subsample]
        unique_lons = unique_lons[::subsample]
        unique_depths = unique_depths[::subsample]
        spacing_below = spacing_below[::subsample]
        spacing_above = spacing_above[::subsample]
    
    # Build depth-to-spacing lookup
    depth_spacing = {}
    for i, d in enumerate(unique_depths):
        depth_spacing[d] = (spacing_below[i], spacing_above[i])
    
    # Generate all (lat, lon, depth) combinations
    locations = []
    names = []
    z_ranges = []
    
    for depth_idx, depth in enumerate(unique_depths):
        half_below = spacing_below[depth_idx] / 2.0
        half_above = spacing_above[depth_idx] / 2.0
        z_min = depth - half_below
        z_max = depth + half_above
        
        for lat in unique_lats:
            for lon in unique_lons:
                locations.append((float(lat), float(lon)))
                names.append(str(len(locations)))  # Simple index
                z_ranges.append((z_min, z_max))
    
    print(f"Extracted {len(locations)} 3D grid points from {model_file}")
    print(f"  Latitude range: {unique_lats.min():.2f} to {unique_lats.max():.2f} ({len(unique_lats)} points)")
    print(f"  Longitude range: {unique_lons.min():.2f} to {unique_lons.max():.2f} ({len(unique_lons)} points)")
    print(f"  Depth range: {unique_depths.min():.1f} to {unique_depths.max():.1f} km ({len(unique_depths)} levels)")
    
    return locations, names, z_ranges


def _load_2d_tomography(
    model_file: str,
    lats: np.ndarray,
    lons: np.ndarray,
    z_range: Tuple[float, float],
    subsample: int,
) -> Tuple[List[Tuple[float, float]], List[str], List[Tuple[float, float]]]:
    """
    Load locations from a 2D tomography model (horizontal slice).
    
    Uses a fixed z_range for all points.
    """
    if z_range is None:
        raise ValueError("z_range must be specified for 2D tomography models. "
                        "Use tomo_z_range config option or --tomo-z-range CLI argument.")
    
    # Get unique lat/lon values
    if lats.ndim == 1 and lons.ndim == 1:
        if len(lats) == len(lons):
            unique_lats = np.unique(lats)
            unique_lons = np.unique(lons)
            
            if len(unique_lats) * len(unique_lons) == len(lats):
                grid_lats = unique_lats
                grid_lons = unique_lons
            else:
                grid_lats = lats
                grid_lons = lons
        else:
            grid_lats = lats
            grid_lons = lons
    else:
        grid_lats = np.unique(lats)
        grid_lons = np.unique(lons)
    
    # Subsample if requested
    if subsample > 1:
        grid_lats = grid_lats[::subsample]
        grid_lons = grid_lons[::subsample]
    
    # Generate all combinations
    locations = []
    names = []
    z_ranges = []
    
    for lat in grid_lats:
        for lon in grid_lons:
            locations.append((float(lat), float(lon)))
            names.append(str(len(locations)))  # Simple index
            z_ranges.append(z_range)
    
    print(f"Extracted {len(locations)} 2D grid points from {model_file}")
    print(f"  Latitude range: {grid_lats.min():.2f} to {grid_lats.max():.2f}")
    print(f"  Longitude range: {grid_lons.min():.2f} to {grid_lons.max():.2f}")
    print(f"  Depth range: {z_range[0]} to {z_range[1]} km (fixed for all points)")
    
    return locations, names, z_ranges


# =============================================================================
# Seismic Model Loading Functions (locations + observations together)
# =============================================================================

def _calculate_depth_ranges(depths: np.ndarray) -> Dict[float, Tuple[float, float]]:
    """
    Calculate z_ranges for each depth based on spacing to adjacent depths.
    
    Uses half the spacing to the nearest neighbor on each side.
    
    Returns
    -------
    dict
        Mapping from depth value to (z_min, z_max) tuple
    """
    unique_depths = np.sort(np.unique(depths))
    n = len(unique_depths)
    
    if n == 1:
        # Single depth - use a small range
        d = unique_depths[0]
        return {d: (d - 2.5, d + 2.5)}
    
    # Calculate spacing
    depth_diffs = np.diff(unique_depths)
    spacing_below = np.concatenate([[depth_diffs[0]], depth_diffs])
    spacing_above = np.concatenate([depth_diffs, [depth_diffs[-1]]])
    
    # Build lookup from depth to z_range
    depth_to_range = {}
    for i, d in enumerate(unique_depths):
        z_min = d - spacing_below[i] / 2.0
        z_max = d + spacing_above[i] / 2.0
        depth_to_range[d] = (z_min, z_max)
    
    return depth_to_range


def load_seismic_model_from_csv(
    filepath: str,
    default_vs_error: float = 0.05,
    default_q_error: float = 10.0,
    z_range: Optional[Tuple[float, float]] = None,
    subsample: int = 1,
) -> SeismicModelData:
    """
    Load seismic model data from a delimited text file (CSV, TSV, or space-separated).
    
    Expected columns: lon, lat, depth, Vs, [Q], [Vs_error], [Q_error]
    The header row is used to identify columns. Column names are case-insensitive
    and can include variations like 'z', 'zvalue', 'depth_km' for depth.
    
    Supports comma, tab, or space/whitespace delimited files. The delimiter
    is auto-detected from the file content.
    
    Parameters
    ----------
    filepath : str
        Path to the delimited text file
    default_vs_error : float
        Default Vs error if not provided in file [km/s]
    default_q_error : float
        Default Q error if not provided in file
    z_range : tuple, optional
        (z_min, z_max) to filter depths. If None, use all depths.
    subsample : int
        Use every Nth point (1 = all points)
        
    Returns
    -------
    SeismicModelData
        Container with locations and seismic observations
    """
    import pandas as pd
    
    # Auto-detect delimiter by reading first few non-comment lines
    with open(filepath, 'r') as f:
        lines = []
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                lines.append(stripped)
                if len(lines) >= 3:  # Get header + a couple data lines
                    break
    
    if not lines:
        raise ValueError(f"Could not find any data lines in {filepath}")
    
    # Detect delimiter by checking which one produces consistent column counts
    # Try comma, tab, then whitespace
    def count_cols(line, sep):
        if sep == r'\s+':
            return len(line.split())
        else:
            return len(line.split(sep))
    
    best_sep = None
    best_name = None
    
    for sep, name in [(',', 'comma'), ('\t', 'tab'), (r'\s+', 'whitespace')]:
        col_counts = [count_cols(line, sep) for line in lines]
        # Good delimiter should give >= 3 columns and be consistent
        if all(c >= 3 for c in col_counts) and len(set(col_counts)) == 1:
            best_sep = sep
            best_name = name
            break
    
    if best_sep is None:
        # Fallback: try whitespace as default
        best_sep = r'\s+'
        best_name = 'whitespace'
    
    sep = best_sep
    delim_name = best_name
    
    # Read file with detected delimiter
    df = pd.read_csv(filepath, comment='#', sep=sep, engine='python' if sep == r'\s+' else 'c')
    
    print(f"Reading {filepath} with {delim_name}-delimited format")
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Map common column name variations
    col_map = {
        'longitude': 'lon', 'long': 'lon', 'x': 'lon',
        'latitude': 'lat', 'y': 'lat',
        'z': 'depth', 'zvalue': 'depth', 'depth_km': 'depth', 'z_km': 'depth',
        'vs_km_s': 'vs', 'vs_kms': 'vs', 'shear_velocity': 'vs',
        'qinv': 'q', 'quality_factor': 'q',
        'vs_err': 'vs_error', 'sigma_vs': 'vs_error', 'vs_sigma': 'vs_error',
        'q_err': 'q_error', 'sigma_q': 'q_error', 'q_sigma': 'q_error',
    }
    df.rename(columns=col_map, inplace=True)
    
    # Validate required columns
    required = ['lon', 'lat', 'depth']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    
    if 'vs' not in df.columns and 'q' not in df.columns:
        raise ValueError("CSV must contain at least 'Vs' or 'Q' column")
    
    # Filter by z_range if specified
    if z_range is not None:
        mask = (df['depth'] >= z_range[0]) & (df['depth'] <= z_range[1])
        df = df[mask].copy()
        if len(df) == 0:
            raise ValueError(f"No data in depth range {z_range}")
    
    # Subsample if requested
    if subsample > 1:
        df = df.iloc[::subsample].copy()
    
    # Calculate z_ranges based on depth spacing
    depth_to_range = _calculate_depth_ranges(df['depth'].values)
    
    # Build output data
    locations = []
    names = []
    z_ranges = []
    depths = []
    vs_list = []
    vs_err_list = []
    q_list = []
    q_err_list = []
    
    has_vs = 'vs' in df.columns
    has_q = 'q' in df.columns
    has_vs_err = 'vs_error' in df.columns
    has_q_err = 'q_error' in df.columns
    
    for idx, (_, row) in enumerate(df.iterrows()):
        lat, lon, depth = row['lat'], row['lon'], row['depth']
        
        locations.append((float(lat), float(lon)))
        names.append(str(idx + 1))  # Simple index (1-based)
        depths.append(float(depth))
        
        # Find z_range for this depth
        z_range_pt = depth_to_range.get(depth, (depth - 2.5, depth + 2.5))
        z_ranges.append(z_range_pt)
        
        if has_vs:
            vs_list.append(float(row['vs']))
            vs_err_list.append(float(row['vs_error']) if has_vs_err else default_vs_error)
        
        if has_q:
            q_list.append(float(row['q']))
            q_err_list.append(float(row['q_error']) if has_q_err else default_q_error)
    
    print(f"Loaded {len(locations)} points from {filepath}")
    print(f"  Depth range: {min(depths):.1f} to {max(depths):.1f} km")
    if has_vs:
        print(f"  Vs range: {min(vs_list):.3f} to {max(vs_list):.3f} km/s")
    if has_q:
        print(f"  Q range: {min(q_list):.1f} to {max(q_list):.1f}")
    
    return SeismicModelData(
        locations=locations,
        names=names,
        z_ranges=z_ranges,
        depths=depths,
        Vs=np.array(vs_list) if has_vs else None,
        Vs_error=np.array(vs_err_list) if has_vs else None,
        Q=np.array(q_list) if has_q else None,
        Q_error=np.array(q_err_list) if has_q else None,
    )


def load_seismic_model_from_mat(
    filepath: str,
    default_vs_error: float = 0.05,
    default_q_error: float = 10.0,
    z_range: Optional[Tuple[float, float]] = None,
    subsample: int = 1,
) -> SeismicModelData:
    """
    Load seismic model data from a .mat file with depth information.
    
    The .mat file should contain a structure with fields:
    - Latitude, Longitude, Depth (required)
    - Vs or Vsv (shear velocity)
    - Q or Qmu (quality factor)
    - Error or Vs_error, Q_error (optional)
    
    Parameters
    ----------
    filepath : str
        Path to the .mat file
    default_vs_error : float
        Default Vs error if not provided [km/s]
    default_q_error : float
        Default Q error if not provided
    z_range : tuple, optional
        (z_min, z_max) to filter depths. If None, use all depths.
    subsample : int
        Use every Nth point (1 = all points)
        
    Returns
    -------
    SeismicModelData
        Container with locations and seismic observations
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray is required for mat_model mode. Install with: pip install xarray")
    
    mat_data = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    
    # Find the model variable
    model_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    if len(model_keys) == 0:
        raise ValueError(f"No model data found in {filepath}")
    
    model_obj = mat_data[model_keys[0]]
    
    # Extract coordinates
    lats = _get_field(model_obj, ['Latitude', 'latitude', 'lat', 'Lat'])
    lons = _get_field(model_obj, ['Longitude', 'longitude', 'lon', 'Lon'])
    depths = _get_field(model_obj, ['Depth', 'depth', 'Z', 'z'])
    
    if lats is None or lons is None:
        raise ValueError(f"Model file must contain Latitude and Longitude fields")
    if depths is None:
        raise ValueError(f"Model file must contain Depth field for this mode")
    
    lats = np.atleast_1d(lats).flatten()
    lons = np.atleast_1d(lons).flatten()
    depths = np.atleast_1d(depths).flatten()
    
    # Extract Vs and Q
    vs_data = _get_field(model_obj, ['Vs', 'Vsv', 'vs', 'V', 'Velocity'])
    q_data = _get_field(model_obj, ['Q', 'Qmu', 'q', 'Quality'])
    
    if vs_data is None and q_data is None:
        raise ValueError(f"Model file must contain Vs or Q data")
    
    if vs_data is not None:
        vs_data = np.atleast_1d(vs_data)
    if q_data is not None:
        q_data = np.atleast_1d(q_data)
    
    # Determine if data is gridded (3D) or flattened (1D)
    # If Vs/Q is multi-dimensional, treat as grid with lat/lon/depth axes
    ref_data = vs_data if vs_data is not None else q_data
    
    if ref_data.ndim > 1:
        # Gridded data - convert to xarray Dataset and process like NetCDF
        # Determine dimension order by matching array shape to coordinate lengths
        shape = ref_data.shape
        dim_sizes = {'lat': len(lats), 'lon': len(lons), 'depth': len(depths)}
        
        # Find which axis corresponds to which dimension
        dim_order = []
        used_dims = set()
        for i, s in enumerate(shape):
            for dim_name, dim_len in dim_sizes.items():
                if dim_name not in used_dims and s == dim_len:
                    dim_order.append(dim_name)
                    used_dims.add(dim_name)
                    break
            else:
                # If no match found, try to infer
                raise ValueError(
                    f"Cannot match data shape {shape} to coordinate lengths: "
                    f"lat={len(lats)}, lon={len(lons)}, depth={len(depths)}"
                )
        
        # Create xarray Dataset
        coords = {'lat': lats, 'lon': lons, 'depth': depths}
        data_vars = {}
        if vs_data is not None:
            data_vars['Vs'] = (dim_order, vs_data)
        if q_data is not None:
            data_vars['Q'] = (dim_order, q_data)
        
        ds = xr.Dataset(data_vars, coords=coords)
        
        # Filter by z_range
        if z_range is not None:
            depth_mask = (ds['depth'].values >= z_range[0]) & (ds['depth'].values <= z_range[1])
            ds = ds.isel(depth=depth_mask)
            depths = ds['depth'].values
        
        if len(depths) == 0:
            raise ValueError(f"No data in depth range {z_range}")
        
        # Subsample
        if subsample > 1:
            ds = ds.isel(
                lat=slice(None, None, subsample),
                lon=slice(None, None, subsample),
                depth=slice(None, None, subsample),
            )
            lats = ds['lat'].values
            lons = ds['lon'].values
            depths = ds['depth'].values
        
        # Get updated data arrays
        vs_arr = ds['Vs'].values if 'Vs' in ds else None
        q_arr = ds['Q'].values if 'Q' in ds else None
        
        # Calculate z_ranges based on depth spacing
        depth_to_range = _calculate_depth_ranges(depths)
        
        # Build flattened output (iterate over all grid points)
        locations = []
        names = []
        z_ranges_list = []
        depth_list = []
        vs_list = []
        q_list = []
        
        # Get dimension indices
        updated_dim_order = list(ds['Vs'].dims if 'Vs' in ds else ds['Q'].dims)
        lat_idx = updated_dim_order.index('lat')
        lon_idx = updated_dim_order.index('lon')
        depth_idx = updated_dim_order.index('depth')
        
        for k, depth in enumerate(depths):
            z_range_pt = depth_to_range.get(depth, (depth - 2.5, depth + 2.5))
            
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    locations.append((float(lat), float(lon)))
                    names.append(str(len(locations)))  # Simple index
                    depth_list.append(float(depth))
                    z_ranges_list.append(z_range_pt)
                    
                    # Extract value based on dimension order
                    idx = [None, None, None]
                    idx[lat_idx] = i
                    idx[lon_idx] = j
                    idx[depth_idx] = k
                    
                    if vs_arr is not None:
                        vs_list.append(float(vs_arr[tuple(idx)]))
                    if q_arr is not None:
                        q_list.append(float(q_arr[tuple(idx)]))
        
        print(f"Loaded {len(locations)} points from {filepath}")
        print(f"  Grid: {len(lats)} lat x {len(lons)} lon x {len(depths)} depth")
        print(f"  Depth range: {min(depth_list):.1f} to {max(depth_list):.1f} km")
        if vs_list:
            print(f"  Vs range: {min(vs_list):.3f} to {max(vs_list):.3f} km/s")
        if q_list:
            print(f"  Q range: {min(q_list):.1f} to {max(q_list):.1f}")
        
        return SeismicModelData(
            locations=locations,
            names=names,
            z_ranges=z_ranges_list,
            depths=depth_list,
            Vs=np.array(vs_list) if vs_list else None,
            Vs_error=np.full(len(vs_list), default_vs_error) if vs_list else None,
            Q=np.array(q_list) if q_list else None,
            Q_error=np.full(len(q_list), default_q_error) if q_list else None,
        )
    
    else:
        # Flattened 1D data - all arrays should be the same length
        if vs_data is not None:
            vs_data = vs_data.flatten()
        if q_data is not None:
            q_data = q_data.flatten()
        
        # For 1D data, coordinates must match data length
        n_pts = len(ref_data.flatten())
        if len(lats) == 1:
            lats = np.full(n_pts, lats[0])
        if len(lons) == 1:
            lons = np.full(n_pts, lons[0])
        if len(depths) == 1:
            depths = np.full(n_pts, depths[0])
        
        if not (len(lats) == len(lons) == len(depths) == n_pts):
            raise ValueError(
                f"For 1D data, coordinate arrays must match data length ({n_pts}). "
                f"Got lat={len(lats)}, lon={len(lons)}, depth={len(depths)}"
            )
        
        # Extract errors if available
        vs_err = _get_field(model_obj, ['Vs_error', 'Error', 'Vs_err', 'sigma_Vs'])
        q_err = _get_field(model_obj, ['Q_error', 'Q_err', 'sigma_Q'])
        
        if vs_err is not None:
            vs_err = np.atleast_1d(vs_err).flatten()
        if q_err is not None:
            q_err = np.atleast_1d(q_err).flatten()
        
        # Filter by z_range if specified
        if z_range is not None:
            mask = (depths >= z_range[0]) & (depths <= z_range[1])
            lats = lats[mask]
            lons = lons[mask]
            depths = depths[mask]
            if vs_data is not None:
                vs_data = vs_data[mask]
            if q_data is not None:
                q_data = q_data[mask]
            if vs_err is not None:
                vs_err = vs_err[mask]
            if q_err is not None:
                q_err = q_err[mask]
        
        if len(depths) == 0:
            raise ValueError(f"No data in depth range {z_range}")
        
        # Subsample if requested
        if subsample > 1:
            idx = np.arange(0, len(depths), subsample)
            lats = lats[idx]
            lons = lons[idx]
            depths = depths[idx]
            if vs_data is not None:
                vs_data = vs_data[idx]
            if q_data is not None:
                q_data = q_data[idx]
            if vs_err is not None:
                vs_err = vs_err[idx]
            if q_err is not None:
                q_err = q_err[idx]
        
        # Calculate z_ranges based on depth spacing
        depth_to_range = _calculate_depth_ranges(depths)
        
        # Build output
        locations = []
        names = []
        z_ranges = []
        depth_list = []
        
        for i in range(len(depths)):
            lat, lon, depth = lats[i], lons[i], depths[i]
            locations.append((float(lat), float(lon)))
            names.append(str(i + 1))  # Simple index (1-based)
            depth_list.append(float(depth))
            z_range_pt = depth_to_range.get(depth, (depth - 2.5, depth + 2.5))
            z_ranges.append(z_range_pt)
        
        # Set default errors if not provided
        if vs_data is not None and vs_err is None:
            vs_err = np.full(len(vs_data), default_vs_error)
        if q_data is not None and q_err is None:
            q_err = np.full(len(q_data), default_q_error)
        
        print(f"Loaded {len(locations)} points from {filepath}")
        print(f"  Depth range: {min(depth_list):.1f} to {max(depth_list):.1f} km")
        if vs_data is not None:
            print(f"  Vs range: {vs_data.min():.3f} to {vs_data.max():.3f} km/s")
        if q_data is not None:
            print(f"  Q range: {q_data.min():.1f} to {q_data.max():.1f}")
        
        return SeismicModelData(
            locations=locations,
            names=names,
            z_ranges=z_ranges,
            depths=depth_list,
            Vs=vs_data,
            Vs_error=vs_err,
            Q=q_data,
            Q_error=q_err,
        )


def _get_field(obj: Any, field_names: List[str]) -> Optional[np.ndarray]:
    """Try to get a field from an object using multiple possible names."""
    for name in field_names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def load_seismic_model_from_netcdf(
    filepath: str,
    default_vs_error: float = 0.05,
    default_q_error: float = 10.0,
    z_range: Optional[Tuple[float, float]] = None,
    subsample: int = 1,
    vs_var: Optional[str] = None,
    q_var: Optional[str] = None,
) -> SeismicModelData:
    """
    Load seismic model data from a NetCDF file using xarray.
    
    The NetCDF should have dimensions for lat, lon, and depth, with
    data variables for Vs and/or Q.
    
    Parameters
    ----------
    filepath : str
        Path to the NetCDF file
    default_vs_error : float
        Default Vs error if not provided [km/s]
    default_q_error : float
        Default Q error if not provided
    z_range : tuple, optional
        (z_min, z_max) to filter depths. If None, use all depths.
    subsample : int
        Use every Nth point in each dimension (1 = all points)
    vs_var : str, optional
        Name of Vs variable. If None, will try to auto-detect.
    q_var : str, optional
        Name of Q variable. If None, will try to auto-detect.
        
    Returns
    -------
    SeismicModelData
        Container with locations and seismic observations
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray is required for NetCDF support. Install with: pip install xarray netCDF4")
    
    # Open dataset
    ds = xr.open_dataset(filepath)
    
    # Find coordinate dimensions
    lat_dim = _find_dim(ds, ['lat', 'latitude', 'y', 'Latitude', 'Lat'])
    lon_dim = _find_dim(ds, ['lon', 'longitude', 'x', 'Longitude', 'Lon'])
    depth_dim = _find_dim(ds, ['depth', 'z', 'level', 'Depth', 'Z'])
    
    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Could not find lat/lon dimensions. Found: {list(ds.dims)}")
    if depth_dim is None:
        raise ValueError(f"Could not find depth dimension. Found: {list(ds.dims)}")
    
    # Get coordinate values
    lats = ds[lat_dim].values
    lons = ds[lon_dim].values
    depths = ds[depth_dim].values
    
    # Filter by z_range
    if z_range is not None:
        depth_mask = (depths >= z_range[0]) & (depths <= z_range[1])
        depths = depths[depth_mask]
        ds = ds.isel({depth_dim: depth_mask})
    
    if len(depths) == 0:
        raise ValueError(f"No data in depth range {z_range}")
    
    # Subsample
    if subsample > 1:
        lats = lats[::subsample]
        lons = lons[::subsample]
        depths = depths[::subsample]
        ds = ds.isel({
            lat_dim: slice(None, None, subsample),
            lon_dim: slice(None, None, subsample),
            depth_dim: slice(None, None, subsample),
        })
    
    # Find Vs variable
    if vs_var is None:
        vs_var = _find_var(ds, ['vs', 'Vs', 'vsv', 'Vsv', 'shear_velocity', 'velocity'])
    
    # Find Q variable
    if q_var is None:
        q_var = _find_var(ds, ['q', 'Q', 'qmu', 'Qmu', 'quality_factor'])
    
    if vs_var is None and q_var is None:
        raise ValueError(f"Could not find Vs or Q variable. Found: {list(ds.data_vars)}")
    
    # Get data arrays
    vs_data = ds[vs_var].values if vs_var else None
    q_data = ds[q_var].values if q_var else None
    
    # Calculate z_ranges based on depth spacing
    depth_to_range = _calculate_depth_ranges(depths)
    
    # Build flattened output (iterate over all grid points)
    locations = []
    names = []
    z_ranges_list = []
    depth_list = []
    vs_list = []
    q_list = []
    
    # Determine dimension order
    dim_order = list(ds[vs_var or q_var].dims)
    lat_idx = dim_order.index(lat_dim)
    lon_idx = dim_order.index(lon_dim)
    depth_idx = dim_order.index(depth_dim)
    
    for k, depth in enumerate(depths):
        z_range_pt = depth_to_range.get(depth, (depth - 2.5, depth + 2.5))
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                locations.append((float(lat), float(lon)))
                names.append(str(len(locations)))  # Simple index
                depth_list.append(float(depth))
                z_ranges_list.append(z_range_pt)
                
                # Extract value based on dimension order
                idx = [None, None, None]
                idx[lat_idx] = i
                idx[lon_idx] = j
                idx[depth_idx] = k
                
                if vs_data is not None:
                    vs_list.append(float(vs_data[tuple(idx)]))
                if q_data is not None:
                    q_list.append(float(q_data[tuple(idx)]))
    
    ds.close()
    
    print(f"Loaded {len(locations)} points from {filepath}")
    print(f"  Grid: {len(lats)} lat x {len(lons)} lon x {len(depths)} depth")
    print(f"  Depth range: {min(depth_list):.1f} to {max(depth_list):.1f} km")
    if vs_list:
        print(f"  Vs range: {min(vs_list):.3f} to {max(vs_list):.3f} km/s")
    if q_list:
        print(f"  Q range: {min(q_list):.1f} to {max(q_list):.1f}")
    
    return SeismicModelData(
        locations=locations,
        names=names,
        z_ranges=z_ranges_list,
        depths=depth_list,
        Vs=np.array(vs_list) if vs_list else None,
        Vs_error=np.full(len(vs_list), default_vs_error) if vs_list else None,
        Q=np.array(q_list) if q_list else None,
        Q_error=np.full(len(q_list), default_q_error) if q_list else None,
    )


def _find_dim(ds, candidates: List[str]) -> Optional[str]:
    """Find a dimension name from a list of candidates."""
    for name in candidates:
        if name in ds.dims:
            return name
        # Also check coordinates
        if name in ds.coords:
            return name
    return None


def _find_var(ds, candidates: List[str]) -> Optional[str]:
    """Find a variable name from a list of candidates."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None
