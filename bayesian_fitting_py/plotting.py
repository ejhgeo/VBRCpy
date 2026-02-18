"""
Plotting functions for Bayesian inversion results.

Translated from MATLAB Projects/bayesian_fitting/plotting/*.m
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

from .prior import calculate_levels


def plot_seismic_obs(
    obs_value_z: np.ndarray,
    obs_error_z: np.ndarray,
    obs_name: str,
    depth: np.ndarray,
    location: Any,
    obs_value: float,
    obs_error: float,
) -> Figure:
    """
    Plot seismic observation as a function of depth.

    Parameters
    ----------
    obs_value_z : np.ndarray
        Observation values as a function of depth
    obs_error_z : np.ndarray
        Error values as a function of depth
    obs_name : str
        Observation name ('Vs', 'Q', etc.)
    depth : np.ndarray
        Depth values [km]
    location : Location
        Location specification with z_min, z_max
    obs_value : float
        Mean observation in depth range
    obs_error : float
        Mean error in depth range

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    
    ax.fill_betweenx(
        depth,
        obs_value_z - obs_error_z,
        obs_value_z + obs_error_z,
        alpha=0.3,
        color='blue',
        label='±1σ'
    )
    ax.plot(obs_value_z, depth, 'b-', linewidth=2, label=obs_name)
    
    # Mark the averaging depth range
    ax.axhline(location.z_min, color='r', linestyle='--', alpha=0.7)
    ax.axhline(location.z_max, color='r', linestyle='--', alpha=0.7)
    ax.axvline(obs_value, color='g', linestyle='-', alpha=0.7,
               label=f'Mean: {obs_value:.3f}')
    
    ax.set_ylabel('Depth (km)')
    ax.set_xlabel(obs_name)
    ax.invert_yaxis()
    ax.legend()
    ax.set_title(f'{obs_name} Profile at ({location.lat:.1f}°N, {location.lon:.1f}°E)')
    
    plt.tight_layout()
    return fig


def plot_tradeoffs_posterior(
    posterior: np.ndarray,
    sweep: Dict[str, Any],
    obs_name: str,
    anelastic_method: str,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot the posterior probability distribution across parameter space.

    Parameters
    ----------
    posterior : np.ndarray
        3D posterior probability array (T, phi, gs)
    sweep : dict
        Parameter sweep dictionary
    obs_name : str
        Observation description string
    anelastic_method : str
        Anelastic method name
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    Figure
        Matplotlib figure
    """
    anelastic_method_display = anelastic_method.replace('_', ' ')
    
    fig = plt.figure(figsize=(14, 7), facecolor='w')
    
    # Normalize posterior
    posterior = posterior / np.sum(posterior)
    
    # State variable info
    state_names = sweep['state_names']
    fnames = {
        'T': 'Temperature (°C)',
        'phi': 'Melt Fraction, φ',
        'gs': 'Grain Size',
    }
    fnames_short = {'T': 'T', 'phi': 'φ', 'gs': 'd'}
    
    # Handle grain size axis
    gs_values = sweep['gs'].copy()
    gs_params = sweep.get('gs_params', {'type': 'linear'})
    
    if gs_params.get('type') == 'log':
        gs_values = np.log(gs_values / gs_params['gsref'])
        fnames['gs'] = 'ln(Grain Size)'
    else:
        gs_values = gs_values / 1e3  # Convert to mm
        fnames['gs'] = 'Grain Size (mm)'
    
    # Create marginal plots for each pair
    # In MATLAB: plot_box(posterior, sweep, 2, 3, 1) -> phi vs gs, marg over T
    #            plot_box(posterior, sweep, 1, 3, 2) -> T vs gs, marg over phi  
    #            plot_box(posterior, sweep, 1, 2, 3) -> T vs phi, marg over gs
    # i1 is y-axis, i2 is x-axis, i3 is marginalized
    plot_indices = [
        (1, 2, 0),  # phi (y) vs gs (x), marginalize over T
        (0, 2, 1),  # T (y) vs gs (x), marginalize over phi
        (0, 1, 2),  # T (y) vs phi (x), marginalize over gs
    ]
    
    axes_dict = {}
    for panel_idx, (i1, i2, i3) in enumerate(plot_indices):
        ax = fig.add_axes([0.05 + 0.325 * panel_idx, 0.47, 0.225, 0.4])
        
        # Marginal over i3 (sum over the third index)
        p_joint = np.sum(posterior, axis=i3)
        
        # Get axis values: i2 is x-axis, i1 is y-axis
        x_var = state_names[i2]
        y_var = state_names[i1]
        
        x_vals = gs_values if x_var == 'gs' else sweep[x_var]
        y_vals = gs_values if y_var == 'gs' else sweep[y_var]
        
        # p_joint has shape (len(state[i1]), len(state[i2])) after sum over i3
        # For imagesc-like behavior: data[i,j] at position (x[j], y[i])
        im = ax.imshow(
            p_joint,
            aspect='auto',
            origin='lower',
            extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
            cmap='viridis',
        )
        
        ax.set_xlabel(fnames[x_var])
        ax.set_ylabel(fnames[y_var])
        plt.colorbar(im, ax=ax)
        
        axes_dict[f'panel_{panel_idx}'] = ax
        
        # Add marginal plot below
        ax_marg = fig.add_axes([0.05 + 0.325 * panel_idx, 0.3, 0.225, 0.05])
        marg_var = state_names[i3]
        if marg_var == 'gs':
            marg_x = gs_values
        else:
            marg_x = sweep[marg_var]
        # Sum over both other axes to get the marginal for variable i3
        marg_y = np.sum(posterior, axis=(i1, i2))
        ax_marg.plot(marg_x, marg_y)
        ax_marg.set_xlabel(fnames[marg_var])
        ax_marg.set_xlim([marg_x.min(), marg_x.max()])
        
        # Add depth/pressure indicator (like MATLAB version)
        if 'z_inds' in sweep and 'P_GPa' in sweep and 'z' in sweep:
            z_inds = sweep['z_inds']
            p_range = [sweep['P_GPa'][z_inds[0]], sweep['P_GPa'][z_inds[-1]]]
            z_range = [sweep['z'][z_inds[0]] / 1000, sweep['z'][z_inds[-1]] / 1000]  # Convert to km
            
            # Pressure axis on top
            ax_pressure = fig.add_axes([0.05 + 0.325 * panel_idx, 0.12, 0.225, 0.05])
            ax_pressure.axvspan(p_range[0], p_range[1], alpha=0.5, color='blue')
            ax_pressure.set_xlim([sweep['P_GPa'].min(), sweep['P_GPa'].max()])
            ax_pressure.set_yticks([])
            ax_pressure.xaxis.set_ticks_position('top')
            ax_pressure.xaxis.set_label_position('top')
            ax_pressure.set_xlabel('Pressure (GPa)')
            ax_pressure.spines['bottom'].set_visible(False)
            ax_pressure.spines['left'].set_visible(False)
            ax_pressure.spines['right'].set_visible(False)
            
            # Depth axis on bottom
            ax_depth = fig.add_axes([0.05 + 0.325 * panel_idx, 0.12, 0.225, 0.05])
            ax_depth.set_xlim([sweep['z'].min() / 1000, sweep['z'].max() / 1000])
            ax_depth.set_yticks([])
            ax_depth.set_xlabel('Depth (km)')
            ax_depth.patch.set_alpha(0)
            ax_depth.spines['top'].set_visible(False)
            ax_depth.spines['left'].set_visible(False)
            ax_depth.spines['right'].set_visible(False)
    
    # Title
    title_parts = ', '.join([fnames_short[n] for n in state_names])
    fig.suptitle(f'p({title_parts} | {obs_name}), using {anelastic_method_display}',
                 fontsize=14, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_regional_fits(
    regional_fits: Dict[str, Any],
    locs: np.ndarray,
    names: List[str],
    location_colors: List[Tuple[float, ...]],
    fname_prefix: str,
    save_dir: str = 'plots',
) -> Figure:
    """
    Plot regional fits for all anelastic methods and locations.

    Parameters
    ----------
    regional_fits : dict
        Nested dict: regional_fits[anelastic_method][locname] with p_joint, phi_post, T_post
    locs : np.ndarray
        Array of (lat, lon) locations
    names : list
        Location names
    location_colors : list
        Colors for each location
    fname_prefix : str
        Prefix for saved figure files
    save_dir : str
        Directory to save figures

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor='w')
    
    meth_order = ['andrade_psp', 'eburgers_psp', 'xfit_mxw', 'xfit_premelt']
    
    for i_meth, anelastic_method in enumerate(meth_order):
        ax = axes.flat[i_meth]
        ax.set_title(anelastic_method.replace('_', ' '))
        
        if anelastic_method not in regional_fits:
            continue
        
        for il, locname in enumerate(names):
            if locname not in regional_fits[anelastic_method]:
                continue
            
            data = regional_fits[anelastic_method][locname]
            p_joint = data['p_joint']
            post_phi = data['phi_post']
            post_T = data['T_post']
            
            # Calculate contour levels
            targets = [0.7, 0.8, 0.9, 0.95]
            targ_cutoffs, _, _ = calculate_levels(p_joint, targets)
            sizes = [2.5, 2.0, 1.5, 1.0, 0.75]
            
            color = location_colors[il]
            
            for i_cut, cutoff in enumerate(targ_cutoffs):
                if cutoff > 0:
                    ax.contour(
                        post_phi, post_T, p_joint,
                        levels=[cutoff],
                        colors=[color],
                        linewidths=sizes[i_cut],
                    )
        
        ax.set_xlabel('Melt Fraction φ')
        ax.set_ylabel('Temperature (°C)')
    
    plt.tight_layout()
    
    # Save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{save_dir}/{fname_prefix}_regional_fits.png', dpi=150)
    fig.savefig(f'{save_dir}/{fname_prefix}_regional_fits.pdf')
    
    print(f"    saving regional fits to {save_dir}/")
    
    return fig


def plot_ensemble_panel(
    ax: Axes,
    ensemble_pdf: Dict[str, Any],
    locs: np.ndarray,
    names: List[str],
    location_colors: List[Tuple[float, ...]],
    title: str,
    linestyle: str = '-',
) -> Axes:
    """
    Plot ensemble PDF panel.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    ensemble_pdf : dict
        Ensemble PDF dictionary
    locs : np.ndarray
        Location array
    names : list
        Location names
    location_colors : list
        Colors for each location
    title : str
        Panel title
    linestyle : str
        Line style for contours

    Returns
    -------
    Axes
        Updated axes
    """
    ax.set_title(title)
    ax.set_xlabel('Melt Fraction φ')
    ax.set_ylabel('Temperature (°C)')
    
    for il, locname in enumerate(names):
        if locname not in ensemble_pdf:
            continue
        
        pdf = ensemble_pdf[locname]['p_joint']
        T_ax = ensemble_pdf[locname]['post_T']
        phi_ax = ensemble_pdf[locname]['post_phi']
        
        targets = [0.7, 0.8, 0.9, 0.95]
        targ_cutoffs, _, _ = calculate_levels(pdf, targets)
        sizes = [2.5, 2.0, 1.5, 1.0, 0.75]
        
        color = location_colors[il]
        
        for i_cut, cutoff in enumerate(targ_cutoffs):
            if cutoff > 0:
                ax.contour(
                    phi_ax, T_ax, pdf,
                    levels=[cutoff],
                    colors=[color],
                    linewidths=sizes[i_cut],
                    linestyles=linestyle,
                )
    
    return ax


def plot_ensemble_pdfs(
    ensemble_pdf: Dict[str, Any],
    ensemble_pdf_no_mxw: Optional[Dict[str, Any]],
    locs: np.ndarray,
    names: List[str],
    location_colors: List[Tuple[float, ...]],
    fname_prefix: str,
    save_dir: str = 'plots',
) -> Figure:
    """
    Plot ensemble PDFs (with and without xfit_mxw).

    Parameters
    ----------
    ensemble_pdf : dict
        Full ensemble PDF
    ensemble_pdf_no_mxw : dict or None
        Ensemble excluding xfit_mxw
    locs : np.ndarray
        Location array
    names : list
        Location names
    location_colors : list
        Colors for each location
    fname_prefix : str
        Prefix for saved files
    save_dir : str
        Directory to save figures

    Returns
    -------
    Figure
        Matplotlib figure
    """
    n_panels = 2 if ensemble_pdf_no_mxw else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4), facecolor='w')
    
    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    plot_ensemble_panel(
        axes[0], ensemble_pdf, locs, names, location_colors,
        'Full Ensemble'
    )
    
    if ensemble_pdf_no_mxw:
        plot_ensemble_panel(
            axes[1], ensemble_pdf_no_mxw, locs, names, location_colors,
            'Excluding xfit_mxw'
        )
    
    plt.tight_layout()
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{save_dir}/{fname_prefix}_ensemble_fits.png', dpi=150)
    fig.savefig(f'{save_dir}/{fname_prefix}_ensemble_fits.pdf')
    
    print(f"    saving ensemble plots to {save_dir}/")
    
    return fig


def save_figure_for_posterior(
    posterior: Dict[str, Any],
    sweep: Dict[str, Any],
    location_name: str,
    anelastic_method: str,
    save_dir: str,
    obs_type: str = 'VQ',
) -> None:
    """
    Generate and save posterior plots.

    Parameters
    ----------
    posterior : dict
        Posterior dictionary with pS and observation info
    sweep : dict
        Parameter sweep dictionary
    location_name : str
        Name of the location
    anelastic_method : str
        Anelastic method
    save_dir : str
        Directory to save figures
    obs_type : str
        'V', 'Q', or 'VQ' indicating which observations were used
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Build observation string
    obs_parts = []
    if 'obs_Vs' in posterior:
        obs_parts.append(f"Vs = {posterior['obs_Vs']:.3f} ± {posterior['sigma_Vs']:.2f} km/s")
    if 'obs_Q' in posterior:
        obs_parts.append(f"Q = {posterior['obs_Q']:.1f} ± {posterior['sigma_Q']:.1f}")
    obs_str = ', '.join(obs_parts)
    
    fig = plot_tradeoffs_posterior(
        posterior['pS'], sweep, obs_str, anelastic_method,
        save_path=f'{save_dir}/{location_name}_{obs_type}_{anelastic_method}.png'
    )
    plt.close(fig)


def generate_colors(n: int) -> List[Tuple[float, ...]]:
    """
    Generate N distinct colors using a colormap.
    
    Parameters
    ----------
    n : int
        Number of colors to generate
        
    Returns
    -------
    list
        List of (R, G, B) tuples with values in [0, 1]
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis')
    colors = [tuple(cmap(i / max(n-1, 1))[:3]) for i in range(n)]
    return colors
