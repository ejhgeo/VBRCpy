#!/usr/bin/env python3
"""
Visualize VBR lookup tables (LUT).

Creates figures like Fig. 7 from the paper showing Vs and Q as functions
of temperature, grain size, and melt fraction at fixed pressure/depth.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from typing import Optional, Dict, Any


def plot_lut_slices(
    sweep: Dict[str, Any],
    method: str = 'andrade_psp',
    P_GPa: float = 2.0,
    T_fixed: float = 2000,
    phi_fixed: float = 0.0,
    gs_fixed_mm: float = 10.0,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None,
    T_lim: Optional[tuple] = (800, 2500),
    phi_lim: Optional[tuple] = (0, 0.025),
    gs_lim_mm: Optional[tuple] = (1, 30),
):
    """
    Plot lookup table slices showing Vs and Q as functions of T, φ, and d.
    
    Parameters
    ----------
    sweep : dict
        Sweep structure from generate_parameter_sweep or loaded from .mat
    method : str
        Anelastic method to plot (e.g., 'andrade_psp', 'eburgers_psp')
    P_GPa : float
        Pressure in GPa for the slice
    T_fixed : float
        Temperature in °C for the T-fixed slice
    phi_fixed : float  
        Melt fraction for the φ-fixed slice
    gs_fixed_mm : float
        Grain size in mm for the gs-fixed slice
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    T_lim : tuple, optional
        Temperature axis limits (min, max) in °C
    phi_lim : tuple, optional
        Melt fraction axis limits (min, max)
    gs_lim_mm : tuple, optional
        Grain size axis limits (min, max) in mm
    """
    # Extract arrays from sweep
    T_arr = np.atleast_1d(sweep['T'] if isinstance(sweep, dict) else sweep.T)
    phi_arr = np.atleast_1d(sweep['phi'] if isinstance(sweep, dict) else sweep.phi)
    gs_arr = np.atleast_1d(sweep['gs'] if isinstance(sweep, dict) else sweep.gs)
    P_arr = np.atleast_1d(sweep['P_GPa'] if isinstance(sweep, dict) else sweep.P_GPa)
    
    # Find pressure index
    i_P = np.argmin(np.abs(P_arr - P_GPa))
    actual_P = P_arr[i_P]
    
    # Find fixed indices
    i_T_fixed = np.argmin(np.abs(T_arr - T_fixed))
    i_phi_fixed = np.argmin(np.abs(phi_arr - phi_fixed))
    gs_fixed_um = gs_fixed_mm * 1000
    i_gs_fixed = np.argmin(np.abs(gs_arr - gs_fixed_um))
    
    # Extract Vs and Q arrays
    if isinstance(sweep, dict):
        # Python dict structure
        box = sweep['Box']
        Vs_full = box[method]['meanVs']  # Shape: (nT, nphi, ngs, nP)
        Q_full = box[method]['meanQ']
    else:
        # MATLAB structure loaded via scipy
        Box = sweep.Box
        Vs_list = []
        Q_list = []
        for i_T in range(len(T_arr)):
            vs_phi = []
            q_phi = []
            for i_phi in range(len(phi_arr)):
                vs_gs = []
                q_gs = []
                for i_gs in range(len(gs_arr)):
                    box_elem = Box[i_T, i_phi, i_gs]
                    method_data = getattr(box_elem, method)
                    vs_gs.append(np.atleast_1d(method_data.meanVs))
                    q_gs.append(np.atleast_1d(method_data.meanQ))
                vs_phi.append(vs_gs)
                q_phi.append(q_gs)
            Vs_list.append(vs_phi)
            Q_list.append(q_phi)
        Vs_full = np.array(Vs_list)
        Q_full = np.array(Q_list)
    
    # Extract slices at fixed P
    Vs = Vs_full[:, :, :, i_P]  # Shape: (nT, nphi, ngs)
    Q = Q_full[:, :, :, i_P]
    
    # Create figure with GridSpec for better colorbar positioning
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3, hspace=0.25)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)])
    cax_vs = fig.add_subplot(gs[0, 3])
    cax_q = fig.add_subplot(gs[1, 3])
    
    # Get period range for title
    per_min = sweep['per_bw_min'] if isinstance(sweep, dict) else sweep.per_bw_min
    per_max = sweep['per_bw_max'] if isinstance(sweep, dict) else sweep.per_bw_max
    
    # Grain size in mm for plotting
    gs_mm = gs_arr / 1000
    
    # Fixed color ranges
    Vs_vmin, Vs_vmax = 4.1, 4.6  # km/s
    Q_vmin, Q_vmax = 10, 150
    
    # Create level arrays for consistent contours
    Vs_levels = np.linspace(Vs_vmin, Vs_vmax, 21)
    Q_levels = np.linspace(Q_vmin, Q_vmax, 21)
    
    # Reversed colormap (RdBu instead of RdBu_r)
    cmap = 'RdBu'
    
    # Store image mappables for shared colorbars
    Vs_ims = []
    Q_ims = []
    
    # Slice 1: T fixed - plot φ vs gs
    # Vs[i_T_fixed, :, :] has shape (nphi, ngs)
    ax = axes[0, 0]
    im = ax.contourf(gs_mm, phi_arr, Vs[i_T_fixed, :, :], levels=Vs_levels, cmap=cmap, 
                     vmin=Vs_vmin, vmax=Vs_vmax, extend='both')
    ax.contour(gs_mm, phi_arr, Vs[i_T_fixed, :, :], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_ylabel('Melt Fraction, φ')
    ax.set_title(f'Temperature fixed at {T_arr[i_T_fixed]:.0f}°C')
    Vs_ims.append(im)
    
    ax = axes[1, 0]
    im = ax.contourf(gs_mm, phi_arr, Q[i_T_fixed, :, :], levels=Q_levels, cmap=cmap,
                     vmin=Q_vmin, vmax=Q_vmax, extend='both')
    ax.contour(gs_mm, phi_arr, Q[i_T_fixed, :, :], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_ylabel('Melt Fraction, φ')
    Q_ims.append(im)
    
    # Slice 2: φ fixed - plot T vs gs
    # Vs[:, i_phi_fixed, :] has shape (nT, ngs)
    ax = axes[0, 1]
    im = ax.contourf(gs_mm, T_arr, Vs[:, i_phi_fixed, :], levels=Vs_levels, cmap=cmap,
                     vmin=Vs_vmin, vmax=Vs_vmax, extend='both')
    ax.contour(gs_mm, T_arr, Vs[:, i_phi_fixed, :], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'Melt Fraction, φ fixed at {phi_arr[i_phi_fixed]:.4f}')
    Vs_ims.append(im)
    
    ax = axes[1, 1]
    im = ax.contourf(gs_mm, T_arr, Q[:, i_phi_fixed, :], levels=Q_levels, cmap=cmap,
                     vmin=Q_vmin, vmax=Q_vmax, extend='both')
    ax.contour(gs_mm, T_arr, Q[:, i_phi_fixed, :], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_ylabel('Temperature (°C)')
    Q_ims.append(im)
    
    # Slice 3: gs fixed - plot T vs φ
    # Vs[:, :, i_gs_fixed] has shape (nT, nphi)
    ax = axes[0, 2]
    im = ax.contourf(phi_arr, T_arr, Vs[:, :, i_gs_fixed], levels=Vs_levels, cmap=cmap,
                     vmin=Vs_vmin, vmax=Vs_vmax, extend='both')
    ax.contour(phi_arr, T_arr, Vs[:, :, i_gs_fixed], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_xlabel('Melt Fraction, φ')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title(f'Grain Size fixed at {gs_arr[i_gs_fixed]/1000:.0f} mm')
    Vs_ims.append(im)
    
    ax = axes[1, 2]
    im = ax.contourf(phi_arr, T_arr, Q[:, :, i_gs_fixed], levels=Q_levels, cmap=cmap,
                     vmin=Q_vmin, vmax=Q_vmax, extend='both')
    ax.contour(phi_arr, T_arr, Q[:, :, i_gs_fixed], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_xlabel('Melt Fraction, φ')
    ax.set_ylabel('Temperature (°C)')
    Q_ims.append(im)
    
    # Add shared colorbars in dedicated axes
    cbar_vs = fig.colorbar(Vs_ims[0], cax=cax_vs)
    cbar_vs.set_label('Vs (km/s)')
    cbar_vs.set_ticks(np.arange(Vs_vmin, Vs_vmax + 0.1, 0.1))
    
    cbar_q = fig.colorbar(Q_ims[0], cax=cax_q)
    cbar_q.set_label('Q')
    cbar_q.set_ticks(np.arange(Q_vmin, Q_vmax + 20, 20))
    
    # Apply axis limits if specified
    for ax in axes.flatten():
        # Determine which axes this panel uses based on labels
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        
        if 'Grain Size' in xlabel and gs_lim_mm is not None:
            ax.set_xlim(gs_lim_mm)
        if 'Melt Fraction' in xlabel and phi_lim is not None:
            ax.set_xlim(phi_lim)
        if 'Melt Fraction' in ylabel and phi_lim is not None:
            ax.set_ylim(phi_lim)
        if 'Temperature' in ylabel and T_lim is not None:
            ax.set_ylim(T_lim)
    
    # Main title
    fig.suptitle(f'Vs, Q as a function of T, φ, d  ({actual_P:.1f} GPa, {per_min:.0f} - {per_max:.0f} s)\n'
                 f'using {method}', fontsize=14)
    
    if save_path:
        import os
        # If save_path is a directory, create a filename
        if os.path.isdir(save_path) or save_path.endswith('/'):
            os.makedirs(save_path, exist_ok=True)
            filename = f'lut_slices_{method}_P{actual_P:.1f}GPa.png'
            save_path = os.path.join(save_path, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved figure to {save_path}')
    else:
        plt.show()

    return fig


def compare_lut_slices(
    sweep_ref: Any,
    sweep_new: Any,
    method: str = 'andrade_psp',
    P_GPa: float = 2.0,
    figsize: tuple = (24, 10),
    save_path: Optional[str] = None,
    T_lim: Optional[tuple] = (1200, 1700),
    gs_lim_mm: Optional[tuple] = (1, 30),
):
    """
    Compare reference and new LUT slices side by side.
    
    Both sweeps can be either MATLAB structures or Python dicts.
    
    Parameters
    ----------
    sweep_ref : Any
        Reference sweep (MATLAB struct or dict)
    sweep_new : Any
        New sweep to compare (MATLAB struct or dict)
    method : str
        Anelastic method to plot
    P_GPa : float
        Pressure in GPa for the slice
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    T_lim : tuple, optional
        Temperature axis limits (min, max) in °C
    gs_lim_mm : tuple, optional
        Grain size axis limits (min, max) in mm
    """
    
    def _extract_arrays(sweep):
        """Extract T, phi, gs, P arrays from either format."""
        if isinstance(sweep, dict):
            return (np.atleast_1d(sweep['T']),
                    np.atleast_1d(sweep['phi']),
                    np.atleast_1d(sweep['gs']),
                    np.atleast_1d(sweep['P_GPa']))
        else:
            return (np.atleast_1d(sweep.T),
                    np.atleast_1d(sweep.phi),
                    np.atleast_1d(sweep.gs),
                    np.atleast_1d(sweep.P_GPa))
    
    def _extract_VsQ(sweep, method, i_P, T_arr, phi_arr, gs_arr):
        """Extract Vs and Q arrays from either format."""
        if isinstance(sweep, dict):
            Vs = sweep['Box'][method]['meanVs'][:, :, :, i_P]
            Q = sweep['Box'][method]['meanQ'][:, :, :, i_P]
        else:
            # MATLAB structure - need to iterate
            Vs = np.zeros((len(T_arr), len(phi_arr), len(gs_arr)))
            Q = np.zeros_like(Vs)
            for i_T in range(len(T_arr)):
                for i_phi in range(len(phi_arr)):
                    for i_gs in range(len(gs_arr)):
                        box = sweep.Box[i_T, i_phi, i_gs]
                        Vs[i_T, i_phi, i_gs] = np.atleast_1d(getattr(box, method).meanVs)[i_P]
                        Q[i_T, i_phi, i_gs] = np.atleast_1d(getattr(box, method).meanQ)[i_P]
        return Vs, Q
    
    # Extract arrays from reference
    T_arr, phi_arr, gs_arr, P_arr = _extract_arrays(sweep_ref)
    
    # Find pressure index
    i_P = np.argmin(np.abs(P_arr - P_GPa))
    
    # Fixed indices for comparison
    i_T_fixed = len(T_arr) // 3  # ~1300°C
    i_phi_fixed = 0  # phi=0
    i_gs_fixed = len(gs_arr) // 2  # middle grain size
    
    # Extract Vs and Q from both sweeps
    Vs_ref, Q_ref = _extract_VsQ(sweep_ref, method, i_P, T_arr, phi_arr, gs_arr)
    Vs_new, Q_new = _extract_VsQ(sweep_new, method, i_P, T_arr, phi_arr, gs_arr)
    
    gs_mm = gs_arr / 1000
    
    # Fixed color ranges (same as plot_lut_slices)
    Vs_vmin, Vs_vmax = 4.1, 4.6  # km/s
    Q_vmin, Q_vmax = 10, 150
    Vs_levels = np.linspace(Vs_vmin, Vs_vmax, 21)
    Q_levels = np.linspace(Q_vmin, Q_vmax, 21)
    cmap = 'RdBu'  # Reversed colormap
    
    # Create figure with GridSpec - each difference plot gets its own colorbar
    # Layout: [ref, new, cbar, gap, diff, cbar, gap, %diff, cbar]
    fig = plt.figure(figsize=figsize)
    gs_fig = fig.add_gridspec(2, 10, 
                              width_ratios=[1, 1, 0.08, 0.15, 1, 0.08, 0.15, 1, 0.08, 0.05], 
                              wspace=0.1, hspace=0.25)
    
    # Create axes for each panel and colorbar (skip gap columns 3, 6)
    ax_vs_ref = fig.add_subplot(gs_fig[0, 0])
    ax_vs_new = fig.add_subplot(gs_fig[0, 1])
    cax_vs = fig.add_subplot(gs_fig[0, 2])
    ax_vs_diff = fig.add_subplot(gs_fig[0, 4])
    cax_vs_diff = fig.add_subplot(gs_fig[0, 5])
    ax_vs_pct = fig.add_subplot(gs_fig[0, 7])
    cax_vs_pct = fig.add_subplot(gs_fig[0, 8])
    
    ax_q_ref = fig.add_subplot(gs_fig[1, 0])
    ax_q_new = fig.add_subplot(gs_fig[1, 1])
    cax_q = fig.add_subplot(gs_fig[1, 2])
    ax_q_diff = fig.add_subplot(gs_fig[1, 4])
    cax_q_diff = fig.add_subplot(gs_fig[1, 5])
    ax_q_pct = fig.add_subplot(gs_fig[1, 7])
    cax_q_pct = fig.add_subplot(gs_fig[1, 8])
    
    # Row 1: Vs
    # Col 1: Reference φ=0
    ax = ax_vs_ref
    im_vs = ax.contourf(gs_mm, T_arr, Vs_ref[:, i_phi_fixed, :], levels=Vs_levels, cmap=cmap,
                        vmin=Vs_vmin, vmax=Vs_vmax, extend='both')
    ax.contour(gs_mm, T_arr, Vs_ref[:, i_phi_fixed, :], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_title('Reference Vs (φ=0)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xticklabels([])  # Share x-axis with Q row below
    
    # Col 2: New φ=0
    ax = ax_vs_new
    ax.contourf(gs_mm, T_arr, Vs_new[:, i_phi_fixed, :], levels=Vs_levels, cmap=cmap,
                vmin=Vs_vmin, vmax=Vs_vmax, extend='both')
    ax.contour(gs_mm, T_arr, Vs_new[:, i_phi_fixed, :], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_title('New Vs (φ=0)')
    ax.set_yticklabels([])  # Share y-axis with ref
    ax.set_xticklabels([])
    
    # Shared Vs colorbar for ref/new
    cbar_vs = fig.colorbar(im_vs, cax=cax_vs)
    cbar_vs.set_label('Vs (km/s)')
    cbar_vs.set_ticks(np.arange(Vs_vmin, Vs_vmax + 0.1, 0.1))
    
    # Col 3: Difference
    ax = ax_vs_diff
    diff_vs = Vs_new[:, i_phi_fixed, :] - Vs_ref[:, i_phi_fixed, :]
    vmax_diff_vs = max(abs(diff_vs.min()), abs(diff_vs.max()))
    im_vs_diff = ax.contourf(gs_mm, T_arr, diff_vs, levels=20, cmap='RdBu_r', vmin=-vmax_diff_vs, vmax=vmax_diff_vs)
    ax.set_title('Vs Difference')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    # Vs diff colorbar
    cbar_vs_diff = fig.colorbar(im_vs_diff, cax=cax_vs_diff)
    cbar_vs_diff.set_label('Δ km/s')
    
    # Col 4: Percent difference
    ax = ax_vs_pct
    pct_diff_vs = 100 * diff_vs / Vs_ref[:, i_phi_fixed, :]
    vmax_pct_vs = max(abs(pct_diff_vs.min()), abs(pct_diff_vs.max()))
    im_vs_pct = ax.contourf(gs_mm, T_arr, pct_diff_vs, levels=20, cmap='RdBu_r', vmin=-vmax_pct_vs, vmax=vmax_pct_vs)
    ax.set_title('Vs % Difference')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    # Vs % diff colorbar
    cbar_vs_pct = fig.colorbar(im_vs_pct, cax=cax_vs_pct)
    cbar_vs_pct.set_label('%')
    
    # Row 2: Q
    ax = ax_q_ref
    im_q = ax.contourf(gs_mm, T_arr, Q_ref[:, i_phi_fixed, :], levels=Q_levels, cmap=cmap,
                       vmin=Q_vmin, vmax=Q_vmax, extend='both')
    ax.contour(gs_mm, T_arr, Q_ref[:, i_phi_fixed, :], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_title('Reference Q (φ=0)')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_ylabel('Temperature (°C)')
    
    ax = ax_q_new
    ax.contourf(gs_mm, T_arr, Q_new[:, i_phi_fixed, :], levels=Q_levels, cmap=cmap,
                vmin=Q_vmin, vmax=Q_vmax, extend='both')
    ax.contour(gs_mm, T_arr, Q_new[:, i_phi_fixed, :], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_title('New Q (φ=0)')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_yticklabels([])  # Share y-axis with ref
    
    # Shared Q colorbar for ref/new
    cbar_q = fig.colorbar(im_q, cax=cax_q)
    cbar_q.set_label('Q')
    cbar_q.set_ticks(np.arange(Q_vmin, Q_vmax + 20, 20))
    
    ax = ax_q_diff
    diff_q = Q_new[:, i_phi_fixed, :] - Q_ref[:, i_phi_fixed, :]
    vmax_diff_q = max(abs(diff_q.min()), abs(diff_q.max()))
    im_q_diff = ax.contourf(gs_mm, T_arr, diff_q, levels=20, cmap='RdBu_r', vmin=-vmax_diff_q, vmax=vmax_diff_q)
    ax.set_title('Q Difference')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_yticklabels([])
    
    # Q diff colorbar
    cbar_q_diff = fig.colorbar(im_q_diff, cax=cax_q_diff)
    cbar_q_diff.set_label('ΔQ')
    
    ax = ax_q_pct
    pct_diff_q = 100 * diff_q / Q_ref[:, i_phi_fixed, :]
    vmax_pct_q = min(100, max(abs(pct_diff_q.min()), abs(pct_diff_q.max())))
    im_q_pct = ax.contourf(gs_mm, T_arr, pct_diff_q, levels=20, cmap='RdBu_r', vmin=-vmax_pct_q, vmax=vmax_pct_q)
    ax.set_title('Q % Difference')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_yticklabels([])
    
    # Q % diff colorbar
    cbar_q_pct = fig.colorbar(im_q_pct, cax=cax_q_pct)
    cbar_q_pct.set_label('%')
    
    # Apply axis limits only to Vs/Q ref and new panels (not difference panels)
    value_axes = [ax_vs_ref, ax_vs_new, ax_q_ref, ax_q_new]
    for ax in value_axes:
        if gs_lim_mm is not None:
            ax.set_xlim(gs_lim_mm)
        if T_lim is not None:
            ax.set_ylim(T_lim)
    
    fig.suptitle(f'Comparison (T vs gs, φ=0): {method} at P={P_arr[i_P]:.2f} GPa', fontsize=14)
    
    if save_path:
        import os
        # If save_path is a directory, create a filename
        if os.path.isdir(save_path) or save_path.endswith('/'):
            os.makedirs(save_path, exist_ok=True)
            filename = f'lut_comparison_T_vs_gs_{method}_P{P_arr[i_P]:.1f}GPa.png'
            save_path = os.path.join(save_path, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved figure to {save_path}')
    else: 
        plt.show()
    
    return fig


def compare_lut_slices_gs_phi(
    sweep_ref: Any,
    sweep_new: Any,
    method: str = 'andrade_psp',
    P_GPa: float = 2.0,
    T_fixed: float = 1300,
    figsize: tuple = (24, 10),
    save_path: Optional[str] = None,
    phi_lim: Optional[tuple] = None,
    gs_lim_mm: Optional[tuple] = None,
):
    """
    Compare reference and new LUT slices: grain size vs melt fraction at fixed T.
    
    Both sweeps can be either MATLAB structures or Python dicts.
    """
    
    def _extract_arrays(sweep):
        if isinstance(sweep, dict):
            return (np.atleast_1d(sweep['T']),
                    np.atleast_1d(sweep['phi']),
                    np.atleast_1d(sweep['gs']),
                    np.atleast_1d(sweep['P_GPa']))
        else:
            return (np.atleast_1d(sweep.T),
                    np.atleast_1d(sweep.phi),
                    np.atleast_1d(sweep.gs),
                    np.atleast_1d(sweep.P_GPa))
    
    def _extract_VsQ(sweep, method, i_P, T_arr, phi_arr, gs_arr):
        if isinstance(sweep, dict):
            Vs = sweep['Box'][method]['meanVs'][:, :, :, i_P]
            Q = sweep['Box'][method]['meanQ'][:, :, :, i_P]
        else:
            Vs = np.zeros((len(T_arr), len(phi_arr), len(gs_arr)))
            Q = np.zeros_like(Vs)
            for i_T in range(len(T_arr)):
                for i_phi in range(len(phi_arr)):
                    for i_gs in range(len(gs_arr)):
                        box = sweep.Box[i_T, i_phi, i_gs]
                        Vs[i_T, i_phi, i_gs] = np.atleast_1d(getattr(box, method).meanVs)[i_P]
                        Q[i_T, i_phi, i_gs] = np.atleast_1d(getattr(box, method).meanQ)[i_P]
        return Vs, Q
    
    T_arr, phi_arr, gs_arr, P_arr = _extract_arrays(sweep_ref)
    i_P = np.argmin(np.abs(P_arr - P_GPa))
    i_T_fixed = np.argmin(np.abs(T_arr - T_fixed))
    
    Vs_ref, Q_ref = _extract_VsQ(sweep_ref, method, i_P, T_arr, phi_arr, gs_arr)
    Vs_new, Q_new = _extract_VsQ(sweep_new, method, i_P, T_arr, phi_arr, gs_arr)
    
    gs_mm = gs_arr / 1000
    
    # Fixed color ranges
    Vs_vmin, Vs_vmax = 4.1, 4.6
    Q_vmin, Q_vmax = 10, 150
    Vs_levels = np.linspace(Vs_vmin, Vs_vmax, 21)
    Q_levels = np.linspace(Q_vmin, Q_vmax, 21)
    cmap = 'RdBu'
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs_fig = fig.add_gridspec(2, 10, 
                              width_ratios=[1, 1, 0.08, 0.15, 1, 0.08, 0.15, 1, 0.08, 0.05], 
                              wspace=0.1, hspace=0.25)
    
    ax_vs_ref = fig.add_subplot(gs_fig[0, 0])
    ax_vs_new = fig.add_subplot(gs_fig[0, 1])
    cax_vs = fig.add_subplot(gs_fig[0, 2])
    ax_vs_diff = fig.add_subplot(gs_fig[0, 4])
    cax_vs_diff = fig.add_subplot(gs_fig[0, 5])
    ax_vs_pct = fig.add_subplot(gs_fig[0, 7])
    cax_vs_pct = fig.add_subplot(gs_fig[0, 8])
    
    ax_q_ref = fig.add_subplot(gs_fig[1, 0])
    ax_q_new = fig.add_subplot(gs_fig[1, 1])
    cax_q = fig.add_subplot(gs_fig[1, 2])
    ax_q_diff = fig.add_subplot(gs_fig[1, 4])
    cax_q_diff = fig.add_subplot(gs_fig[1, 5])
    ax_q_pct = fig.add_subplot(gs_fig[1, 7])
    cax_q_pct = fig.add_subplot(gs_fig[1, 8])
    
    # Slice: Vs[i_T_fixed, :, :] has shape (nphi, ngs) - plot gs vs phi
    # Row 1: Vs
    ax = ax_vs_ref
    im_vs = ax.contourf(gs_mm, phi_arr, Vs_ref[i_T_fixed, :, :], levels=Vs_levels, cmap=cmap,
                        vmin=Vs_vmin, vmax=Vs_vmax, extend='both')
    ax.contour(gs_mm, phi_arr, Vs_ref[i_T_fixed, :, :], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_title(f'Reference Vs (T={T_arr[i_T_fixed]:.0f}°C)')
    ax.set_ylabel('Melt Fraction, φ')
    ax.set_xticklabels([])
    
    ax = ax_vs_new
    ax.contourf(gs_mm, phi_arr, Vs_new[i_T_fixed, :, :], levels=Vs_levels, cmap=cmap,
                vmin=Vs_vmin, vmax=Vs_vmax, extend='both')
    ax.contour(gs_mm, phi_arr, Vs_new[i_T_fixed, :, :], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_title(f'New Vs (T={T_arr[i_T_fixed]:.0f}°C)')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    cbar_vs = fig.colorbar(im_vs, cax=cax_vs)
    cbar_vs.set_label('Vs (km/s)')
    cbar_vs.set_ticks(np.arange(Vs_vmin, Vs_vmax + 0.1, 0.1))
    
    ax = ax_vs_diff
    diff_vs = Vs_new[i_T_fixed, :, :] - Vs_ref[i_T_fixed, :, :]
    vmax_diff_vs = max(abs(diff_vs.min()), abs(diff_vs.max()))
    im_vs_diff = ax.contourf(gs_mm, phi_arr, diff_vs, levels=20, cmap='RdBu_r', vmin=-vmax_diff_vs, vmax=vmax_diff_vs)
    ax.set_title('Vs Difference')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    cbar_vs_diff = fig.colorbar(im_vs_diff, cax=cax_vs_diff)
    cbar_vs_diff.set_label('Δ km/s')
    
    ax = ax_vs_pct
    pct_diff_vs = 100 * diff_vs / Vs_ref[i_T_fixed, :, :]
    vmax_pct_vs = max(abs(pct_diff_vs.min()), abs(pct_diff_vs.max()))
    im_vs_pct = ax.contourf(gs_mm, phi_arr, pct_diff_vs, levels=20, cmap='RdBu_r', vmin=-vmax_pct_vs, vmax=vmax_pct_vs)
    ax.set_title('Vs % Difference')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    cbar_vs_pct = fig.colorbar(im_vs_pct, cax=cax_vs_pct)
    cbar_vs_pct.set_label('%')
    
    # Row 2: Q
    ax = ax_q_ref
    im_q = ax.contourf(gs_mm, phi_arr, Q_ref[i_T_fixed, :, :], levels=Q_levels, cmap=cmap,
                       vmin=Q_vmin, vmax=Q_vmax, extend='both')
    ax.contour(gs_mm, phi_arr, Q_ref[i_T_fixed, :, :], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_title(f'Reference Q (T={T_arr[i_T_fixed]:.0f}°C)')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_ylabel('Melt Fraction, φ')
    
    ax = ax_q_new
    ax.contourf(gs_mm, phi_arr, Q_new[i_T_fixed, :, :], levels=Q_levels, cmap=cmap,
                vmin=Q_vmin, vmax=Q_vmax, extend='both')
    ax.contour(gs_mm, phi_arr, Q_new[i_T_fixed, :, :], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_title(f'New Q (T={T_arr[i_T_fixed]:.0f}°C)')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_yticklabels([])
    
    cbar_q = fig.colorbar(im_q, cax=cax_q)
    cbar_q.set_label('Q')
    cbar_q.set_ticks(np.arange(Q_vmin, Q_vmax + 20, 20))
    
    ax = ax_q_diff
    diff_q = Q_new[i_T_fixed, :, :] - Q_ref[i_T_fixed, :, :]
    vmax_diff_q = max(abs(diff_q.min()), abs(diff_q.max()))
    im_q_diff = ax.contourf(gs_mm, phi_arr, diff_q, levels=20, cmap='RdBu_r', vmin=-vmax_diff_q, vmax=vmax_diff_q)
    ax.set_title('Q Difference')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_yticklabels([])
    cbar_q_diff = fig.colorbar(im_q_diff, cax=cax_q_diff)
    cbar_q_diff.set_label('ΔQ')
    
    ax = ax_q_pct
    pct_diff_q = 100 * diff_q / Q_ref[i_T_fixed, :, :]
    vmax_pct_q = min(100, max(abs(pct_diff_q.min()), abs(pct_diff_q.max())))
    im_q_pct = ax.contourf(gs_mm, phi_arr, pct_diff_q, levels=20, cmap='RdBu_r', vmin=-vmax_pct_q, vmax=vmax_pct_q)
    ax.set_title('Q % Difference')
    ax.set_xlabel('Grain Size (mm)')
    ax.set_yticklabels([])
    cbar_q_pct = fig.colorbar(im_q_pct, cax=cax_q_pct)
    cbar_q_pct.set_label('%')
    
    # Apply axis limits only to ref/new panels
    value_axes = [ax_vs_ref, ax_vs_new, ax_q_ref, ax_q_new]
    for ax in value_axes:
        if gs_lim_mm is not None:
            ax.set_xlim(gs_lim_mm)
        if phi_lim is not None:
            ax.set_ylim(phi_lim)
    
    fig.suptitle(f'Comparison (φ vs gs, T={T_arr[i_T_fixed]:.0f}°C): {method} at P={P_arr[i_P]:.2f} GPa', fontsize=14)
    
    if save_path:
        import os
        if os.path.isdir(save_path) or save_path.endswith('/'):
            os.makedirs(save_path, exist_ok=True)
            filename = f'lut_comparison_phi_vs_gs_{method}_P{P_arr[i_P]:.1f}GPa.png'
            save_path = os.path.join(save_path, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved figure to {save_path}')
    else:
        plt.show()
    
    return fig


def compare_lut_slices_T_phi(
    sweep_ref: Any,
    sweep_new: Any,
    method: str = 'andrade_psp',
    P_GPa: float = 2.0,
    gs_fixed_mm: float = 10.0,
    figsize: tuple = (24, 10),
    save_path: Optional[str] = None,
    T_lim: Optional[tuple] = None,
    phi_lim: Optional[tuple] = None,
):
    """
    Compare reference and new LUT slices: temperature vs melt fraction at fixed grain size.
    
    Both sweeps can be either MATLAB structures or Python dicts.
    """
    
    def _extract_arrays(sweep):
        if isinstance(sweep, dict):
            return (np.atleast_1d(sweep['T']),
                    np.atleast_1d(sweep['phi']),
                    np.atleast_1d(sweep['gs']),
                    np.atleast_1d(sweep['P_GPa']))
        else:
            return (np.atleast_1d(sweep.T),
                    np.atleast_1d(sweep.phi),
                    np.atleast_1d(sweep.gs),
                    np.atleast_1d(sweep.P_GPa))
    
    def _extract_VsQ(sweep, method, i_P, T_arr, phi_arr, gs_arr):
        if isinstance(sweep, dict):
            Vs = sweep['Box'][method]['meanVs'][:, :, :, i_P]
            Q = sweep['Box'][method]['meanQ'][:, :, :, i_P]
        else:
            Vs = np.zeros((len(T_arr), len(phi_arr), len(gs_arr)))
            Q = np.zeros_like(Vs)
            for i_T in range(len(T_arr)):
                for i_phi in range(len(phi_arr)):
                    for i_gs in range(len(gs_arr)):
                        box = sweep.Box[i_T, i_phi, i_gs]
                        Vs[i_T, i_phi, i_gs] = np.atleast_1d(getattr(box, method).meanVs)[i_P]
                        Q[i_T, i_phi, i_gs] = np.atleast_1d(getattr(box, method).meanQ)[i_P]
        return Vs, Q
    
    T_arr, phi_arr, gs_arr, P_arr = _extract_arrays(sweep_ref)
    i_P = np.argmin(np.abs(P_arr - P_GPa))
    gs_fixed_um = gs_fixed_mm * 1000
    i_gs_fixed = np.argmin(np.abs(gs_arr - gs_fixed_um))
    
    Vs_ref, Q_ref = _extract_VsQ(sweep_ref, method, i_P, T_arr, phi_arr, gs_arr)
    Vs_new, Q_new = _extract_VsQ(sweep_new, method, i_P, T_arr, phi_arr, gs_arr)
    
    # Fixed color ranges
    Vs_vmin, Vs_vmax = 4.1, 4.6
    Q_vmin, Q_vmax = 10, 150
    Vs_levels = np.linspace(Vs_vmin, Vs_vmax, 21)
    Q_levels = np.linspace(Q_vmin, Q_vmax, 21)
    cmap = 'RdBu'
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs_fig = fig.add_gridspec(2, 10, 
                              width_ratios=[1, 1, 0.08, 0.15, 1, 0.08, 0.15, 1, 0.08, 0.05], 
                              wspace=0.1, hspace=0.25)
    
    ax_vs_ref = fig.add_subplot(gs_fig[0, 0])
    ax_vs_new = fig.add_subplot(gs_fig[0, 1])
    cax_vs = fig.add_subplot(gs_fig[0, 2])
    ax_vs_diff = fig.add_subplot(gs_fig[0, 4])
    cax_vs_diff = fig.add_subplot(gs_fig[0, 5])
    ax_vs_pct = fig.add_subplot(gs_fig[0, 7])
    cax_vs_pct = fig.add_subplot(gs_fig[0, 8])
    
    ax_q_ref = fig.add_subplot(gs_fig[1, 0])
    ax_q_new = fig.add_subplot(gs_fig[1, 1])
    cax_q = fig.add_subplot(gs_fig[1, 2])
    ax_q_diff = fig.add_subplot(gs_fig[1, 4])
    cax_q_diff = fig.add_subplot(gs_fig[1, 5])
    ax_q_pct = fig.add_subplot(gs_fig[1, 7])
    cax_q_pct = fig.add_subplot(gs_fig[1, 8])
    
    # Slice: Vs[:, :, i_gs_fixed] has shape (nT, nphi) - plot phi vs T
    # Row 1: Vs
    ax = ax_vs_ref
    im_vs = ax.contourf(phi_arr, T_arr, Vs_ref[:, :, i_gs_fixed], levels=Vs_levels, cmap=cmap,
                        vmin=Vs_vmin, vmax=Vs_vmax, extend='both')
    ax.contour(phi_arr, T_arr, Vs_ref[:, :, i_gs_fixed], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_title(f'Reference Vs (gs={gs_arr[i_gs_fixed]/1000:.1f} mm)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xticklabels([])
    
    ax = ax_vs_new
    ax.contourf(phi_arr, T_arr, Vs_new[:, :, i_gs_fixed], levels=Vs_levels, cmap=cmap,
                vmin=Vs_vmin, vmax=Vs_vmax, extend='both')
    ax.contour(phi_arr, T_arr, Vs_new[:, :, i_gs_fixed], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_title(f'New Vs (gs={gs_arr[i_gs_fixed]/1000:.1f} mm)')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    cbar_vs = fig.colorbar(im_vs, cax=cax_vs)
    cbar_vs.set_label('Vs (km/s)')
    cbar_vs.set_ticks(np.arange(Vs_vmin, Vs_vmax + 0.1, 0.1))
    
    ax = ax_vs_diff
    diff_vs = Vs_new[:, :, i_gs_fixed] - Vs_ref[:, :, i_gs_fixed]
    vmax_diff_vs = max(abs(diff_vs.min()), abs(diff_vs.max()))
    im_vs_diff = ax.contourf(phi_arr, T_arr, diff_vs, levels=20, cmap='RdBu_r', vmin=-vmax_diff_vs, vmax=vmax_diff_vs)
    ax.set_title('Vs Difference')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    cbar_vs_diff = fig.colorbar(im_vs_diff, cax=cax_vs_diff)
    cbar_vs_diff.set_label('Δ km/s')
    
    ax = ax_vs_pct
    pct_diff_vs = 100 * diff_vs / Vs_ref[:, :, i_gs_fixed]
    vmax_pct_vs = max(abs(pct_diff_vs.min()), abs(pct_diff_vs.max()))
    im_vs_pct = ax.contourf(phi_arr, T_arr, pct_diff_vs, levels=20, cmap='RdBu_r', vmin=-vmax_pct_vs, vmax=vmax_pct_vs)
    ax.set_title('Vs % Difference')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    cbar_vs_pct = fig.colorbar(im_vs_pct, cax=cax_vs_pct)
    cbar_vs_pct.set_label('%')
    
    # Row 2: Q
    ax = ax_q_ref
    im_q = ax.contourf(phi_arr, T_arr, Q_ref[:, :, i_gs_fixed], levels=Q_levels, cmap=cmap,
                       vmin=Q_vmin, vmax=Q_vmax, extend='both')
    ax.contour(phi_arr, T_arr, Q_ref[:, :, i_gs_fixed], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_title(f'Reference Q (gs={gs_arr[i_gs_fixed]/1000:.1f} mm)')
    ax.set_xlabel('Melt Fraction, φ')
    ax.set_ylabel('Temperature (°C)')
    
    ax = ax_q_new
    ax.contourf(phi_arr, T_arr, Q_new[:, :, i_gs_fixed], levels=Q_levels, cmap=cmap,
                vmin=Q_vmin, vmax=Q_vmax, extend='both')
    ax.contour(phi_arr, T_arr, Q_new[:, :, i_gs_fixed], levels=10, colors='k', linewidths=0.5, linestyles='--')
    ax.set_title(f'New Q (gs={gs_arr[i_gs_fixed]/1000:.1f} mm)')
    ax.set_xlabel('Melt Fraction, φ')
    ax.set_yticklabels([])
    
    cbar_q = fig.colorbar(im_q, cax=cax_q)
    cbar_q.set_label('Q')
    cbar_q.set_ticks(np.arange(Q_vmin, Q_vmax + 20, 20))
    
    ax = ax_q_diff
    diff_q = Q_new[:, :, i_gs_fixed] - Q_ref[:, :, i_gs_fixed]
    vmax_diff_q = max(abs(diff_q.min()), abs(diff_q.max()))
    im_q_diff = ax.contourf(phi_arr, T_arr, diff_q, levels=20, cmap='RdBu_r', vmin=-vmax_diff_q, vmax=vmax_diff_q)
    ax.set_title('Q Difference')
    ax.set_xlabel('Melt Fraction, φ')
    ax.set_yticklabels([])
    cbar_q_diff = fig.colorbar(im_q_diff, cax=cax_q_diff)
    cbar_q_diff.set_label('ΔQ')
    
    ax = ax_q_pct
    pct_diff_q = 100 * diff_q / Q_ref[:, :, i_gs_fixed]
    vmax_pct_q = min(100, max(abs(pct_diff_q.min()), abs(pct_diff_q.max())))
    im_q_pct = ax.contourf(phi_arr, T_arr, pct_diff_q, levels=20, cmap='RdBu_r', vmin=-vmax_pct_q, vmax=vmax_pct_q)
    ax.set_title('Q % Difference')
    ax.set_xlabel('Melt Fraction, φ')
    ax.set_yticklabels([])
    cbar_q_pct = fig.colorbar(im_q_pct, cax=cax_q_pct)
    cbar_q_pct.set_label('%')
    
    # Apply axis limits only to ref/new panels
    value_axes = [ax_vs_ref, ax_vs_new, ax_q_ref, ax_q_new]
    for ax in value_axes:
        if phi_lim is not None:
            ax.set_xlim(phi_lim)
        if T_lim is not None:
            ax.set_ylim(T_lim)
    
    fig.suptitle(f'Comparison (T vs φ, gs={gs_arr[i_gs_fixed]/1000:.1f} mm): {method} at P={P_arr[i_P]:.2f} GPa', fontsize=14)
    
    if save_path:
        import os
        if os.path.isdir(save_path) or save_path.endswith('/'):
            os.makedirs(save_path, exist_ok=True)
            filename = f'lut_comparison_T_vs_phi_{method}_P{P_arr[i_P]:.1f}GPa.png'
            save_path = os.path.join(save_path, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved figure to {save_path}')
    else:
        plt.show()
    
    return fig


def _load_sweep_file(filepath: str) -> Dict[str, Any]:
    """
    Load a sweep file in any supported format (.mat, .npz, .pkl/.pickle).

    For .mat files, tries the Python-native loader first (returns a dict),
    falling back to raw loadmat (returns a MATLAB struct object).

    Returns
    -------
    dict or MATLAB struct
        Sweep data usable by the plotting functions.
    """
    import os
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ('.npz', '.pkl', '.pickle'):
        from .generate_sweep import load_sweep
        return load_sweep(filepath)

    # .mat — try the Python-native loader first (gives a dict)
    try:
        from .generate_sweep import load_sweep
        return load_sweep(filepath)
    except Exception:
        # Fall back to raw loadmat (returns MATLAB struct, still works
        # with the isinstance(sweep, dict) branches in the plot funcs)
        mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
        return mat['sweep']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize VBR lookup tables (.mat, .npz, .pkl)')
    parser.add_argument('sweep_file',
                        help='Path to sweep file (.mat, .npz, or .pkl)')
    parser.add_argument('--method', default='andrade_psp',
                        help='Anelastic method (default: andrade_psp)')
    parser.add_argument('--P', type=float, default=2.0,
                        help='Pressure in GPa for the slice')
    parser.add_argument('--compare',
                        help='Reference sweep file for side-by-side comparison')
    parser.add_argument('--save', help='Save path (file or directory) for figure')

    args = parser.parse_args()

    # Load sweep (any format)
    sweep = _load_sweep_file(args.sweep_file)

    if args.compare:
        ref_sweep = _load_sweep_file(args.compare)
        # Compare new sweep against reference - all three slice types
        compare_lut_slices(ref_sweep, sweep, method=args.method, P_GPa=args.P, save_path=args.save)
        compare_lut_slices_gs_phi(ref_sweep, sweep, method=args.method, P_GPa=args.P, save_path=args.save)
        compare_lut_slices_T_phi(ref_sweep, sweep, method=args.method, P_GPa=args.P, save_path=args.save)
    else:
        plot_lut_slices(sweep, method=args.method, P_GPa=args.P, save_path=args.save)
