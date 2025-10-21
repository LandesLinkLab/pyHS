import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple, Optional, Any, Union

def fit_lorentz(args: Dict[str, Any], y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Fit Lorentzian function to experimental spectrum data
    
    This function implements MATLAB-compatible Lorentzian fitting using the same
    mathematical form and parameter definitions as the original MATLAB code.
    The fitting is performed over a specified wavelength range but returns
    the fit evaluated over the full wavelength array.
    
    Parameters:
    -----------
    args : Dict[str, Any]
        Configuration dictionary containing:
        - FIT_RANGE_NM: Tuple of (min_wl, max_wl) for fitting range
    y : np.ndarray
        Experimental intensity data (spectrum)
    x : np.ndarray
        Wavelength array corresponding to y
    
    Returns:
    --------
    Tuple[np.ndarray, Dict[str, float], float]
        - y_fit: Fitted Lorentzian curve over full wavelength range
        - params: Dictionary with fitted parameters {'a': amplitude, 'b1': center, 'c1': FWHM}
        - rsq: R-squared goodness of fit value
    
    Notes:
    ------
    The Lorentzian function used is: f(x) = (2*a/π) * (c / (4*(x-b)² + c²))
    where: a = amplitude parameter, b = center wavelength, c = FWHM (Γ)
    """

    # Get fitting range from configuration
    fit_min, fit_max = args['FIT_RANGE_NM']
    mask = (x >= fit_min) & (x <= fit_max)
    x_fit = x[mask]  # Wavelengths for fitting
    y_fit = y[mask]  # Intensities for fitting
        
    def lorentz_matlab_form(x, a, b, c):
        """
        MATLAB-compatible Lorentzian function
        
        Parameters:
        - a: Amplitude scaling factor
        - b: Center wavelength (resonance position)
        - c: Full Width at Half Maximum (FWHM)
        """
        return (2*a/np.pi) * (c / (4*(x-b)**2 + c**2))
    
    # Validate input data
    if len(y_fit) == 0 or np.all(y_fit == 0) or np.isnan(y_fit).any():
        print("[warning] Invalid spectrum data for fitting")
        return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0}, 0.0
    
    # Generate initial parameter estimates
    idx = int(np.argmax(y_fit))  # Index of peak intensity
    if y_fit[idx] <= 0:
        print("[warning] No positive values in spectrum")
        return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0}, 0.0
    
    # Initial parameter guesses
    a0 = float(y_fit[idx] * np.pi / 2)  # Amplitude estimate
    b0 = float(x_fit[idx])              # Center wavelength estimate
    
    # Estimate FWHM from half-maximum width
    half_max = y_fit[idx] / 2
    indices_above_half = np.where(y_fit > half_max)[0]
    if len(indices_above_half) > 1:
        c0 = float(x_fit[indices_above_half[-1]] - x_fit[indices_above_half[0]])
    else:
        c0 = 70.0  # Default FWHM if estimation fails
    
    p0 = [a0, b0, c0]  # Initial parameter vector
    
    try:
        # Set parameter bounds for physical constraints
        bounds = ([0, x_fit.min(), 0],              # Lower bounds
                 [np.inf, x_fit.max(), np.inf])     # Upper bounds
        
        # Perform nonlinear least squares fitting
        popt, pcov = curve_fit(lorentz_matlab_form, x_fit, y_fit, p0=p0, 
                              bounds=bounds, maxfev=8000, 
                              method='trf')
        
        # Generate fit curve over FULL wavelength range (not just fitting range)
        y_fit_full = lorentz_matlab_form(x, *popt)
        
        # Calculate R-squared over fitting range only
        y_fit_range = lorentz_matlab_form(x_fit, *popt)
        ss_res = np.sum((y_fit - y_fit_range)**2)  # Sum of squared residuals
        ss_tot = np.sum((y_fit - y_fit.mean())**2) # Total sum of squares
        rsq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Package parameters in MATLAB-compatible format
        params = {
            'a': popt[0],   # Amplitude parameter
            'b1': popt[1],  # Center wavelength (λ_max)
            'c1': popt[2],  # FWHM (Γ)
        }
        
        return y_fit_full, params, float(rsq)
    
    except Exception as e:
        print(f"[warning] Fitting failed: {str(e)}")
        return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0}, 0.0

def plot_spectrum(x: np.ndarray, 
                y: np.ndarray, 
                y_fit: np.ndarray, 
                title: str, 
                out_png: Path, 
                dpi: int = 300, 
                params: Optional[Dict[str, float]] = None, 
                snr: Optional[float] = None,
                args: Optional[Dict[str, Any]] = None,
                show_fit: bool = True) -> None:
    """
    Create publication-quality spectrum plot identical to MATLAB version
    
    This function generates plots with MATLAB-compatible styling including:
    - Consistent line styles and colors
    - Large font sizes for readability
    - Parameter annotations with proper formatting
    - Fixed axis ranges for comparison
    - Professional appearance for publications
    
    Parameters:
    -----------
    x : np.ndarray
        Wavelength array
    y : np.ndarray
        Experimental spectrum data
    y_fit : np.ndarray
        Fitted Lorentzian curve
    title : str
        Plot title
    out_png : Path
        Output file path for saving
    dpi : int
        Resolution for saved figure (default: 300)
    params : Optional[Dict[str, float]]
        Fitting parameters for annotation
    snr : Optional[float]
        Signal-to-noise ratio for annotation
    args : Optional[Dict[str, Any]]
        Additional configuration parameters
    show_fit : bool
        Whether to show fitted curve (default: True)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    output_unit = args.get('OUTPUT_UNIT')

    if output_unit == 'eV':
        x = 1239.842 / x
        x = x[::-1]
        y = y[::-1]
        y_fit = y_fit[::-1]

    elif output_unit == 'nm':
        pass

    else:
        raise ValueError("Invalid x-axis unit")
    
    # Plot data
    ax.plot(x, y, 'b-', linewidth=3, label='Data')
    
    # Plot fit if requested
    if show_fit:
        ax.plot(x, y_fit, 'k--', linewidth=3, label='Lorentz fit')
    
    # Axis labels with large font sizes (MATLAB-compatible)
    if output_unit == 'eV':
        ax.set_xlabel('Energy (eV)', fontsize=32)
    else:
        ax.set_xlabel('Wavelength (nm)', fontsize=32)

    ax.set_ylabel('Scattering', fontsize=32)
    
    # Large tick labels for readability
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Show all box edges (MATLAB default)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # Add parameter text annotations if available and showing fit
    if show_fit and params is not None and snr is not None:
        lambda_max_nm = params.get('b1', 0)  # Center wavelength
        lambda_max_ev = 1239.842 / lambda_max_nm
        gamma_nm = params.get('c1', 0)       # FWHM
        gamma_eV = 1239.842 / (lambda_max_nm - gamma_nm/2) - 1239.842/(lambda_max_nm + gamma_nm/2)
        
        # Position annotations in upper right area
        if output_unit == 'nm':
            ax.text(0.55, 0.9, f'λ_max = {lambda_max_nm:.0f} nm', transform=ax.transAxes, fontsize=20)
            ax.text(0.55, 0.78, f'Γ = {gamma_nm:.0f} nm', transform=ax.transAxes, fontsize=20)
            ax.text(0.55, 0.66, f'S/N = {snr:.0f}', transform=ax.transAxes, fontsize=20)

        elif output_unit == 'eV':
            ax.text(0.55, 0.9, f'E_max = {lambda_max_ev:.3f} eV', transform=ax.transAxes, fontsize=20)
            ax.text(0.55, 0.78, f'Γ = {gamma_eV:.3f} eV', transform=ax.transAxes, fontsize=20)
            ax.text(0.55, 0.66, f'S/N = {snr:.0f}', transform=ax.transAxes, fontsize=20)
    
    # Set fixed axis ranges for consistency across plots
    xmin, xmax = args['CROP_RANGE_NM']
    if output_unit == 'nm':
        ax.set_xlim(xmin, xmax)
    elif output_unit == 'eV':
        ax.set_xlim(1239.842 / xmax, 1239.842 / xmin)
    
    # Set Y-axis range with error protection
    y_max = max(y.max(), y_fit.max() if show_fit else 0) if len(y) > 0 else 1.0
    if y_max <= 0:
        y_max = 1.0
    ax.set_ylim(0, y_max * 1.05)  # 5% padding above maximum
    
    # Add title
    ax.set_title(title, fontsize=16)
    
    # Remove grid for clean appearance
    ax.grid(False)
    
    # Save figure with tight layout
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def save_dfs_particle_map(max_map: np.ndarray, 
                        representatives: List[Dict[str, Any]], 
                        output_path: Path, 
                        sample_name: str,
                        args: Optional[Dict[str, Any]] = None) -> None:
    """
    Save annotated particle map showing all analyzed particles with markers
    
    This function creates a publication-ready particle map that includes:
    - Background intensity map with optimized contrast
    - Numbered particle markers with consistent styling
    - Resonance wavelength annotations
    - Professional color scheme and formatting
    - Scalable vector graphics elements for publication use
    
    Parameters:
    -----------
    max_map : np.ndarray
        2D maximum intensity map as background
    representatives : List[Dict[str, Any]]
        List of representative particle data including positions and parameters
    output_path : Path
        Full path for saving the output image
    sample_name : str
        Sample name for plot title
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set dynamic contrast based on non-zero values for better visibility
    if np.any(max_map > 0):
        vmin, vmax = np.percentile(max_map[max_map > 0], [5, 95])
    else:
        vmin, vmax = (0, 1)
    
    # Display intensity map with hot colormap (standard for thermal imaging)
    im = ax.imshow(max_map,
                   cmap='hot',                    # Hot colormap for intensity
                   origin='lower',                # Origin at bottom-left
                   vmin=vmin, vmax=vmax,         # Dynamic range
                   interpolation='nearest')       # No interpolation for pixel data
    
    # Add colorbar with proper formatting
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Max Intensity', fontsize=12)
    
    output_unit = args.get('OUTPUT_UNIT')

    # Add particle markers and annotations
    for i, rep in enumerate(representatives):
        row, col = rep['row'], rep['col']
        
        # White circle marker for high visibility on hot colormap
        particle_num = i + 1
        circle_inner = plt.Circle((col, row), 
                                 radius=1.5,
                                 edgecolor='white',
                                 facecolor='none',
                                 linewidth=2)
        ax.add_patch(circle_inner)
        
        # Particle number label with black background for readability
        ax.text(col - 1, row + 3,
                f'{particle_num}',
                color='white',
                fontsize=6,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        if output_unit == 'eV':
            
            energy = 1239.842 / rep['peak_wl']
            ax.text(col - 3, row - 3,
                    f'{energy:.3f} eV',
                    color='yellow',
                    fontsize=6,
                    fontweight='bold',
                    ha='left')

        else:

            ax.text(col - 3, row - 3,
                    f'{rep["peak_wl"]:.0f} nm',
                    color='yellow',
                    fontsize=6,
                    fontweight='bold',
                    ha='left')
    
    # Title and axis labels
    ax.set_title(f'{sample_name} - DFS Particle Map ({len(representatives)} particles)', 
                fontsize=16, pad=10)
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    
    # Add grid for easier position reading
    ax.grid(True, alpha=0.3, linestyle='--', color='white')
    
    # Save with high quality
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"[info] Saved DFS particle map: {output_path}")