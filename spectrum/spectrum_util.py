import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from typing import List, Dict, Tuple, Optional, Any, Union

def fit_lorentz(args: Dict[str, Any], y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Fit N Lorentzian peaks to experimental spectrum data
    
    This function implements flexible multi-peak Lorentzian fitting that supports:
    - Arbitrary number of peaks (1, 2, 3, 4, ...)
    - Automatic peak detection or manual initial guess
    - MATLAB-compatible single peak mode for backward compatibility
    
    Parameters:
    -----------
    args : Dict[str, Any]
        Configuration dictionary containing:
        - NUM_PEAKS: Number of peaks to fit (default: 1)
        - PEAK_INITIAL_GUESS: 'auto' or list of wavelengths in nm
        - FIT_RANGE_NM: Tuple of (min_wl, max_wl) for fitting range
    y : np.ndarray
        Experimental intensity data (spectrum)
    x : np.ndarray
        Wavelength array corresponding to y
    
    Returns:
    --------
    Tuple[np.ndarray, Dict[str, float], float]
        - y_fit: Fitted curve over full wavelength range
        - params: Dictionary with fitted parameters
                  Single peak: {'a', 'b1', 'c1'}
                  Multi peak: {'a1', 'b1', 'c1', 'a2', 'b2', 'c2', ...}
        - rsq: R-squared goodness of fit value
    """
    num_peaks = args.get('NUM_PEAKS', 1)
    peak_guess = args.get('PEAK_INITIAL_GUESS', 'auto')
    
    # Validate consistency
    if peak_guess != 'auto' and not isinstance(peak_guess, (list, tuple)):
        raise ValueError(f"PEAK_INITIAL_GUESS must be 'auto' or a list, got {type(peak_guess)}")
    
    if isinstance(peak_guess, (list, tuple)):
        if len(peak_guess) != num_peaks:
            raise ValueError(
                f"PEAK_INITIAL_GUESS length ({len(peak_guess)}) "
                f"must match NUM_PEAKS ({num_peaks})"
            )
        manual_positions = peak_guess
    else:
        manual_positions = None
    
    return fit_n_lorentz(args, y, x, num_peaks, manual_positions)


def generate_initial_guess(y_fit: np.ndarray, 
                          x_fit: np.ndarray, 
                          num_peaks: int, 
                          manual_positions: Optional[List[float]] = None) -> List[float]:
    """
    Generate initial parameter guesses for N-peak Lorentzian fitting
    
    Parameters:
    -----------
    y_fit : np.ndarray
        Spectrum data in fitting range
    x_fit : np.ndarray
        Wavelength array in fitting range
    num_peaks : int
        Number of peaks to find
    manual_positions : Optional[List[float]]
        Manual peak positions in wavelength (nm)
    
    Returns:
    --------
    List[float]
        Initial parameters [a1, b1, c1, a2, b2, c2, ...]
    """
    if manual_positions is not None:
        # Mode B: Manual specification
        print(f"[debug] Using manual peak positions: {manual_positions}")
        p0 = []
        
        for pos in manual_positions:
            # Find closest index to specified position
            idx = np.argmin(np.abs(x_fit - pos))
            
            # Initial guess for this peak
            a0 = max(y_fit[idx] * np.pi / 2, 0.1)  # Ensure positive
            b0 = x_fit[idx]  # Use actual wavelength at this index
            c0 = 50.0  # Default FWHM in nm
            
            p0.extend([a0, b0, c0])
            print(f"  Peak at {pos} nm → idx={idx}, wl={b0:.1f} nm, amp={a0:.2f}")
        
        return p0
    
    else:
        # Mode A: Automatic peak detection
        print(f"[debug] Auto-detecting {num_peaks} peaks...")
        
        # Find peaks with scipy
        peaks_idx, properties = find_peaks(
            y_fit,
            distance=10,  # Minimum distance between peaks
            prominence=y_fit.max() * 0.05  # At least 5% of maximum
        )
        
        print(f"[debug] Found {len(peaks_idx)} candidate peaks")
        
        if len(peaks_idx) < num_peaks:
            print(f"[warning] Only found {len(peaks_idx)} peaks, needed {num_peaks}")
            print(f"[warning] Filling missing peaks with uniform distribution")
            
            # Use found peaks and fill rest with uniform distribution
            selected_peaks = list(peaks_idx)
            
            # Add uniformly spaced peaks in remaining space
            missing = num_peaks - len(peaks_idx)
            if missing > 0:
                # Distribute evenly across spectrum
                spacing = len(x_fit) // (missing + 1)
                for i in range(1, missing + 1):
                    extra_idx = i * spacing
                    if extra_idx < len(x_fit):
                        selected_peaks.append(extra_idx)
            
            selected_peaks = np.array(selected_peaks[:num_peaks])
        
        elif len(peaks_idx) > num_peaks:
            # More peaks found than needed - select highest ones
            peak_heights = y_fit[peaks_idx]
            sorted_indices = np.argsort(peak_heights)[-num_peaks:]
            selected_peaks = peaks_idx[sorted_indices]
        
        else:
            # Exact match
            selected_peaks = peaks_idx
        
        # Sort by wavelength (ascending)
        selected_peaks = np.sort(selected_peaks)
        
        # Generate initial parameters
        p0 = []
        for idx in selected_peaks:
            a0 = max(y_fit[idx] * np.pi / 2, 0.1)
            b0 = x_fit[idx]
            c0 = 50.0
            
            p0.extend([a0, b0, c0])
            print(f"  Peak at idx={idx}, wl={b0:.1f} nm, amp={a0:.2f}")
        
        return p0


def fit_n_lorentz(args: Dict[str, Any], 
                  y: np.ndarray, 
                  x: np.ndarray, 
                  num_peaks: int,
                  manual_positions: Optional[List[float]] = None) -> Tuple[np.ndarray, Dict[str, float], float]:
    """Generic N-peak Lorentzian fitting function"""
    
    # Get fitting range
    fit_min, fit_max = args['FIT_RANGE_NM']
    mask = (x >= fit_min) & (x <= fit_max)
    x_fit = x[mask]
    y_fit = y[mask]
    
    # Define N-peak Lorentzian function
    def lorentz_n(x_val, *params):
        """Sum of N Lorentzian functions"""
        result = np.zeros_like(x_val, dtype=float)
        n = len(params) // 3
        
        for i in range(n):
            a = params[3*i]
            b = params[3*i + 1]
            c = params[3*i + 2]
            result += (2*a/np.pi) * (c / (4*(x_val-b)**2 + c**2))
        
        return result
    
    # Validate input data
    if len(y_fit) == 0 or np.all(y_fit == 0) or np.isnan(y_fit).any():
        print("[warning] Invalid spectrum data for fitting")
        # ✓ 수정 1
        params = {}
        for i in range(num_peaks):
            peak_num = i + 1
            params[f'a{peak_num}'] = 0
            params[f'b{peak_num}'] = 0
            params[f'c{peak_num}'] = 0
        if num_peaks == 1:
            params['a'] = 0
            params['b1'] = 0
            params['c1'] = 0
        return np.zeros_like(y), params, 0.0
    
    # Check for positive values
    if y_fit.max() <= 0:
        print("[warning] No positive values in spectrum")
        # ✓ 수정 2
        params = {}
        for i in range(num_peaks):
            peak_num = i + 1
            params[f'a{peak_num}'] = 0
            params[f'b{peak_num}'] = 0
            params[f'c{peak_num}'] = 0
        if num_peaks == 1:
            params['a'] = 0
            params['b1'] = 0
            params['c1'] = 0
        return np.zeros_like(y), params, 0.0
    
    # Generate initial guess
    p0 = generate_initial_guess(y_fit, x_fit, num_peaks, manual_positions)
    
    # Set bounds for all parameters
    lower_bounds = [0, x_fit.min(), 0] * num_peaks
    upper_bounds = [np.inf, x_fit.max(), np.inf] * num_peaks
    
    try:
        # Perform fitting
        popt, pcov = curve_fit(
            lorentz_n, 
            x_fit, 
            y_fit,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000,
            method='trf'
        )
        
        # Generate fitted curve over FULL wavelength range
        y_fit_full = lorentz_n(x, *popt)
        
        # Calculate R-squared over fitting range only
        y_fit_range = lorentz_n(x_fit, *popt)
        ss_res = np.sum((y_fit - y_fit_range)**2)
        ss_tot = np.sum((y_fit - y_fit.mean())**2)
        rsq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Package parameters into dictionary
        params = {}
        for i in range(num_peaks):
            peak_num = i + 1
            params[f'a{peak_num}'] = popt[3*i]
            params[f'b{peak_num}'] = popt[3*i + 1]
            params[f'c{peak_num}'] = popt[3*i + 2]
        
        # Backward compatibility: for single peak, also provide non-indexed keys
        if num_peaks == 1:
            params['a'] = params['a1']
            # b1 and c1 already exist
        
        print(f"[debug] Fitting successful: R²={rsq:.4f}")
        for i in range(num_peaks):
            peak_num = i + 1
            print(f"  Peak {peak_num}: λ={params[f'b{peak_num}']:.1f} nm, "
                  f"FWHM={params[f'c{peak_num}']:.1f} nm")
        
        return y_fit_full, params, float(rsq)
    
    except Exception as e:
        print(f"[warning] {num_peaks}-peak Lorentzian fitting failed: {str(e)}")
        
        # ✓ 수정 3
        params = {}
        for i in range(num_peaks):
            peak_num = i + 1
            params[f'a{peak_num}'] = 0
            params[f'b{peak_num}'] = 0
            params[f'c{peak_num}'] = 0
        if num_peaks == 1:
            params['a'] = 0
            params['b1'] = 0
            params['c1'] = 0
        
        return np.zeros_like(y), params, 0.0


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
    Create publication-quality spectrum plot
    
    Handles both single and multi-peak display with appropriate annotations.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    output_unit = args.get('OUTPUT_UNIT', 'nm')

    if output_unit == 'eV':
        x = 1239.842 / x
        x = x[::-1]
        y = y[::-1]
        y_fit = y_fit[::-1]

    # Plot data
    ax.plot(x, y, 'b-', linewidth=3, label='Data')
    
    # Plot fit if requested
    if show_fit:
        ax.plot(x, y_fit, 'k--', linewidth=3, label='Lorentz fit')
    
    # Axis labels
    if output_unit == 'eV':
        ax.set_xlabel('Energy (eV)', fontsize=32)
    else:
        ax.set_xlabel('Wavelength (nm)', fontsize=32)

    ax.set_ylabel('Scattering', fontsize=32)
    
    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Show all box edges
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # Add parameter annotations if available
    if show_fit and params is not None and snr is not None:
        # Determine number of peaks from params
        num_peaks = sum(1 for key in params.keys() if key.startswith('b'))
        
        text_y_start = 0.9
        text_y_step = 0.12
        
        for i in range(num_peaks):
            peak_num = i + 1
            
            # Get parameters for this peak
            b_key = f'b{peak_num}'
            c_key = f'c{peak_num}'
            
            if b_key in params and c_key in params:
                lambda_max_nm = params[b_key]
                gamma_nm = params[c_key]
                
                if output_unit == 'nm':
                    text_y = text_y_start - i * text_y_step
                    ax.text(0.55, text_y, 
                           f'Peak {peak_num}: λ={lambda_max_nm:.0f} nm, Γ={gamma_nm:.0f} nm',
                           transform=ax.transAxes, fontsize=18)
                
                elif output_unit == 'eV':
                    lambda_max_ev = 1239.842 / lambda_max_nm
                    gamma_eV = abs(1239.842 / (lambda_max_nm - gamma_nm/2) - 
                                  1239.842 / (lambda_max_nm + gamma_nm/2))
                    
                    text_y = text_y_start - i * text_y_step
                    ax.text(0.55, text_y,
                           f'Peak {peak_num}: E={lambda_max_ev:.3f} eV, Γ={gamma_eV:.3f} eV',
                           transform=ax.transAxes, fontsize=18)
        
        # Add SNR below all peaks
        snr_y = text_y_start - num_peaks * text_y_step
        ax.text(0.55, snr_y, f'S/N = {snr:.0f}', 
               transform=ax.transAxes, fontsize=20)
    
    # Set axis ranges
    xmin, xmax = args['CROP_RANGE_NM']
    if output_unit == 'nm':
        ax.set_xlim(xmin, xmax)
    elif output_unit == 'eV':
        ax.set_xlim(1239.842 / xmax, 1239.842 / xmin)
    
    # Y-axis range
    y_max = max(y.max(), y_fit.max() if show_fit else 0) if len(y) > 0 else 1.0
    if y_max <= 0:
        y_max = 1.0
    ax.set_ylim(0, y_max * 1.05)
    
    # Title
    ax.set_title(title, fontsize=16)
    
    # No grid
    ax.grid(False)
    
    # Save
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
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Dynamic contrast
    if np.any(max_map > 0):
        vmin, vmax = np.percentile(max_map[max_map > 0], [5, 95])
    else:
        vmin, vmax = (0, 1)
    
    # Display intensity map
    im = ax.imshow(max_map,
                   cmap='hot',
                   origin='lower',
                   vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Max Intensity', fontsize=12)
    
    output_unit = args.get('OUTPUT_UNIT', 'nm')

    # Add particle markers
    for i, rep in enumerate(representatives):
        row, col = rep['row'], rep['col']
        
        particle_num = i + 1
        circle = plt.Circle((col, row), 
                           radius=1.5,
                           edgecolor='white',
                           facecolor='none',
                           linewidth=2)
        ax.add_patch(circle)
        
        # Particle number
        ax.text(col - 1, row + 3,
                f'{particle_num}',
                color='white',
                fontsize=6,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Peak wavelength (only show first peak for multi-peak)
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
    
    # Title and labels
    ax.set_title(f'{sample_name} - DFS Particle Map ({len(representatives)} particles)', 
                fontsize=16, pad=10)
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', color='white')
    
    # Save
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"[info] Saved DFS particle map: {output_path}")