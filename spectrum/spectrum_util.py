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
    """
    Fit N Lorentzian peaks with iterative strategy exploration
    
    New features:
    1. Peak position constraints via PEAK_POSITION_TOLERANCE
    2. Iterative optimization via FIT_MAX_ITERATIONS
    3. Multiple strategy exploration per iteration
    4. Progressive parameter refinement (gradient-descent-like)
    
    Algorithm:
    - Each iteration tries all strategies (current_best, shift_left, shift_right, narrow, widen, random)
    - Best result from iteration N becomes starting point for iteration N+1
    - Returns best parameters from final iteration
    
    Parameters:
    -----------
    args : Dict[str, Any]
        Config dict with:
        - NUM_PEAKS: Number of peaks to fit
        - PEAK_INITIAL_GUESS: 'auto' or manual wavelengths (nm)
        - PEAK_POSITION_TOLERANCE: None, float, or list (nm range)
        - FIT_MAX_ITERATIONS: Number of iterative refinement cycles
    y : np.ndarray
        Spectrum intensity data
    x : np.ndarray
        Wavelength array (nm)
    num_peaks : int
        Number of peaks to fit
    manual_positions : Optional[List[float]]
        Manual peak positions in nm
    
    Returns:
    --------
    Tuple[np.ndarray, Dict[str, float], float]
        - Fitted curve over full wavelength range
        - Parameter dict: {a1, b1, c1, a2, b2, c2, ...}
        - R-squared goodness of fit (best from all iterations)
    """
    
    # 1. Get fitting range
    fit_min, fit_max = args['FIT_RANGE_NM']
    mask = (x >= fit_min) & (x <= fit_max)
    x_fit = x[mask]
    y_fit = y[mask]
    
    # 2. Define N-peak Lorentzian function
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
    
    # 3. Validate input data
    if len(y_fit) == 0 or np.all(y_fit == 0) or np.isnan(y_fit).any():
        print("[warning] Invalid spectrum data for fitting")
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
    
    # 4. Check for positive values
    if y_fit.max() <= 0:
        print("[warning] No positive values in spectrum")
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
    
    # 5. Generate initial guess
    p0 = generate_initial_guess(y_fit, x_fit, num_peaks, manual_positions)
    p0_original = p0.copy()
    
    # 6. Setup tolerance and bounds
    peak_tolerance = args.get('PEAK_POSITION_TOLERANCE', None)
    
    lower_bounds = [0, x_fit.min(), 0] * num_peaks
    upper_bounds = [np.inf, x_fit.max(), np.inf] * num_peaks
    lower_bounds_original = lower_bounds.copy()
    upper_bounds_original = upper_bounds.copy()
    
    tolerances = None
    if peak_tolerance is not None:
        if isinstance(peak_tolerance, (list, tuple)):
            if len(peak_tolerance) != num_peaks:
                print(f"[warning] PEAK_POSITION_TOLERANCE length mismatch, using first value")
                tolerances = [peak_tolerance[0]] * num_peaks
            else:
                tolerances = peak_tolerance
        else:
            tolerances = [peak_tolerance] * num_peaks
        
        print(f"[debug] Applying peak position tolerance: {tolerances}")
        
        for i in range(num_peaks):
            b_initial = p0[3*i + 1]
            tol = tolerances[i]
            
            lower_bounds[3*i + 1] = max(x_fit.min(), b_initial - tol)
            upper_bounds[3*i + 1] = min(x_fit.max(), b_initial + tol)
            
            print(f"  Peak {i+1}: center={b_initial:.1f} nm, "
                  f"allowed range=[{lower_bounds[3*i + 1]:.1f}, {upper_bounds[3*i + 1]:.1f}] nm")
        
        lower_bounds_original = lower_bounds.copy()
        upper_bounds_original = upper_bounds.copy()
    
    # 7. Iterative fitting with strategy exploration
    max_iterations = args.get('FIT_MAX_ITERATIONS', 1)
    
    # Strategy list (all strategies tried each iteration)
    strategies = ['current_best', 'shift_peaks_left', 'shift_peaks_right', 
                  'narrow_fwhm', 'widen_fwhm', 'random_explore']
    
    # Track best across all iterations
    global_best_result = None
    global_best_r2 = -np.inf
    global_best_params_array = None
    
    # Current iteration's starting point
    current_p0 = p0.copy()
    current_lower_bounds = lower_bounds.copy()
    current_upper_bounds = upper_bounds.copy()
    
    for iteration in range(max_iterations):
        print(f"\n[Iteration {iteration+1}/{max_iterations}]")
        
        iteration_best_result = None
        iteration_best_r2 = -np.inf
        iteration_best_params_array = None
        
        # Try all strategies in this iteration
        for strategy_idx, strategy in enumerate(strategies):
            # Apply strategy to get trial parameters
            if strategy == 'current_best':
                # Use current best parameters as-is
                trial_p0 = current_p0.copy()
                trial_lower = current_lower_bounds.copy()
                trial_upper = current_upper_bounds.copy()
                print(f"  Strategy {strategy_idx+1}/6: {strategy}")
            else:
                # Apply transformation strategy
                trial_p0, trial_lower, trial_upper = apply_strategy_transform(
                    strategy=strategy,
                    p0_current=current_p0,
                    lower_bounds_original=lower_bounds_original,
                    upper_bounds_original=upper_bounds_original,
                    x_fit=x_fit,
                    num_peaks=num_peaks,
                    tolerances=tolerances,
                    iteration=iteration
                )
                print(f"  Strategy {strategy_idx+1}/6: {strategy}")
            
            # Perform fitting
            try:
                popt, pcov = curve_fit(
                    lorentz_n, 
                    x_fit, 
                    y_fit,
                    p0=trial_p0,
                    bounds=(trial_lower, trial_upper),
                    maxfev=10000,
                    method='trf'
                )
                
                # Generate fitted curve
                y_fit_full = lorentz_n(x, *popt)
                y_fit_range = lorentz_n(x_fit, *popt)
                
                # Calculate R²
                ss_res = np.sum((y_fit - y_fit_range)**2)
                ss_tot = np.sum((y_fit - y_fit.mean())**2)
                rsq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                # Update iteration best
                if rsq > iteration_best_r2:
                    iteration_best_r2 = rsq
                    iteration_best_params_array = popt.copy()
                    
                    # Package parameters
                    params = {}
                    for i in range(num_peaks):
                        peak_num = i + 1
                        params[f'a{peak_num}'] = popt[3*i]
                        params[f'b{peak_num}'] = popt[3*i + 1]
                        params[f'c{peak_num}'] = popt[3*i + 2]
                    if num_peaks == 1:
                        params['a'] = params['a1']
                    
                    iteration_best_result = (y_fit_full, params, float(rsq))
                    
                    print(f"    → R²={rsq:.4f} ✓ (NEW ITERATION BEST)")
                    for i in range(num_peaks):
                        peak_num = i + 1
                        print(f"       Peak {peak_num}: λ={params[f'b{peak_num}']:.1f} nm, "
                              f"FWHM={params[f'c{peak_num}']:.1f} nm")
                else:
                    print(f"    → R²={rsq:.4f}")
                
            except Exception as e:
                print(f"    → FAILED: {str(e)}")
        
        # Update global best if this iteration improved
        if iteration_best_r2 > global_best_r2:
            global_best_r2 = iteration_best_r2
            global_best_result = iteration_best_result
            global_best_params_array = iteration_best_params_array
            print(f"  → Iteration {iteration+1} best: R²={iteration_best_r2:.4f} ✓ (GLOBAL BEST)")
        else:
            print(f"  → Iteration {iteration+1} best: R²={iteration_best_r2:.4f} (no global improvement)")
        
        # Prepare for next iteration: use best parameters as starting point
        if iteration_best_params_array is not None:
            current_p0 = iteration_best_params_array.copy()
            
            # Update bounds based on new peak positions (if tolerance is set)
            if tolerances is not None:
                for i in range(num_peaks):
                    b_new = current_p0[3*i + 1]
                    tol = tolerances[i]
                    current_lower_bounds[3*i + 1] = max(x_fit.min(), b_new - tol)
                    current_upper_bounds[3*i + 1] = min(x_fit.max(), b_new + tol)
        
        # Early termination if excellent fit
        if global_best_r2 > 0.99:
            print(f"\n[Early termination] Excellent fit achieved (R² > 0.99)")
            break
    
    # Return best result from all iterations
    if global_best_result is not None:
        print(f"\n[Final Result] Best R²={global_best_r2:.4f} from {iteration+1} iteration(s)")
        return global_best_result
    else:
        print(f"[warning] All fitting attempts failed across {max_iterations} iteration(s)")
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


def apply_strategy_transform(strategy: str,
                             p0_current: List[float],
                             lower_bounds_original: List[float],
                             upper_bounds_original: List[float],
                             x_fit: np.ndarray,
                             num_peaks: int,
                             tolerances: Optional[List[float]],
                             iteration: int) -> Tuple[List[float], List[float], List[float]]:
    """
    Apply strategy transformation to current parameters
    
    Parameters:
    -----------
    strategy : str
        Strategy name ('shift_peaks_left', 'shift_peaks_right', 'narrow_fwhm', 'widen_fwhm', 'random_explore')
    p0_current : List[float]
        Current best parameters from previous strategy/iteration [a1, b1, c1, a2, b2, c2, ...]
    lower_bounds_original : List[float]
        Original lower bounds
    upper_bounds_original : List[float]
        Original upper bounds
    x_fit : np.ndarray
        Wavelength array in fitting range
    num_peaks : int
        Number of peaks
    tolerances : Optional[List[float]]
        Peak position tolerances (None if unconstrained)
    iteration : int
        Current iteration number (for reproducible randomness)
    
    Returns:
    --------
    Tuple[List[float], List[float], List[float]]
        (trial_p0, trial_lower_bounds, trial_upper_bounds)
    """
    p0 = p0_current.copy()
    lower_bounds = lower_bounds_original.copy()
    upper_bounds = upper_bounds_original.copy()
    
    if strategy == 'shift_peaks_left':
        shift_amount = 10  # nm
        for i in range(num_peaks):
            b_current = p0_current[3*i + 1]
            b_new = b_current - shift_amount
            
            # Ensure within bounds
            if tolerances is not None:
                b_new = max(lower_bounds[3*i + 1], min(upper_bounds[3*i + 1], b_new))
            else:
                b_new = max(x_fit.min(), min(x_fit.max(), b_new))
            
            p0[3*i + 1] = b_new
    
    elif strategy == 'shift_peaks_right':
        shift_amount = 10  # nm
        for i in range(num_peaks):
            b_current = p0_current[3*i + 1]
            b_new = b_current + shift_amount
            
            # Ensure within bounds
            if tolerances is not None:
                b_new = max(lower_bounds[3*i + 1], min(upper_bounds[3*i + 1], b_new))
            else:
                b_new = max(x_fit.min(), min(x_fit.max(), b_new))
            
            p0[3*i + 1] = b_new
    
    elif strategy == 'narrow_fwhm':
        # Try narrower peaks (60% of current FWHM)
        for i in range(num_peaks):
            c_current = p0_current[3*i + 2]
            p0[3*i + 2] = c_current * 0.6
    
    elif strategy == 'widen_fwhm':
        # Try broader peaks (150% of current FWHM)
        for i in range(num_peaks):
            c_current = p0_current[3*i + 2]
            p0[3*i + 2] = c_current * 1.5
    
    elif strategy == 'random_explore':
        # Random perturbation (reproducible per iteration)
        np.random.seed(iteration * 100 + 42)
        
        for i in range(num_peaks):
            # Random factors
            amp_factor = 0.5 + np.random.rand() * 1.0  # 0.5 ~ 1.5
            fwhm_factor = 0.7 + np.random.rand() * 0.6  # 0.7 ~ 1.3
            
            p0[3*i] = p0_current[3*i] * amp_factor
            p0[3*i + 2] = p0_current[3*i + 2] * fwhm_factor
            
            # Position: small random shift
            if tolerances is not None:
                shift = (np.random.rand() - 0.5) * tolerances[i] * 0.3
                b_new = p0_current[3*i + 1] + shift
                p0[3*i + 1] = max(lower_bounds[3*i + 1], min(upper_bounds[3*i + 1], b_new))
    
    return p0, lower_bounds, upper_bounds


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
        # Convert wavelength to energy
        x_plot = 1239.842 / x
        # Sort in ascending energy order (reverse of wavelength)
        x_plot = x_plot[::-1]
        y_plot = y[::-1]
        y_fit_plot = y_fit[::-1]
    else:
        x_plot = x
        y_plot = y
        y_fit_plot = y_fit

    # Plot data
    ax.plot(x_plot, y_plot, 'b-', linewidth=3, label='Data')
    
    # Plot fit if requested
    if show_fit:
        ax.plot(x_plot, y_fit_plot, 'k--', linewidth=3, label='Lorentz fit')
    
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
    y_max = max(y_plot.max(), y_fit_plot.max() if show_fit else 0) if len(y_plot) > 0 else 1.0
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