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


def plot_spectrum(wavelengths: np.ndarray,
                  spectrum: np.ndarray,
                  fit: np.ndarray,
                  title: str,
                  save_path: Path,
                  dpi: int = 300,
                  params: Optional[Dict[str, float]] = None,
                  snr: Optional[float] = None,
                  args: Optional[Dict[str, Any]] = None,
                  show_fit: bool = True):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get output unit
    output_unit = args.get('OUTPUT_UNIT', 'eV') if args else 'eV'
    x_label = 'Energy (eV)' if output_unit == 'eV' else 'Wavelength (nm)'
    
    # Get fitting model
    fitting_model = args.get('FITTING_MODEL', 'lorentzian') if args else 'lorentzian'
    
    # ✅✅✅ 수정: eV 변환 시 X축 데이터와 Y축 데이터를 정렬해서 변환
    if output_unit == 'eV':
        # nm → eV 변환
        energy = 1239.842 / wavelengths
        
        # energy 증가 순서로 정렬
        sort_idx = np.argsort(energy)
        x_data = energy[sort_idx]
        spectrum_sorted = spectrum[sort_idx]
        fit_sorted = fit[sort_idx] if fit is not None else None
    else:
        # nm 단위는 그대로 사용
        x_data = wavelengths
        spectrum_sorted = spectrum
        fit_sorted = fit
    
    # Plot experimental data
    ax.plot(x_data, spectrum_sorted, 'ko', markersize=4, alpha=0.6, label='Experimental')
    
    if show_fit and fit_sorted is not None:
        # Plot total fit
        ax.plot(x_data, fit_sorted, 'b-', linewidth=2, label='Total Fit')
        
        # =====================================================================
        # NEW: Plot individual components based on fitting model
        # =====================================================================
        
        if fitting_model == 'fano':
            # ============================================
            # FANO MODEL: Plot bright and dark components
            # ============================================
            
            if params is not None:
                # Count bright and dark modes
                num_bright = 0
                num_dark = 0
                
                for key in params.keys():
                    if key.startswith('bright') and key.endswith('_c'):
                        num_bright += 1
                    elif key.startswith('dark') and key.endswith('_d'):
                        num_dark += 1
                
                print(f"[debug plot] Fano model: {num_bright} bright, {num_dark} dark modes")
                
                # Plot bright modes
                for i in range(num_bright):
                    c = params.get(f'bright{i+1}_c', 0)
                    lam = params.get(f'bright{i+1}_lambda', 0)  # nm 기준
                    gamma = params.get(f'bright{i+1}_gamma', 0) # nm 기준
                    
                    if lam > 0 and gamma > 0:
                        # 1) 계산은 항상 nm에서
                        x_vals_nm = wavelengths  # nm
                        A_bright = c * (gamma/2) / (x_vals_nm - lam + 1j*gamma/2)
                        I_bright = np.abs(A_bright)**2

                        # 2) 플롯 직전에만 단위 변환 + 정렬
                        if output_unit == 'eV':
                            x_vals_plot = 1239.842 / x_vals_nm
                            order = np.argsort(x_vals_plot)
                            x_vals_plot = x_vals_plot[order]
                            I_bright = I_bright[order]
                            label_lam = 1239.842 / lam
                            unit_tag = 'eV'
                        else:
                            x_vals_plot = x_vals_nm
                            label_lam = lam
                            unit_tag = 'nm'

                        ax.plot(x_vals_plot, I_bright, '--', linewidth=1.5, label=f'Bright {i+1} ({label_lam:.3f} {unit_tag})', alpha=0.7, color='green')

                # Plot dark modes
                for j in range(num_dark):
                    d = params.get(f'dark{j+1}_d', 0)
                    lam = params.get(f'dark{j+1}_lambda', 0)
                    Gamma = params.get(f'dark{j+1}_Gamma', 0)
                    theta = params.get(f'dark{j+1}_theta', 0)
                    
                    if lam > 0 and Gamma > 0:
                        # 1) 계산은 항상 nm에서
                        x_vals_nm = wavelengths  # nm
                        A_dark = d * np.exp(1j * theta) * (Gamma/2) / (x_vals_nm - lam + 1j*Gamma/2)
                        I_dark = np.abs(A_dark)**2
                        
                        # 2) 플롯 직전에만 단위 변환 + 정렬
                        if output_unit == 'eV':
                            x_vals_plot = 1239.842 / x_vals_nm
                            order = np.argsort(x_vals_plot)
                            x_vals_plot = x_vals_plot[order]
                            I_dark = I_dark[order]
                            label_lam = 1239.842 / lam
                            unit_tag = 'eV'
                        else:
                            x_vals_plot = x_vals_nm
                            label_lam = lam
                            unit_tag = 'nm'
                        
                        theta_pi = theta / np.pi
                        ax.plot(x_vals_plot, I_dark, ':', linewidth=2, label=f'Dark {j+1} ({label_lam:.3f} {unit_tag}, θ={theta_pi:.2f}π)', alpha=0.7, color='red')
        
        elif fitting_model == 'lorentzian':
            # ============================================
            # LORENTZIAN MODEL: Plot individual peaks
            # ============================================
            
            if params is not None:
                # Count peaks
                num_peaks = 0
                for key in params.keys():
                    if key.startswith('b') and len(key) == 2:  # b1, b2, b3, ...
                        num_peaks += 1
                
                print(f"[debug plot] Lorentzian model: {num_peaks} peaks")
                
                # Plot each peak
                for i in range(1, num_peaks + 1):
                    a = params.get(f'a{i}', 0)
                    b = params.get(f'b{i}', 0)
                    c = params.get(f'c{i}', 0)
                    
                    if a > 0 and b > 0 and c > 0:
                        # 1) Lorentzian 계산은 항상 nm에서
                        x_vals_nm = wavelengths  # nm
                        component = (2*a/np.pi) * (c / (4*(x_vals_nm - b)**2 + c**2))
                        
                        # 2) 플롯 직전에만 단위 변환 + 정렬
                        if output_unit == 'eV':
                            x_vals_plot = 1239.842 / x_vals_nm
                            order = np.argsort(x_vals_plot)
                            x_vals_plot = x_vals_plot[order]
                            component = component[order]
                            label_b = 1239.842 / b
                            unit_tag = 'eV'
                        else:
                            x_vals_plot = x_vals_nm
                            label_b = b
                            unit_tag = 'nm'
                        
                        ax.plot(x_vals_plot, component, '--', linewidth=1.5, 
                               label=f'Peak {i} ({label_b:.3f} {unit_tag})', alpha=0.7)
        
        # =====================================================================
        # Add fitting parameters as text annotation
        # =====================================================================
        
        if params is not None:
            # Calculate R² from fit
            ss_res = np.sum((spectrum - fit)**2)
            ss_tot = np.sum((spectrum - spectrum.mean())**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Build parameter text
            param_text = f'R² = {r2:.4f}\n'
            
            if fitting_model == 'fano':
                # Fano parameters
                param_text += f'\n[Bright Modes]\n'
                for i in range(num_bright):
                    lam = params.get(f'bright{i+1}_lambda', 0)
                    gamma = params.get(f'bright{i+1}_gamma', 0)
                    c = params.get(f'bright{i+1}_c', 0)
                    
                    if output_unit == 'eV' and lam > 0:
                        lam_display = 1239.842 / lam
                        param_text += f'  B{i+1}: {lam_display:.3f} eV\n'
                        param_text += f'      γ={gamma:.1f} nm, c={c:.2f}\n'
                    else:
                        param_text += f'  B{i+1}: {lam:.1f} nm\n'
                        param_text += f'      γ={gamma:.1f} nm, c={c:.2f}\n'
                
                param_text += f'\n[Dark Modes]\n'
                for j in range(num_dark):
                    lam = params.get(f'dark{j+1}_lambda', 0)
                    Gamma = params.get(f'dark{j+1}_Gamma', 0)
                    d = params.get(f'dark{j+1}_d', 0)
                    theta = params.get(f'dark{j+1}_theta', 0)
                    
                    if output_unit == 'eV' and lam > 0:
                        lam_display = 1239.842 / lam
                        param_text += f'  D{j+1}: {lam_display:.3f} eV\n'
                        param_text += f'      Γ={Gamma:.1f} nm\n'
                        param_text += f'      d={d:.2f}, θ={theta:.2f} rad\n'
                    else:
                        param_text += f'  D{j+1}: {lam:.1f} nm\n'
                        param_text += f'      Γ={Gamma:.1f} nm\n'
                        param_text += f'      d={d:.2f}, θ={theta:.2f} rad\n'
            
            elif fitting_model == 'lorentzian':
                # Lorentzian parameters
                for i in range(1, num_peaks + 1):
                    b = params.get(f'b{i}', 0)
                    c = params.get(f'c{i}', 0)
                    
                    if output_unit == 'eV' and b > 0:
                        b_display = 1239.842 / b
                        param_text += f'Peak {i}: {b_display:.3f} eV\n'
                        if c > 0:
                            c_ev = abs(1239.842/(b - c/2) - 1239.842/(b + c/2))
                            param_text += f'  FWHM: {c_ev:.3f} eV\n'
                    else:
                        param_text += f'Peak {i}: {b:.1f} nm\n'
                        param_text += f'  FWHM: {c:.1f} nm\n'
            
            if snr:
                param_text += f'\nSNR: {snr:.1f}'
            
            # Add text box
            ax.text(0.02, 0.98, param_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Labels and legend
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Scattering (a.u.)', fontsize=14)
    ax.set_title(title, fontsize=14, pad=15)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


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


# ============================================================================
# FANO RESONANCE FITTING (Physical Interference Model)
# ============================================================================

def fit_fano(args: Dict[str, Any], y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Fit Fano resonance using Physical Interference Model: I = |Σ A_i|²
    
    Two-step fitting procedure:
    Step 1: Fit bright modes only (phase = 0)
    Step 2: Add dark modes with fitted phase
    
    Parameters:
    -----------
    args : Dict[str, Any]
        Configuration dictionary containing:
        - NUM_BRIGHT_MODES: Number of bright modes (phase = 0)
        - BRIGHT_INITIAL_GUESS: List of wavelengths in nm (REQUIRED)
        - BRIGHT_POSITION_TOLERANCE: List or float (nm range constraint)
        - NUM_DARK_MODES: Number of dark modes (phase fitted)
        - DARK_INITIAL_GUESS: List of wavelengths in nm (REQUIRED)
        - DARK_POSITION_TOLERANCE: List or float (nm range constraint)
        - FANO_PHI_INIT: Initial phase for dark modes (default: π)
        - FANO_Q_RANGE: Amplitude range (default: (-20, 20))
        - FANO_PHI_RANGE: Phase range (default: (0, 2π))
        - FANO_GAMMA_RANGE: Linewidth range in nm (default: (5, 100))
        - FIT_RANGE_NM: Tuple of (min_wl, max_wl) for fitting range
        - FANO_DEBUG: If True, print detailed debug info
    
    y : np.ndarray
        Experimental intensity data (spectrum)
    x : np.ndarray
        Wavelength array corresponding to y
    
    Returns:
    --------
    Tuple[np.ndarray, Dict[str, float], float]
        - y_fit: Fitted curve over full wavelength range
        - params: Dictionary with fitted parameters
                  Format: {'bright1_c', 'bright1_lambda', 'bright1_gamma',
                          'dark1_d', 'dark1_lambda', 'dark1_Gamma', 'dark1_theta', ...}
        - rsq: R-squared goodness of fit value
    
    Physical Model:
    ---------------
    Bright mode: A_bright^(i) = c_i × (γ_i/2) / (λ - λ_i + i×γ_i/2)  [phase = 0]
    Dark mode:   A_dark^(j) = d_j × exp(i×θ_j) × (Γ_j/2) / (λ - λ_j + i×Γ_j/2)
    Total:       I(λ) = |Σ A_bright^(i) + Σ A_dark^(j)|²
    """
    
    # Validate configuration
    num_bright = args.get('NUM_BRIGHT_MODES', 0)
    num_dark = args.get('NUM_DARK_MODES', 0)
    bright_guess = args.get('BRIGHT_INITIAL_GUESS', None)
    dark_guess = args.get('DARK_INITIAL_GUESS', None)
    debug = args.get('FANO_DEBUG', False)
    
    if num_bright == 0 and num_dark == 0:
        raise ValueError("At least one bright or dark mode must be specified")
    
    if bright_guess is None and num_bright > 0:
        raise ValueError("BRIGHT_INITIAL_GUESS is REQUIRED when NUM_BRIGHT_MODES > 0")
    
    if dark_guess is None and num_dark > 0:
        raise ValueError("DARK_INITIAL_GUESS is REQUIRED when NUM_DARK_MODES > 0")
    
    if bright_guess is not None and len(bright_guess) != num_bright:
        raise ValueError(f"BRIGHT_INITIAL_GUESS length ({len(bright_guess)}) must match NUM_BRIGHT_MODES ({num_bright})")
    
    if dark_guess is not None and len(dark_guess) != num_dark:
        raise ValueError(f"DARK_INITIAL_GUESS length ({len(dark_guess)}) must match NUM_DARK_MODES ({num_dark})")
    
    # Get fitting range
    fit_min, fit_max = args['FIT_RANGE_NM']
    mask = (x >= fit_min) & (x <= fit_max)
    x_fit = x[mask]
    y_fit = y[mask]
    
    # Validate input data
    if len(y_fit) == 0 or np.all(y_fit == 0) or np.isnan(y_fit).any():
        print("[warning] Invalid spectrum data for Fano fitting")
        return np.zeros_like(y), {}, 0.0
    
    if y_fit.max() <= 0:
        print("[warning] No positive values in spectrum")
        return np.zeros_like(y), {}, 0.0
    
    if debug:
        print("\n" + "="*60)
        print("FANO RESONANCE FITTING (Two-Step)")
        print("="*60)
        print(f"Bright modes: {num_bright}, positions: {bright_guess}")
        print(f"Dark modes: {num_dark}, positions: {dark_guess}")
    
    # Step 1: Fit bright modes only
    if num_bright > 0:
        if debug:
            print("\n[STEP 1] Fitting bright modes only...")
        
        y_fit_step1, params_step1, r2_step1 = fit_fano_bright_only(
            args, y_fit, x_fit, num_bright, bright_guess
        )
        
        if debug:
            print(f"Step 1 R² = {r2_step1:.4f}")
            for i in range(num_bright):
                print(f"  Bright {i+1}: λ={params_step1[f'bright{i+1}_lambda']:.1f} nm, "
                      f"γ={params_step1[f'bright{i+1}_gamma']:.1f} nm, "
                      f"c={params_step1[f'bright{i+1}_c']:.3f}")
    else:
        params_step1 = {}
        r2_step1 = 0.0
    
    # Step 2: Add dark modes
    if num_dark > 0:
        if debug:
            print("\n[STEP 2] Adding dark modes...")
        
        y_fit_step2, params_step2, r2_step2 = fit_fano_with_dark(
            args, y_fit, x_fit, num_bright, num_dark, 
            bright_guess, dark_guess, params_step1
        )
        
        if debug:
            print(f"Step 2 R² = {r2_step2:.4f}")
            for i in range(num_dark):
                print(f"  Dark {i+1}: λ={params_step2[f'dark{i+1}_lambda']:.1f} nm, "
                      f"Γ={params_step2[f'dark{i+1}_Gamma']:.1f} nm, "
                      f"d={params_step2[f'dark{i+1}_d']:.3f}, "
                      f"θ={params_step2[f'dark{i+1}_theta']:.3f} rad")
        
        # Generate full fitted curve
        y_fit_full = fano_model_full(x, num_bright, num_dark, params_step2)
        
        return y_fit_full, params_step2, r2_step2
    
    else:
        # Only bright modes
        y_fit_full = fano_model_bright_only(x, num_bright, params_step1)
        return y_fit_full, params_step1, r2_step1


def fit_fano_bright_only(args: Dict[str, Any], 
                         y_fit: np.ndarray, 
                         x_fit: np.ndarray,
                         num_bright: int,
                         bright_guess: List[float]) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Step 1: Fit bright modes only (phase = 0)
    
    Model: I = |Σ c_i × (γ_i/2) / (λ - λ_i + i×γ_i/2)|²
    
    Parameters to fit per bright mode: [c_i, λ_i, γ_i]
    Total parameters: 3 × num_bright
    """
    
    # Define bright-only model
    def fano_bright(x_val, *params):
        """Bright modes: phase = 0"""
        A_total = np.zeros_like(x_val, dtype=complex)
        
        for i in range(num_bright):
            c = params[3*i]
            lam = params[3*i + 1]
            gamma = params[3*i + 2]
            
            # Complex amplitude (phase = 0 for bright modes)
            A_i = c * (gamma/2) / (x_val - lam + 1j*gamma/2)
            A_total += A_i
        
        # Intensity: |A|²
        I = np.abs(A_total)**2
        return I
    
    # Initial guess
    p0 = []
    for i in range(num_bright):
        c0 = 1.0  # Default amplitude
        lam0 = bright_guess[i]
        gamma0 = 30.0  # Default linewidth
        p0.extend([c0, lam0, gamma0])
    
    # Bounds
    q_range = args.get('FANO_Q_RANGE', (-20, 20))
    gamma_range = args.get('FANO_GAMMA_RANGE', (5, 100))
    
    lower_bounds = []
    upper_bounds = []
    
    # Apply position tolerance
    bright_tol = args.get('BRIGHT_POSITION_TOLERANCE', None)
    if bright_tol is not None:
        if isinstance(bright_tol, (list, tuple)):
            tolerances = bright_tol if len(bright_tol) == num_bright else [bright_tol[0]] * num_bright
        else:
            tolerances = [bright_tol] * num_bright
    else:
        tolerances = [np.inf] * num_bright
    
    for i in range(num_bright):
        lam0 = bright_guess[i]
        tol = tolerances[i]
        
        lower_bounds.extend([q_range[0], max(x_fit.min(), lam0 - tol), gamma_range[0]])
        upper_bounds.extend([q_range[1], min(x_fit.max(), lam0 + tol), gamma_range[1]])
    
    # Fit
    try:
        popt, _ = curve_fit(
            fano_bright, 
            x_fit, 
            y_fit,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000,
            method='trf'
        )
        
        # Calculate R²
        y_fit_curve = fano_bright(x_fit, *popt)
        ss_res = np.sum((y_fit - y_fit_curve)**2)
        ss_tot = np.sum((y_fit - y_fit.mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Package parameters
        params = {}
        for i in range(num_bright):
            params[f'bright{i+1}_c'] = popt[3*i]
            params[f'bright{i+1}_lambda'] = popt[3*i + 1]
            params[f'bright{i+1}_gamma'] = popt[3*i + 2]
        
        return y_fit_curve, params, r2
        
    except Exception as e:
        print(f"[error] Bright-only fitting failed: {e}")
        params = {}
        for i in range(num_bright):
            params[f'bright{i+1}_c'] = 0
            params[f'bright{i+1}_lambda'] = bright_guess[i]
            params[f'bright{i+1}_gamma'] = 0
        return np.zeros_like(y_fit), params, 0.0


def fit_fano_with_dark(args: Dict[str, Any],
                       y_fit: np.ndarray,
                       x_fit: np.ndarray,
                       num_bright: int,
                       num_dark: int,
                       bright_guess: List[float],
                       dark_guess: List[float],
                       params_bright: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Step 2: Fit bright + dark modes together
    WITH iterative refinement
    
    Bright parameters are initialized from Step 1 results.
    Dark parameters are initialized with manual guess.
    
    Model: I = |Σ bright + Σ dark|²
    
    Parameters per bright mode: [c_i, λ_i, γ_i]
    Parameters per dark mode: [d_j, λ_j, Γ_j, θ_j]
    Total: 3×num_bright + 4×num_dark
    """
    
    # ========================================
    # 1. Define full Fano model
    # ========================================
    def fano_full(x_val, *params):
        """Bright + Dark modes"""
        A_total = np.zeros_like(x_val, dtype=complex)
        
        # Bright modes (phase = 0)
        for i in range(num_bright):
            c = params[3*i]
            lam = params[3*i + 1]
            gamma = params[3*i + 2]
            
            A_i = c * (gamma/2) / (x_val - lam + 1j*gamma/2)
            A_total += A_i
        
        # Dark modes (phase fitted)
        offset = 3 * num_bright
        for j in range(num_dark):
            d = params[offset + 4*j]
            lam = params[offset + 4*j + 1]
            Gamma = params[offset + 4*j + 2]
            theta = params[offset + 4*j + 3]
            
            A_j = d * np.exp(1j * theta) * (Gamma/2) / (x_val - lam + 1j*Gamma/2)
            A_total += A_j
        
        I = np.abs(A_total)**2
        return I
    
    # ========================================
    # 2. Initial guess from Step 1
    # ========================================
    p0_current = []
    for i in range(num_bright):
        p0_current.extend([
            params_bright[f'bright{i+1}_c'],
            params_bright[f'bright{i+1}_lambda'],
            params_bright[f'bright{i+1}_gamma']
        ])
    
    phi_init = args.get('FANO_PHI_INIT', np.pi)
    for j in range(num_dark):
        d0 = 1.0
        lam0 = dark_guess[j]
        Gamma0 = 30.0
        theta0 = phi_init
        p0_current.extend([d0, lam0, Gamma0, theta0])
    
    # ========================================
    # 3. Setup bounds
    # ========================================
    q_range = args.get('FANO_Q_RANGE', (-20, 20))
    gamma_range = args.get('FANO_GAMMA_RANGE', (5, 100))
    phi_range = args.get('FANO_PHI_RANGE', (0, 2*np.pi))
    
    lower_bounds = []
    upper_bounds = []
    
    # Bright tolerances
    bright_tol = args.get('BRIGHT_POSITION_TOLERANCE', None)
    if bright_tol is not None:
        if isinstance(bright_tol, (list, tuple)):
            bright_tolerances = bright_tol if len(bright_tol) == num_bright else [bright_tol[0]] * num_bright
        else:
            bright_tolerances = [bright_tol] * num_bright
    else:
        bright_tolerances = [np.inf] * num_bright
    
    for i in range(num_bright):
        lam0 = bright_guess[i]
        tol = bright_tolerances[i]
        lower_bounds.extend([q_range[0], max(x_fit.min(), lam0 - tol), gamma_range[0]])
        upper_bounds.extend([q_range[1], min(x_fit.max(), lam0 + tol), gamma_range[1]])
    
    # Dark tolerances
    dark_tol = args.get('DARK_POSITION_TOLERANCE', None)
    if dark_tol is not None:
        if isinstance(dark_tol, (list, tuple)):
            dark_tolerances = dark_tol if len(dark_tol) == num_dark else [dark_tol[0]] * num_dark
        else:
            dark_tolerances = [dark_tol] * num_dark
    else:
        dark_tolerances = [np.inf] * num_dark
    
    for j in range(num_dark):
        lam0 = dark_guess[j]
        tol = dark_tolerances[j]
        lower_bounds.extend([q_range[0], max(x_fit.min(), lam0 - tol), gamma_range[0], phi_range[0]])
        upper_bounds.extend([q_range[1], min(x_fit.max(), lam0 + tol), gamma_range[1], phi_range[1]])
    
    # ========================================
    # 4. Iterative fitting with multiple strategies
    # ========================================
    max_iterations = args.get('FIT_MAX_ITERATIONS', 1)
    
    global_best_r2 = -np.inf
    global_best_params = None
    
    for iteration in range(max_iterations):
        print(f"\n[Fano Iteration {iteration+1}/{max_iterations}]")
        
        strategies = ['current_best', 'perturb_phase', 'perturb_amplitude']
        
        for strategy in strategies:
            # Generate trial parameters based on strategy
            if strategy == 'current_best':
                p0_trial = p0_current.copy()
            
            elif strategy == 'perturb_phase':
                p0_trial = p0_current.copy()
                # Dark mode phase에 작은 변화 추가
                offset = 3 * num_bright
                for j in range(num_dark):
                    phase_idx = offset + 4*j + 3
                    # 현재 위상에서 ±0.5 radian 범위로 perturbation
                    perturbation = np.random.uniform(-0.5, 0.5)
                    new_phase = p0_trial[phase_idx] + perturbation
                    # Bounds 내로 clipping
                    new_phase = np.clip(new_phase, phi_range[0], phi_range[1])
                    p0_trial[phase_idx] = new_phase
            
            elif strategy == 'perturb_amplitude':
                p0_trial = p0_current.copy()
                # Amplitude에 변화
                offset = 3 * num_bright
                for j in range(num_dark):
                    amp_idx = offset + 4*j
                    # 현재 amplitude를 0.5~1.5배로 스케일
                    scale_factor = np.random.uniform(0.5, 1.5)
                    new_amp = p0_trial[amp_idx] * scale_factor
                    # Bounds 내로 clipping
                    new_amp = np.clip(new_amp, q_range[0], q_range[1])
                    p0_trial[amp_idx] = new_amp
            
            # Try fitting
            try:
                popt, _ = curve_fit(
                    fano_full,
                    x_fit,
                    y_fit,
                    p0=p0_trial,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=20000,
                    method='trf'
                )
                
                # Calculate R²
                y_fit_curve = fano_full(x_fit, *popt)
                ss_res = np.sum((y_fit - y_fit_curve)**2)
                ss_tot = np.sum((y_fit - y_fit.mean())**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                print(f"  Strategy '{strategy}': R²={r2:.4f}")
                
                # Update best if better
                if r2 > global_best_r2:
                    global_best_r2 = r2
                    global_best_params = popt.copy()
                    p0_current = popt.copy()  # 다음 iteration의 시작점
                    print(f"    → NEW BEST!")
                    
            except Exception as e:
                print(f"  Strategy '{strategy}': FAILED - {str(e)}")
    
    # ========================================
    # 5. Use best result or fallback
    # ========================================
    if global_best_params is not None:
        # Package parameters
        params = {}
        for i in range(num_bright):
            params[f'bright{i+1}_c'] = global_best_params[3*i]
            params[f'bright{i+1}_lambda'] = global_best_params[3*i + 1]
            params[f'bright{i+1}_gamma'] = global_best_params[3*i + 2]
        
        offset = 3 * num_bright
        for j in range(num_dark):
            params[f'dark{j+1}_d'] = global_best_params[offset + 4*j]
            params[f'dark{j+1}_lambda'] = global_best_params[offset + 4*j + 1]
            params[f'dark{j+1}_Gamma'] = global_best_params[offset + 4*j + 2]
            params[f'dark{j+1}_theta'] = global_best_params[offset + 4*j + 3]
        
        y_fit_curve = fano_full(x_fit, *global_best_params)
        return y_fit_curve, params, global_best_r2
    
    else:
        # Fallback: All iterations failed
        print(f"[error] Full Fano fitting failed in all iterations")
        params = {}
        for i in range(num_bright):
            params[f'bright{i+1}_c'] = 0
            params[f'bright{i+1}_lambda'] = bright_guess[i]
            params[f'bright{i+1}_gamma'] = 0
        for j in range(num_dark):
            params[f'dark{j+1}_d'] = 0
            params[f'dark{j+1}_lambda'] = dark_guess[j]
            params[f'dark{j+1}_Gamma'] = 0
            params[f'dark{j+1}_theta'] = 0
        return np.zeros_like(y_fit), params, 0.0


def fano_model_bright_only(x: np.ndarray, num_bright: int, params: Dict[str, float]) -> np.ndarray:
    """Generate fitted curve for bright-only model over full wavelength range"""
    A_total = np.zeros_like(x, dtype=complex)
    
    for i in range(num_bright):
        c = params[f'bright{i+1}_c']
        lam = params[f'bright{i+1}_lambda']
        gamma = params[f'bright{i+1}_gamma']
        
        A_i = c * (gamma/2) / (x - lam + 1j*gamma/2)
        A_total += A_i
    
    I = np.abs(A_total)**2
    return I


def fano_model_full(x: np.ndarray, num_bright: int, num_dark: int, params: Dict[str, float]) -> np.ndarray:
    """Generate fitted curve for full Fano model over full wavelength range"""
    A_total = np.zeros_like(x, dtype=complex)
    
    # Bright modes
    for i in range(num_bright):
        c = params[f'bright{i+1}_c']
        lam = params[f'bright{i+1}_lambda']
        gamma = params[f'bright{i+1}_gamma']
        
        A_i = c * (gamma/2) / (x - lam + 1j*gamma/2)
        A_total += A_i
    
    # Dark modes
    for j in range(num_dark):
        d = params[f'dark{j+1}_d']
        lam = params[f'dark{j+1}_lambda']
        Gamma = params[f'dark{j+1}_Gamma']
        theta = params[f'dark{j+1}_theta']
        
        A_j = d * np.exp(1j * theta) * (Gamma/2) / (x - lam + 1j*Gamma/2)
        A_total += A_j
    
    I = np.abs(A_total)**2
    return I