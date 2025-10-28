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
        - PEAK_POSITION_INITIAL_GUESS: 'auto' or list of wavelengths in nm
        - PEAK_WIDTH_INITIAL_GUESS: None (auto) or list of widths in nm
        - PEAK_WIDTH_MAX: None, single value, or list of max widths in nm
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
    
    # Backward compatibility: support both old and new naming
    peak_guess = args.get('PEAK_POSITION_INITIAL_GUESS', args.get('PEAK_INITIAL_GUESS', 'auto'))
    
    # Validate consistency
    if peak_guess != 'auto' and not isinstance(peak_guess, (list, tuple)):
        raise ValueError(f"PEAK_POSITION_INITIAL_GUESS must be 'auto' or a list, got {type(peak_guess)}")
    
    if isinstance(peak_guess, (list, tuple)):
        if len(peak_guess) != num_peaks:
            raise ValueError(
                f"PEAK_POSITION_INITIAL_GUESS length ({len(peak_guess)}) "
                f"must match NUM_PEAKS ({num_peaks})"
            )
        manual_positions = peak_guess
    else:
        manual_positions = None
    
    return fit_n_lorentz(args, y, x, num_peaks, manual_positions)


def generate_initial_guess(args: Dict[str, Any],
                          y_fit: np.ndarray, 
                          x_fit: np.ndarray, 
                          num_peaks: int, 
                          manual_positions: Optional[List[float]] = None) -> List[float]:
    """
    Generate initial parameter guesses for N-peak Lorentzian fitting
    
    Parameters:
    -----------
    args : Dict[str, Any]
        Configuration dictionary
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
    # Get width initial guess
    width_guess = args.get('PEAK_WIDTH_INITIAL_GUESS', None)
    
    # Validate width_guess
    if width_guess is not None:
        if not isinstance(width_guess, (list, tuple)):
            raise ValueError(f"PEAK_WIDTH_INITIAL_GUESS must be a list, got {type(width_guess)}")
        if len(width_guess) != num_peaks:
            raise ValueError(
                f"PEAK_WIDTH_INITIAL_GUESS length ({len(width_guess)}) "
                f"must match NUM_PEAKS ({num_peaks})"
            )
        print(f"[debug] Using manual width initial guess: {width_guess} nm")
    
    if manual_positions is not None:
        # Mode B: Manual specification
        print(f"[debug] Using manual peak positions: {manual_positions}")
        
        p0 = []
        for i, b in enumerate(manual_positions):
            a0 = y_fit.max() * 0.5  # Amplitude estimate
            
            # Width initial value
            if width_guess is not None:
                c0 = width_guess[i]
            else:
                c0 = 30.0  # Default FWHM
            
            p0.extend([a0, b, c0])
        
        return p0
    
    else:
        # Mode A: Auto-detect peaks
        print(f"[debug] Auto-detecting {num_peaks} peak(s)...")
        
        # Find peaks using scipy
        peak_indices, properties = find_peaks(
            y_fit, 
            height=y_fit.max() * 0.1,
            distance=len(y_fit) // (num_peaks + 1)
        )
        
        if len(peak_indices) == 0:
            print("[warning] No peaks detected, using global maximum")
            peak_indices = [np.argmax(y_fit)]
        
        # Sort peaks by intensity (strongest first)
        peak_heights = y_fit[peak_indices]
        sorted_idx = np.argsort(peak_heights)[::-1]
        peak_indices = peak_indices[sorted_idx]
        
        # Use top N peaks
        peak_indices = peak_indices[:num_peaks]
        
        # Generate initial parameters
        p0 = []
        for i, idx in enumerate(peak_indices):
            a0 = y_fit[idx]  # Peak height
            b0 = x_fit[idx]  # Peak position
            
            # Width initial value
            if width_guess is not None:
                c0 = width_guess[i]
            else:
                # Estimate FWHM from data
                half_max = y_fit[idx] / 2
                left_idx = np.where(y_fit[:idx] < half_max)[0]
                right_idx = np.where(y_fit[idx:] < half_max)[0]
                
                if len(left_idx) > 0 and len(right_idx) > 0:
                    c0 = abs(x_fit[idx + right_idx[0]] - x_fit[left_idx[-1]])
                else:
                    c0 = 30.0
            
            p0.extend([a0, b0, c0])
            print(f"  Peak {i+1}: position={b0:.1f} nm, FWHM={c0:.1f} nm")
        
        # Pad with additional peaks if needed
        if len(p0) < 3 * num_peaks:
            print(f"[warning] Only {len(p0)//3} peaks detected, padding to {num_peaks}")
            for i in range(len(p0)//3, num_peaks):
                spacing = (x_fit.max() - x_fit.min()) / (num_peaks + 1)
                b0 = x_fit.min() + spacing * (i + 1)
                a0 = y_fit.max() * 0.3
                
                if width_guess is not None:
                    c0 = width_guess[i]
                else:
                    c0 = 30.0
                
                p0.extend([a0, b0, c0])
                print(f"  Peak {i+1} (padded): position={b0:.1f} nm, FWHM={c0:.1f} nm")
        
        return p0


def fit_n_lorentz(args: Dict[str, Any],
                  y: np.ndarray, 
                  x: np.ndarray, 
                  num_peaks: int,
                  manual_positions: Optional[List[float]] = None) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Fit N Lorentzian peaks with iterative refinement
    
    Parameters:
    -----------
    args : Dict[str, Any]
        Configuration dictionary with bounds and tolerances
    y : np.ndarray
        Full spectrum data
    x : np.ndarray
        Full wavelength array
    num_peaks : int
        Number of peaks to fit
    manual_positions : Optional[List[float]]
        Manual peak positions (None for auto-detect)
    
    Returns:
    --------
    Tuple[np.ndarray, Dict[str, float], float]
        - Fitted curve over full wavelength range
        - Parameter dict
        - R-squared
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
    p0 = generate_initial_guess(args, y_fit, x_fit, num_peaks, manual_positions)
    p0_original = p0.copy()
    
    # 6. Setup tolerance and bounds
    peak_tolerance = args.get('PEAK_POSITION_TOLERANCE', None)
    width_max = args.get('PEAK_WIDTH_MAX', None)
    
    lower_bounds = [0, x_fit.min(), 0] * num_peaks
    upper_bounds = [np.inf, x_fit.max(), np.inf] * num_peaks
    lower_bounds_original = lower_bounds.copy()
    upper_bounds_original = upper_bounds.copy()
    
    # Apply width maximum constraint (per-peak)
    if width_max is not None:
        if isinstance(width_max, (list, tuple)):
            # List: per-peak max
            if len(width_max) != num_peaks:
                print(f"[warning] PEAK_WIDTH_MAX length mismatch, using first value for all")
                width_max_list = [width_max[0]] * num_peaks
            else:
                width_max_list = width_max
            print(f"[debug] Applying per-peak width maximum: {width_max_list} nm")
        else:
            # Single value: all peaks
            width_max_list = [width_max] * num_peaks
            print(f"[debug] Applying width maximum constraint: {width_max} nm")
        
        for i in range(num_peaks):
            upper_bounds[3*i + 2] = width_max_list[i]
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
    
    # 7. Iterative fitting
    max_iterations = args.get('FIT_MAX_ITERATIONS', 100)
    
    p0_current = p0.copy()
    global_best_r2 = -np.inf
    global_best_params = None
    
    for iteration in range(max_iterations):
        iteration_best_r2 = -np.inf
        iteration_best_params = None
        
        if iteration == 0:
            strategies = ['current_best']
        else:
            strategies = ['current_best', 'shift_peaks_left', 'shift_peaks_right', 
                         'narrow_fwhm', 'widen_fwhm', 'random_explore']
        
        for strategy_idx, strategy in enumerate(strategies):
            
            # Generate trial parameters
            if strategy == 'current_best':
                p0_trial = p0_current.copy()
                if iteration > 0:
                    print(f"\nIteration {iteration+1}/{max_iterations}")
                    print(f"  Strategy {strategy_idx+1}/{len(strategies)}: {strategy}")
            
            elif strategy == 'shift_peaks_left':
                p0_trial = p0_current.copy()
                shift_amount = 10
                for i in range(num_peaks):
                    b_current = p0_current[3*i + 1]
                    b_new = b_current - shift_amount
                    
                    if tolerances is not None:
                        b_new = max(lower_bounds[3*i + 1], min(upper_bounds[3*i + 1], b_new))
                    else:
                        b_new = max(x_fit.min(), min(x_fit.max(), b_new))
                    
                    p0_trial[3*i + 1] = b_new
                print(f"  Strategy {strategy_idx+1}/{len(strategies)}: {strategy}")
            
            elif strategy == 'shift_peaks_right':
                p0_trial = p0_current.copy()
                shift_amount = 10
                for i in range(num_peaks):
                    b_current = p0_current[3*i + 1]
                    b_new = b_current + shift_amount
                    
                    if tolerances is not None:
                        b_new = max(lower_bounds[3*i + 1], min(upper_bounds[3*i + 1], b_new))
                    else:
                        b_new = max(x_fit.min(), min(x_fit.max(), b_new))
                    
                    p0_trial[3*i + 1] = b_new
                print(f"  Strategy {strategy_idx+1}/{len(strategies)}: {strategy}")
            
            elif strategy == 'narrow_fwhm':
                p0_trial = p0_current.copy()
                bounds_trial_lower = lower_bounds.copy()
                bounds_trial_upper = upper_bounds.copy()
                for i in range(num_peaks):
                    c_current = p0_current[3*i + 2]
                    c_new = c_current * 0.6
                    c_new = max(lower_bounds[3*i + 2], min(upper_bounds[3*i + 2], c_new))
                    p0_trial[3*i + 2] = c_new
                    bounds_trial_upper[3*i + 2] = min(c_current * 0.9, upper_bounds[3*i + 2])
                print(f"  Strategy {strategy_idx+1}/{len(strategies)}: {strategy}")

            elif strategy == 'widen_fwhm':
                p0_trial = p0_current.copy()
                bounds_trial_lower = lower_bounds.copy()
                bounds_trial_upper = upper_bounds.copy()
                for i in range(num_peaks):
                    c_current = p0_current[3*i + 2]
                    c_new = c_current * 1.5
                    c_new = max(lower_bounds[3*i + 2], min(upper_bounds[3*i + 2], c_new))
                    p0_trial[3*i + 2] = c_new
                    bounds_trial_lower[3*i + 2] = max(c_current * 1.1, lower_bounds[3*i + 2])
                print(f"  Strategy {strategy_idx+1}/{len(strategies)}: {strategy}")
            
            elif strategy == 'random_explore':
                p0_trial = p0_current.copy()
                np.random.seed(iteration * 100 + 42)
                
                for i in range(num_peaks):
                    # Random perturbation on amplitude
                    a_current = p0_current[3*i]
                    a_new = a_current * np.random.uniform(0.5, 1.5)
                    a_new = max(lower_bounds[3*i], min(upper_bounds[3*i], a_new))
                    p0_trial[3*i] = a_new
                    
                    # Random perturbation on position
                    b_current = p0_current[3*i + 1]
                    b_perturb = np.random.uniform(-5, 5)
                    b_new = b_current + b_perturb
                    b_new = max(lower_bounds[3*i + 1], min(upper_bounds[3*i + 1], b_new))
                    p0_trial[3*i + 1] = b_new
                    
                    # Random perturbation on FWHM
                    c_current = p0_current[3*i + 2]
                    c_new = c_current * np.random.uniform(0.7, 1.3)
                    c_new = max(lower_bounds[3*i + 2], min(upper_bounds[3*i + 2], c_new))
                    p0_trial[3*i + 2] = c_new
                
                print(f"  Strategy {strategy_idx+1}/{len(strategies)}: {strategy}")
            
            # Try fitting
            try:
                # Use strategy-specific bounds if available
                if strategy in ['narrow_fwhm', 'widen_fwhm']:
                    use_bounds = (bounds_trial_lower, bounds_trial_upper)
                else:
                    use_bounds = (lower_bounds, upper_bounds)
                
                popt, _ = curve_fit(
                    lorentz_n, 
                    x_fit, 
                    y_fit, 
                    p0=p0_trial,
                    bounds=use_bounds,
                    maxfev=10000,
                    method='trf'
                )
                
                # Calculate R²
                y_fit_curve = lorentz_n(x_fit, *popt)
                ss_res = np.sum((y_fit - y_fit_curve)**2)
                ss_tot = np.sum((y_fit - y_fit.mean())**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                print(f"    R²={r2:.4f}")
                
                # Update iteration best
                if r2 > iteration_best_r2:
                    iteration_best_r2 = r2
                    iteration_best_params = popt.copy()
                    print(f"    → ITERATION BEST!")
                
                # Update global best
                if r2 > global_best_r2:
                    global_best_r2 = r2
                    global_best_params = popt.copy()
                    print(f"    → GLOBAL BEST!")
                
            except Exception as e:
                print(f"    FAILED: {str(e)}")
        
        # Update starting point for next iteration
        if iteration_best_params is not None:
            p0_current = iteration_best_params.copy()
        
        # Early termination if excellent fit
        if global_best_r2 > 0.99:
            print(f"\n[Early termination] Excellent fit achieved (R² > 0.99)")
            break
    
    # 8. Return best result or fallback
    if global_best_params is not None:
        params = {}
        for i in range(num_peaks):
            peak_num = i + 1
            params[f'a{peak_num}'] = global_best_params[3*i]
            params[f'b{peak_num}'] = global_best_params[3*i + 1]
            params[f'c{peak_num}'] = global_best_params[3*i + 2]
        
        # MATLAB compatibility for single peak
        if num_peaks == 1:
            params['a'] = global_best_params[0]
            params['b1'] = global_best_params[1]
            params['c1'] = global_best_params[2]
        
        # Generate fitted curve over full wavelength range
        y_fit_full = lorentz_n(x, *global_best_params)
        return y_fit_full, params, global_best_r2
    
    else:
        # Fallback: All iterations failed
        print(f"[error] Lorentzian fitting failed in all iterations")
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


# =============================================================================
# FANO RESONANCE FITTING
# =============================================================================

def fit_fano(args: Dict[str, Any], y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Fit Fano resonance model to experimental spectrum data
    
    Two-step fitting procedure:
    1. Fit bright modes only (phase = 0 fixed)
    2. Add dark modes (phase fitted) while keeping bright modes from Step 1
    
    Parameters:
    -----------
    args : Dict[str, Any]
        Configuration dictionary containing:
        - NUM_BRIGHT_MODES: Number of bright modes (>= 0)
        - NUM_DARK_MODES: Number of dark modes (>= 0)
        - BRIGHT_POSITION_INITIAL_GUESS: List of wavelengths for bright modes
        - DARK_POSITION_INITIAL_GUESS: List of wavelengths for dark modes
        - BRIGHT_WIDTH_INITIAL_GUESS: List of widths for bright modes (optional)
        - DARK_WIDTH_INITIAL_GUESS: List of widths for dark modes (optional)
        - BRIGHT_WIDTH_MAX: Single value or list of max widths for bright modes
        - DARK_WIDTH_MAX: Single value or list of max widths for dark modes
        - BRIGHT_POSITION_TOLERANCE: Tolerance for bright peak positions
        - DARK_POSITION_TOLERANCE: Tolerance for dark peak positions
        - FANO_Q_RANGE: Amplitude range (default: (-20, 20))
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
    
    # Backward compatibility: support both old and new naming
    bright_guess = args.get('BRIGHT_POSITION_INITIAL_GUESS', args.get('BRIGHT_INITIAL_GUESS', None))
    dark_guess = args.get('DARK_POSITION_INITIAL_GUESS', args.get('DARK_INITIAL_GUESS', None))
    
    debug = args.get('FANO_DEBUG', False)
    
    if num_bright == 0 and num_dark == 0:
        raise ValueError("At least one bright or dark mode must be specified")
    
    if bright_guess is None and num_bright > 0:
        raise ValueError("BRIGHT_POSITION_INITIAL_GUESS is REQUIRED when NUM_BRIGHT_MODES > 0")
    
    if dark_guess is None and num_dark > 0:
        raise ValueError("DARK_POSITION_INITIAL_GUESS is REQUIRED when NUM_DARK_MODES > 0")
    
    if bright_guess is not None and len(bright_guess) != num_bright:
        raise ValueError(f"BRIGHT_POSITION_INITIAL_GUESS length ({len(bright_guess)}) must match NUM_BRIGHT_MODES ({num_bright})")
    
    if dark_guess is not None and len(dark_guess) != num_dark:
        raise ValueError(f"DARK_POSITION_INITIAL_GUESS length ({len(dark_guess)}) must match NUM_DARK_MODES ({num_dark})")
    
    # Validate width initial guess
    bright_width_guess = args.get('BRIGHT_WIDTH_INITIAL_GUESS', None)
    if bright_width_guess is not None:
        if not isinstance(bright_width_guess, (list, tuple)):
            raise ValueError(f"BRIGHT_WIDTH_INITIAL_GUESS must be a list, got {type(bright_width_guess)}")
        if len(bright_width_guess) != num_bright:
            raise ValueError(f"BRIGHT_WIDTH_INITIAL_GUESS length ({len(bright_width_guess)}) must match NUM_BRIGHT_MODES ({num_bright})")
    
    dark_width_guess = args.get('DARK_WIDTH_INITIAL_GUESS', None)
    if dark_width_guess is not None:
        if not isinstance(dark_width_guess, (list, tuple)):
            raise ValueError(f"DARK_WIDTH_INITIAL_GUESS must be a list, got {type(dark_width_guess)}")
        if len(dark_width_guess) != num_dark:
            raise ValueError(f"DARK_WIDTH_INITIAL_GUESS length ({len(dark_width_guess)}) must match NUM_DARK_MODES ({num_dark})")
    
    # Get fitting range
    fit_min, fit_max = args['FIT_RANGE_NM']
    mask = (x >= fit_min) & (x <= fit_max)
    x_fit = x[mask]
    y_fit = y[mask]

    bright_iter = args.get('FIT_BRIGHT_ITERATIONS')
    dark_iter = args.get('FIT_DARK_ITERATIONS')

    if debug:
        print(f"Bright iterations: {bright_iter}")
        print(f"Dark iterations: {dark_iter}")
    
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
        if bright_width_guess is not None:
            print(f"Bright width guess: {bright_width_guess}")
        if dark_width_guess is not None:
            print(f"Dark width guess: {dark_width_guess}")
    
    # Step 1: Fit bright modes only
    if num_bright > 0:
        if debug:
            print("\n[STEP 1] Fitting bright modes only...")
        
        y_fit_step1, params_step1, r2_step1 = fit_fano_bright_only(
            args, y_fit, x_fit, num_bright, bright_guess, bright_iter
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
            bright_guess, dark_guess, params_step1, dark_iter
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
                         bright_guess: List[float],
                         max_iterations: int) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Step 1: Fit bright modes only (phase = 0) with iterative strategy exploration
    
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
            
            A_i = c * (gamma/2) / (x_val - lam + 1j*gamma/2)
            A_total += A_i
        
        I = np.abs(A_total)**2
        return I
    
    # Initial guess
    bright_width_guess = args.get('BRIGHT_WIDTH_INITIAL_GUESS', None)
    
    p0 = []
    for i in range(num_bright):
        c0 = 1.0
        lam0 = bright_guess[i]
        
        if bright_width_guess is not None:
            gamma0 = bright_width_guess[i]
        else:
            gamma0 = 30.0
        
        p0.extend([c0, lam0, gamma0])
    
    # Setup bounds
    q_range = args.get('FANO_Q_RANGE', (-20, 20))
    gamma_range = args.get('FANO_GAMMA_RANGE', (5, 100))
    bright_width_max = args.get('BRIGHT_WIDTH_MAX', None)
    
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
    
    # Apply width max (per-mode)
    if bright_width_max is not None:
        if isinstance(bright_width_max, (list, tuple)):
            if len(bright_width_max) != num_bright:
                print(f"[warning] BRIGHT_WIDTH_MAX length mismatch, using first value for all")
                gamma_upper_list = [bright_width_max[0]] * num_bright
            else:
                gamma_upper_list = bright_width_max
        else:
            gamma_upper_list = [bright_width_max] * num_bright
    else:
        gamma_upper_list = [gamma_range[1]] * num_bright
    
    for i in range(num_bright):
        lam0 = bright_guess[i]
        tol = tolerances[i]
        gamma_upper = gamma_upper_list[i]
        
        lower_bounds.extend([q_range[0], max(x_fit.min(), lam0 - tol), gamma_range[0]])
        upper_bounds.extend([q_range[1], min(x_fit.max(), lam0 + tol), gamma_upper])
    
    # Iterative fitting
    p0_current = p0.copy()
    global_best_r2 = -np.inf
    global_best_params = None
    
    for iteration in range(max_iterations):
        iteration_best_r2 = -np.inf
        iteration_best_params = None
        
        if iteration == 0:
            strategies = ['current_best']
        else:
            strategies = ['current_best', 'shift_left', 'shift_right', 
                         'narrow_fwhm', 'widen_fwhm', 'random_explore']
        
        for strategy_idx, strategy in enumerate(strategies):
            
            if strategy == 'current_best':
                p0_trial = p0_current.copy()
                if iteration > 0:
                    print(f"\nIteration {iteration+1}/{max_iterations}")
                    print(f"  Strategy {strategy_idx+1}/6: {strategy}")
            
            elif strategy == 'shift_left':
                p0_trial = p0_current.copy()
                for i in range(num_bright):
                    lam_current = p0_current[3*i + 1]
                    lam_new = lam_current - 10
                    lam_new = max(lower_bounds[3*i + 1], min(upper_bounds[3*i + 1], lam_new))
                    p0_trial[3*i + 1] = lam_new
                print(f"  Strategy {strategy_idx+1}/6: {strategy}")
            
            elif strategy == 'shift_right':
                p0_trial = p0_current.copy()
                for i in range(num_bright):
                    lam_current = p0_current[3*i + 1]
                    lam_new = lam_current + 10
                    lam_new = max(lower_bounds[3*i + 1], min(upper_bounds[3*i + 1], lam_new))
                    p0_trial[3*i + 1] = lam_new
                print(f"  Strategy {strategy_idx+1}/6: {strategy}")
            
            elif strategy == 'narrow_fwhm':
                p0_trial = p0_current.copy()
                bounds_trial_lower = lower_bounds.copy()
                bounds_trial_upper = upper_bounds.copy()
                for i in range(num_bright):
                    gamma_current = p0_current[3*i + 2]
                    gamma_new = gamma_current * 0.6
                    # Clamp to valid range
                    gamma_new = max(lower_bounds[3*i + 2], min(upper_bounds[3*i + 2], gamma_new))
                    p0_trial[3*i + 2] = gamma_new
                    # Narrow the bounds dynamically for this strategy
                    # This prevents optimizer from going back to upper bound
                    bounds_trial_upper[3*i + 2] = min(gamma_current * 0.9, upper_bounds[3*i + 2])
                print(f"  Strategy {strategy_idx+1}/6: {strategy}")

            elif strategy == 'widen_fwhm':
                p0_trial = p0_current.copy()
                bounds_trial_lower = lower_bounds.copy()
                bounds_trial_upper = upper_bounds.copy()
                for i in range(num_bright):
                    gamma_current = p0_current[3*i + 2]
                    gamma_new = gamma_current * 1.5
                    # Clamp to valid range
                    gamma_new = max(lower_bounds[3*i + 2], min(upper_bounds[3*i + 2], gamma_new))
                    p0_trial[3*i + 2] = gamma_new
                    # Widen the bounds dynamically for this strategy
                    # This prevents optimizer from staying at lower bound
                    bounds_trial_lower[3*i + 2] = max(gamma_current * 1.1, lower_bounds[3*i + 2])
                print(f"  Strategy {strategy_idx+1}/6: {strategy}")
            
            elif strategy == 'random_explore':
                p0_trial = p0_current.copy()
                np.random.seed(iteration * 100 + 42)
                for i in range(num_bright):
                    c_current = p0_current[3*i]
                    c_new = c_current * np.random.uniform(0.5, 1.5)
                    c_new = max(lower_bounds[3*i], min(upper_bounds[3*i], c_new))
                    p0_trial[3*i] = c_new
                    
                    lam_current = p0_current[3*i + 1]
                    lam_perturb = np.random.uniform(-5, 5)
                    lam_new = lam_current + lam_perturb
                    lam_new = max(lower_bounds[3*i + 1], min(upper_bounds[3*i + 1], lam_new))
                    p0_trial[3*i + 1] = lam_new
                    
                    gamma_current = p0_current[3*i + 2]
                    gamma_new = gamma_current * np.random.uniform(0.7, 1.3)
                    gamma_new = max(lower_bounds[3*i + 2], min(upper_bounds[3*i + 2], gamma_new))
                    p0_trial[3*i + 2] = gamma_new
                print(f"  Strategy {strategy_idx+1}/6: {strategy}")
            
            # Try fitting
            try:
                # Use strategy-specific bounds if available
                if strategy in ['narrow_fwhm', 'widen_fwhm']:
                    use_bounds = (bounds_trial_lower, bounds_trial_upper)
                else:
                    use_bounds = (lower_bounds, upper_bounds)
                
                popt, _ = curve_fit(
                    fano_bright, 
                    x_fit, 
                    y_fit,
                    p0=p0_trial,
                    bounds=use_bounds,  # ← strategy에 따라 다른 bounds 사용
                    maxfev=10000,
                    method='trf'
                )
                
                # Calculate R²
                y_fit_curve = fano_bright(x_fit, *popt)
                ss_res = np.sum((y_fit - y_fit_curve)**2)
                ss_tot = np.sum((y_fit - y_fit.mean())**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                print(f"    R²={r2:.4f}")
                
                if r2 > iteration_best_r2:
                    iteration_best_r2 = r2
                    iteration_best_params = popt.copy()
                    print(f"    → ITERATION BEST!")
                
                if r2 > global_best_r2:
                    global_best_r2 = r2
                    global_best_params = popt.copy()
                    print(f"    → GLOBAL BEST!")
                
            except Exception as e:
                print(f"    FAILED: {str(e)}")
        
        if iteration_best_params is not None:
            p0_current = iteration_best_params.copy()
        
        if global_best_r2 > 0.99:
            print(f"\n[Early termination] Excellent fit achieved (R² > 0.99)")
            break
    
    # Return best result or fallback
    if global_best_params is not None:
        params = {}
        for i in range(num_bright):
            params[f'bright{i+1}_c'] = global_best_params[3*i]
            params[f'bright{i+1}_lambda'] = global_best_params[3*i + 1]
            params[f'bright{i+1}_gamma'] = global_best_params[3*i + 2]
        
        y_fit_curve = fano_bright(x_fit, *global_best_params)
        return y_fit_curve, params, global_best_r2
    
    else:
        print(f"[error] Bright-only fitting failed in all iterations")
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
                       params_bright: Dict[str, float],
                       max_iterations: int) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Step 2: Fit bright + dark modes together with iterative refinement
    """
    
    # Define full Fano model
    def fano_full(x_val, *params):
        """Full Fano model with bright + dark modes"""
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
    
    # Initial guess
    dark_width_guess = args.get('DARK_WIDTH_INITIAL_GUESS', None)
    
    p0 = []
    
    # Bright modes from Step 1
    for i in range(num_bright):
        c = params_bright[f'bright{i+1}_c']
        lam = params_bright[f'bright{i+1}_lambda']
        gamma = params_bright[f'bright{i+1}_gamma']
        p0.extend([c, lam, gamma])
    
    # Dark modes (new parameters)
    phi_init = args.get('FANO_PHI_INIT', np.pi)
    for j in range(num_dark):
        d0 = 1.0
        lam0 = dark_guess[j]
        
        if dark_width_guess is not None:
            Gamma0 = dark_width_guess[j]
        else:
            Gamma0 = 30.0
        
        theta0 = phi_init
        p0.extend([d0, lam0, Gamma0, theta0])
    
    # Setup bounds
    q_range = args.get('FANO_Q_RANGE', (-20, 20))
    gamma_range = args.get('FANO_GAMMA_RANGE', (5, 100))
    phi_range = args.get('FANO_PHI_RANGE', (0, 2*np.pi))
    bright_width_max = args.get('BRIGHT_WIDTH_MAX', None)
    dark_width_max = args.get('DARK_WIDTH_MAX', None)
    
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
    
    # Bright width max (per-mode)
    if bright_width_max is not None:
        if isinstance(bright_width_max, (list, tuple)):
            if len(bright_width_max) != num_bright:
                gamma_bright_upper_list = [bright_width_max[0]] * num_bright
            else:
                gamma_bright_upper_list = bright_width_max
        else:
            gamma_bright_upper_list = [bright_width_max] * num_bright
    else:
        gamma_bright_upper_list = [gamma_range[1]] * num_bright
    
    for i in range(num_bright):
        lam0 = bright_guess[i]
        tol = bright_tolerances[i]
        gamma_upper = gamma_bright_upper_list[i]
        
        lower_bounds.extend([q_range[0], max(x_fit.min(), lam0 - tol), gamma_range[0]])
        upper_bounds.extend([q_range[1], min(x_fit.max(), lam0 + tol), gamma_upper])
    
    # Dark tolerances
    dark_tol = args.get('DARK_POSITION_TOLERANCE', None)
    if dark_tol is not None:
        if isinstance(dark_tol, (list, tuple)):
            dark_tolerances = dark_tol if len(dark_tol) == num_dark else [dark_tol[0]] * num_dark
        else:
            dark_tolerances = [dark_tol] * num_dark
    else:
        dark_tolerances = [np.inf] * num_dark
    
    # Dark width max (per-mode)
    if dark_width_max is not None:
        if isinstance(dark_width_max, (list, tuple)):
            if len(dark_width_max) != num_dark:
                Gamma_dark_upper_list = [dark_width_max[0]] * num_dark
            else:
                Gamma_dark_upper_list = dark_width_max
        else:
            Gamma_dark_upper_list = [dark_width_max] * num_dark
    else:
        Gamma_dark_upper_list = [gamma_range[1]] * num_dark
    
    for j in range(num_dark):
        lam0 = dark_guess[j]
        tol = dark_tolerances[j]
        Gamma_upper = Gamma_dark_upper_list[j]
        
        lower_bounds.extend([q_range[0], max(x_fit.min(), lam0 - tol), gamma_range[0], phi_range[0]])
        upper_bounds.extend([q_range[1], min(x_fit.max(), lam0 + tol), Gamma_upper, phi_range[1]])
    
    # Iterative fitting
    p0_current = p0.copy()
    global_best_r2 = -np.inf
    global_best_params = None
    
    for iteration in range(max_iterations):
        iteration_best_r2 = -np.inf
        iteration_best_params = None
        
        if iteration == 0:
            strategies = ['current_best']
        else:
            strategies = ['current_best', 'shift_left', 'shift_right', 
                         'narrow_fwhm', 'widen_fwhm', 'random_explore']
        
        for strategy_idx, strategy in enumerate(strategies):
            
            if strategy == 'current_best':
                p0_trial = p0_current.copy()
                if iteration > 0:
                    print(f"\nIteration {iteration+1}/{max_iterations}")
                    print(f"  Strategy {strategy_idx+1}/6: {strategy}")
            
            elif strategy == 'shift_left':
                p0_trial = p0_current.copy()
                offset = 3 * num_bright
                for j in range(num_dark):
                    lam_current = p0_current[offset + 4*j + 1]
                    lam_new = lam_current - 10
                    lam_new = max(lower_bounds[offset + 4*j + 1], min(upper_bounds[offset + 4*j + 1], lam_new))
                    p0_trial[offset + 4*j + 1] = lam_new
                print(f"  Strategy {strategy_idx+1}/6: {strategy}")
            
            elif strategy == 'shift_right':
                p0_trial = p0_current.copy()
                offset = 3 * num_bright
                for j in range(num_dark):
                    lam_current = p0_current[offset + 4*j + 1]
                    lam_new = lam_current + 10
                    lam_new = max(lower_bounds[offset + 4*j + 1], min(upper_bounds[offset + 4*j + 1], lam_new))
                    p0_trial[offset + 4*j + 1] = lam_new
                print(f"  Strategy {strategy_idx+1}/6: {strategy}")
            
            elif strategy == 'narrow_fwhm':
                p0_trial = p0_current.copy()
                bounds_trial_lower = lower_bounds.copy()
                bounds_trial_upper = upper_bounds.copy()
                offset = 3 * num_bright
                for j in range(num_dark):
                    Gamma_current = p0_current[offset + 4*j + 2]
                    Gamma_new = Gamma_current * 0.6
                    Gamma_new = max(lower_bounds[offset + 4*j + 2], min(upper_bounds[offset + 4*j + 2], Gamma_new))
                    p0_trial[offset + 4*j + 2] = Gamma_new
                    # Narrow bounds for dark mode width
                    bounds_trial_upper[offset + 4*j + 2] = min(Gamma_current * 0.9, upper_bounds[offset + 4*j + 2])
                print(f"  Strategy {strategy_idx+1}/6: {strategy}")

            elif strategy == 'widen_fwhm':
                p0_trial = p0_current.copy()
                bounds_trial_lower = lower_bounds.copy()
                bounds_trial_upper = upper_bounds.copy()
                offset = 3 * num_bright
                for j in range(num_dark):
                    Gamma_current = p0_current[offset + 4*j + 2]
                    Gamma_new = Gamma_current * 1.5
                    Gamma_new = max(lower_bounds[offset + 4*j + 2], min(upper_bounds[offset + 4*j + 2], Gamma_new))
                    p0_trial[offset + 4*j + 2] = Gamma_new
                    # Widen bounds for dark mode width
                    bounds_trial_lower[offset + 4*j + 2] = max(Gamma_current * 1.1, lower_bounds[offset + 4*j + 2])
                print(f"  Strategy {strategy_idx+1}/6: {strategy}")
            
            elif strategy == 'random_explore':
                p0_trial = p0_current.copy()
                np.random.seed(iteration * 100 + 42)
                offset = 3 * num_bright
                for j in range(num_dark):
                    d_current = p0_current[offset + 4*j]
                    d_new = d_current * np.random.uniform(0.5, 1.5)
                    d_new = max(lower_bounds[offset + 4*j], min(upper_bounds[offset + 4*j], d_new))
                    p0_trial[offset + 4*j] = d_new
                    
                    lam_current = p0_current[offset + 4*j + 1]
                    lam_perturb = np.random.uniform(-5, 5)
                    lam_new = lam_current + lam_perturb
                    lam_new = max(lower_bounds[offset + 4*j + 1], min(upper_bounds[offset + 4*j + 1], lam_new))
                    p0_trial[offset + 4*j + 1] = lam_new
                    
                    Gamma_current = p0_current[offset + 4*j + 2]
                    Gamma_new = Gamma_current * np.random.uniform(0.7, 1.3)
                    Gamma_new = max(lower_bounds[offset + 4*j + 2], min(upper_bounds[offset + 4*j + 2], Gamma_new))
                    p0_trial[offset + 4*j + 2] = Gamma_new
                    
                    theta_current = p0_current[offset + 4*j + 3]
                    theta_new = theta_current + np.random.uniform(-0.5, 0.5)
                    theta_new = max(lower_bounds[offset + 4*j + 3], min(upper_bounds[offset + 4*j + 3], theta_new))
                    p0_trial[offset + 4*j + 3] = theta_new
                print(f"  Strategy {strategy_idx+1}/6: {strategy}")
            
            # Try fitting
            try:
                # Use strategy-specific bounds if available
                if strategy in ['narrow_fwhm', 'widen_fwhm']:
                    use_bounds = (bounds_trial_lower, bounds_trial_upper)
                else:
                    use_bounds = (lower_bounds, upper_bounds)
                
                popt, _ = curve_fit(
                    fano_full, 
                    x_fit, 
                    y_fit,
                    p0=p0_trial,
                    bounds=use_bounds,
                    maxfev=10000,
                    method='trf'
                )
                
                # Calculate R²
                y_fit_curve = fano_full(x_fit, *popt)
                ss_res = np.sum((y_fit - y_fit_curve)**2)
                ss_tot = np.sum((y_fit - y_fit.mean())**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                print(f"    R²={r2:.4f}")
                
                if r2 > iteration_best_r2:
                    iteration_best_r2 = r2
                    iteration_best_params = popt.copy()
                    print(f"    → ITERATION BEST!")
                
                if r2 > global_best_r2:
                    global_best_r2 = r2
                    global_best_params = popt.copy()
                    print(f"    → GLOBAL BEST!")
                
            except Exception as e:
                print(f"    FAILED: {str(e)}")
        
        if iteration_best_params is not None:
            p0_current = iteration_best_params.copy()
        
        if global_best_r2 > 0.99:
            print(f"\n[Early termination] Excellent fit achieved (R² > 0.99)")
            break
    
    # Return best result or fallback
    if global_best_params is not None:
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