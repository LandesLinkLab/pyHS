import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def get_peak(spec, wavelengths):
    idx = int(np.argmax(spec))
    return float(wavelengths[idx]), float(spec[idx])

def extract_spectrum_with_background(cube, row, col, args):
    """
    Extract spectrum from a particle location with background subtraction
    Similar to MATLAB version
    """
    int_size = args.get('INTEGRATION_SIZE', 3)
    bg_offset = args.get('BACKGROUND_OFFSET', 7)
    half_size = (int_size - 1) // 2
    
    H, W, L = cube.shape
    
    # Extract particle spectrum (integrate over int_size x int_size pixels)
    row_start = max(0, row - half_size)
    row_end = min(H, row + half_size + 1)
    col_start = max(0, col - half_size)
    col_end = min(W, col + half_size + 1)
    
    particle_spec = cube[row_start:row_end, col_start:col_end, :].mean(axis=(0, 1))
    
    # Extract background spectrum (offset pixels away)
    bg_row = min(max(0, row + bg_offset), H - 1)
    bg_row_start = max(0, bg_row - half_size)
    bg_row_end = min(H, bg_row + half_size + 1)
    
    background_spec = cube[bg_row_start:bg_row_end, col_start:col_end, :].mean(axis=(0, 1))
    
    # Subtract background
    corrected_spec = particle_spec - background_spec
    corrected_spec = np.maximum(corrected_spec, 0)  # No negative values
    
    return corrected_spec

def pick_manual_representatives(cube, wavelengths, args):
    """Process manually specified coordinates"""
    reps = []
    
    for row, col in args['MANUAL_COORDS']:
        # Extract spectrum with background subtraction
        spec = extract_spectrum_with_background(cube, row, col, args)
        
        # Get peak information
        wl_peak, intensity = get_peak(spec, wavelengths)
        
        reps.append(dict(
            row=int(row), 
            col=int(col),
            wl_peak=float(wl_peak),
            intensity=float(intensity)
        ))
    
    return reps

# def fit_lorentz(y, x, args):
#     """
#     Fit Lorentzian function to spectrum
#     Using the same form as MATLAB: (2*a/pi) * (c / (4*(x-b)^2 + c^2))
#     """
#     def lorentz_matlab_form(x, a, b, c):
#         # MATLAB form: (2*a/pi) * (c / (4*(x-b)^2 + c^2))
#         return (2*a/np.pi) * (c / (4*(x-b)**2 + c**2))
    
#     # Check if input data is valid
#     if len(y) == 0 or np.all(y == 0) or np.isnan(y).any():
#         print("[warning] Invalid spectrum data for fitting")
#         return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0, 'x0': 0, 'gamma': 0}, 0.0
    
#     # Initial guess
#     idx = int(np.argmax(y))
#     if y[idx] <= 0:
#         print("[warning] No positive values in spectrum")
#         return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0, 'x0': 0, 'gamma': 0}, 0.0
    
#     # Better initial guesses
#     a0 = float(y[idx] * np.pi / 2)  # Adjust for the 2/pi factor
#     b0 = float(x[idx])  # Peak position
    
#     # Estimate FWHM from data
#     half_max = y[idx] / 2
#     indices_above_half = np.where(y > half_max)[0]
#     if len(indices_above_half) > 1:
#         c0 = float(x[indices_above_half[-1]] - x[indices_above_half[0]])
#     else:
#         c0 = 70.0  # Default FWHM
    
#     p0 = [a0, b0, c0]
    
#     try:
#         # Set bounds similar to MATLAB
#         bounds = ([0, x.min(), 0], [np.inf, x.max(), np.inf])
        
#         popt, pcov = curve_fit(lorentz_matlab_form, x, y, p0=p0, 
#                               bounds=bounds, maxfev=8000, 
#                               method='trf')  # More robust method
        
#         y_fit = lorentz_matlab_form(x, *popt)
        
#         # Calculate R-squared
#         ss_res = np.sum((y - y_fit)**2)
#         ss_tot = np.sum((y - y.mean())**2)
#         rsq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
#         # Return with MATLAB-like parameter names
#         params = {
#             'a': popt[0], 
#             'b1': popt[1],  # λ_max
#             'c1': popt[2],  # FWHM (Γ)
#             'x0': popt[1],  # Alternative name
#             'gamma': popt[2]  # Alternative name
#         }
        
#         print(f"[debug] Fit successful: λ_max={popt[1]:.1f}, FWHM={popt[2]:.1f}, R²={rsq:.3f}")
        
#         return y_fit, params, float(rsq)
    
#     except Exception as e:
#         print(f"[warning] Fitting failed: {str(e)}")
#         return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0, 'x0': 0, 'gamma': 0}, 0.0

def fit_lorentz(y, x, args):
    """
    Fit Lorentzian function to spectrum
    Using the same form as MATLAB: (2*a/pi) * (c / (4*(x-b)^2 + c^2))
    """
    # Fitting 범위 적용
    if 'FIT_RANGE_NM' in args:
        fit_min, fit_max = args['FIT_RANGE_NM']
        mask = (x >= fit_min) & (x <= fit_max)
        x_fit = x[mask]
        y_fit = y[mask]
        
        if len(x_fit) < 10:  # 최소 10개 포인트 필요
            print(f"[warning] Too few points in fit range {fit_min}-{fit_max}nm")
            x_fit = x
            y_fit = y
    else:
        x_fit = x
        y_fit = y
    
    def lorentz_matlab_form(x, a, b, c):
        return (2*a/np.pi) * (c / (4*(x-b)**2 + c**2))
    
    # Check if input data is valid
    if len(y_fit) == 0 or np.all(y_fit == 0) or np.isnan(y_fit).any():
        print("[warning] Invalid spectrum data for fitting")
        return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0, 'x0': 0, 'gamma': 0}, 0.0
    
    # Initial guess
    idx = int(np.argmax(y_fit))
    if y_fit[idx] <= 0:
        print("[warning] No positive values in spectrum")
        return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0, 'x0': 0, 'gamma': 0}, 0.0
    
    # Better initial guesses
    a0 = float(y_fit[idx] * np.pi / 2)
    b0 = float(x_fit[idx])
    
    # Estimate FWHM
    half_max = y_fit[idx] / 2
    indices_above_half = np.where(y_fit > half_max)[0]
    if len(indices_above_half) > 1:
        c0 = float(x_fit[indices_above_half[-1]] - x_fit[indices_above_half[0]])
    else:
        c0 = 70.0
    
    p0 = [a0, b0, c0]
    
    try:
        # Set bounds
        bounds = ([0, x_fit.min(), 0], [np.inf, x_fit.max(), np.inf])
        
        # Fit on selected range
        popt, pcov = curve_fit(lorentz_matlab_form, x_fit, y_fit, p0=p0, 
                              bounds=bounds, maxfev=8000, 
                              method='trf')
        
        # Generate fit for full range
        y_fit_full = lorentz_matlab_form(x, *popt)
        
        # Calculate R-squared on fit range
        y_fit_range = lorentz_matlab_form(x_fit, *popt)
        ss_res = np.sum((y_fit - y_fit_range)**2)
        ss_tot = np.sum((y_fit - y_fit.mean())**2)
        rsq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Parameters
        params = {
            'a': popt[0], 
            'b1': popt[1],  # λ_max
            'c1': popt[2],  # FWHM (Γ)
            'x0': popt[1],
            'gamma': popt[2]
        }
        
        print(f"[debug] Fit successful on range {x_fit.min():.1f}-{x_fit.max():.1f}nm: "
              f"λ_max={popt[1]:.1f}, FWHM={popt[2]:.1f}, R²={rsq:.3f}")
        
        return y_fit_full, params, float(rsq)
    
    except Exception as e:
        print(f"[warning] Fitting failed: {str(e)}")
        return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0, 'x0': 0, 'gamma': 0}, 0.0

def plot_spectrum(x, y, y_fit, title, out_png, dpi=300, params=None, snr=None, args = None):
    """
    Plot spectrum exactly like MATLAB version
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if args and 'FIT_RANGE_NM' in args:

        fit_min, fit_max = args['FIT_RANGE_NM']
        ax.axvspan(fit_min, fit_max, alpha = 0.1, color = 'gray', label = 'Fit range')
    
    # MATLAB과 동일한 스타일
    ax.plot(x, y, 'b-', linewidth=3, label='Data')
    ax.plot(x, y_fit, 'k--', linewidth=3, label='Lorentz fit')
    
    # 축 라벨
    ax.set_xlabel('Wavelength (nm)', fontsize=32)
    ax.set_ylabel('Scattering', fontsize=32)
    
    # 축 폰트 크기
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # 박스 표시
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # 파라미터 텍스트
    if params is not None and snr is not None:
        lambda_max = params.get('b1', params.get('x0', 0))
        gamma = params.get('c1', params.get('gamma', 0))
        
        ax.text(0.55, 0.9, f'λ_max = {lambda_max:.0f} nm',
                transform=ax.transAxes, fontsize=20)
        ax.text(0.55, 0.78, f'Γ = {gamma:.0f} nm',
                transform=ax.transAxes, fontsize=20)
        ax.text(0.55, 0.66, f'S/N = {snr:.0f}',
                transform=ax.transAxes, fontsize=20)
    
    # 축 범위 설정
    ax.set_xlim(500, 825)
    
    # Y축 범위 - 에러 방지
    y_max = max(y.max(), y_fit.max()) if len(y) > 0 else 1.0
    if y_max <= 0:
        y_max = 1.0
    ax.set_ylim(0, y_max * 1.05)
    
    # 제목
    ax.set_title(title, fontsize=16)
    
    # 그리드 제거
    ax.grid(False)
    
    # Figure 저장
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def save_dfs_particle_map(max_map, representatives, output_path, sample_name):

    fig, ax = plt.subplots(figsize=(10, 10))
    
    vmin, vmax = np.percentile(max_map[max_map > 0], [5, 95]) if np.any(max_map > 0) else (0, 1)
    
    im = ax.imshow(max_map,
                   cmap='hot',
                   origin='lower',
                   vmin=vmin,
                   vmax=vmax,
                   interpolation='nearest')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Max Intensity (500-800 nm)', fontsize=12)
    
    # Particle marker setup (circle + number + center wavelength)
    for i, rep in enumerate(representatives):
        row, col = rep['row'], rep['col']
        
        # White marker
        circle_inner = plt.Circle((col, row), 
                                 radius=1,
                                 edgecolor='white',
                                 facecolor='none',
                                 linewidth=1)
        ax.add_patch(circle_inner)
        
        # Particle number
        ax.text(col - 1.5, row + 2,
                f'{i}',
                color='white',
                fontsize=6,
                fontweight='bold')
        
        # Wavelength
        ax.text(col - 4, row - 3,
                f'{rep["peak_wl"]:.0f}nm',
                color='yellow',
                fontsize=6,
                fontweight='bold',
                ha='left')
    
    # Title and axis label
    ax.set_title(f'{sample_name} - DFS Particle Map ({len(representatives)} particles)', 
                fontsize=16, pad=10)
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', color='white')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"[info] Saved DFS particle map: {output_path}")