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

def pick_representatives(cube, labels, wavelengths, args):
    """Handle both automatic and manual particle selection"""
    
    # Check if using manual coordinates
    if args.get('USE_MANUAL_COORDS', False) and args.get('MANUAL_COORDS'):
        return pick_manual_representatives(cube, wavelengths, args)
    
    # Original automatic selection
    reps = []
    for lab in np.unique(labels):
        if lab == 0: continue
        coords = np.argwhere(labels == lab)
        if coords.shape[0] < args["MIN_PIXELS_CLUS"]: continue
        
        peak_pos = np.array([get_peak(cube[r, c], wavelengths)[0] for r, c in coords])
        if peak_pos.std() > args["PEAK_TOL_NM"]: continue
        
        ints = np.array([get_peak(cube[r, c], wavelengths)[1] for r, c in coords])
        sel = ints.argmax() if args["REP_CRITERION"] == "max_int" else 0
        r_sel, c_sel = map(int, coords[sel])
        
        reps.append(dict(row=r_sel, col=c_sel, 
                        wl_peak=float(peak_pos[sel]), 
                        intensity=float(ints[sel])))
    return reps

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

def fit_lorentz(y, x, args):
    """
    Fit Lorentzian function to spectrum
    Using the same form as MATLAB: (2*a/pi) * (c / (4*(x-b)^2 + c^2))
    """
    def lorentz_matlab_form(x, a, b, c):
        # MATLAB form: (2*a/pi) * (c / (4*(x-b)^2 + c^2))
        return (2*a/np.pi) * (c / (4*(x-b)**2 + c**2))
    
    # Check if input data is valid
    if len(y) == 0 or np.all(y == 0) or np.isnan(y).any():
        print("[warning] Invalid spectrum data for fitting")
        return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0, 'x0': 0, 'gamma': 0}, 0.0
    
    # Initial guess
    idx = int(np.argmax(y))
    if y[idx] <= 0:
        print("[warning] No positive values in spectrum")
        return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0, 'x0': 0, 'gamma': 0}, 0.0
    
    # Better initial guesses
    a0 = float(y[idx] * np.pi / 2)  # Adjust for the 2/pi factor
    b0 = float(x[idx])  # Peak position
    
    # Estimate FWHM from data
    half_max = y[idx] / 2
    indices_above_half = np.where(y > half_max)[0]
    if len(indices_above_half) > 1:
        c0 = float(x[indices_above_half[-1]] - x[indices_above_half[0]])
    else:
        c0 = 70.0  # Default FWHM
    
    p0 = [a0, b0, c0]
    
    try:
        # Set bounds similar to MATLAB
        bounds = ([0, x.min(), 0], [np.inf, x.max(), np.inf])
        
        popt, pcov = curve_fit(lorentz_matlab_form, x, y, p0=p0, 
                              bounds=bounds, maxfev=8000, 
                              method='trf')  # More robust method
        
        y_fit = lorentz_matlab_form(x, *popt)
        
        # Calculate R-squared
        ss_res = np.sum((y - y_fit)**2)
        ss_tot = np.sum((y - y.mean())**2)
        rsq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Return with MATLAB-like parameter names
        params = {
            'a': popt[0], 
            'b1': popt[1],  # λ_max
            'c1': popt[2],  # FWHM (Γ)
            'x0': popt[1],  # Alternative name
            'gamma': popt[2]  # Alternative name
        }
        
        print(f"[debug] Fit successful: λ_max={popt[1]:.1f}, FWHM={popt[2]:.1f}, R²={rsq:.3f}")
        
        return y_fit, params, float(rsq)
    
    except Exception as e:
        print(f"[warning] Fitting failed: {str(e)}")
        return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0, 'x0': 0, 'gamma': 0}, 0.0

def plot_spectrum(x, y, y_fit, title, out_png, dpi=300, params=None, snr=None):
    """
    Plot spectrum exactly like MATLAB version
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
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

def save_markers(cube, reps, out_png, dpi=300):
    """
    Save particle map exactly like MATLAB version
    """
    # MATLAB과 동일: 전체 파장 범위 합산
    specfin_final = cube.sum(axis=2)
    
    # 에러 방지: 모든 값이 0인 경우 처리
    if np.all(specfin_final == 0):
        print("[warning] All values in image are zero")
        specfin_final = np.ones_like(specfin_final) * 0.1
        high = 1.0
    else:
        high = np.max(specfin_final)
    
    low = 0
    
    # Dynamic range가 너무 작은 경우 처리
    if high - low < 1e-10:
        high = low + 1.0
    
    # Figure 생성
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # MATLAB imshow와 동일한 표시
    im = ax.imshow(specfin_final,
                   cmap='gray',
                   origin='upper',  # MATLAB default
                   vmin=low,
                   vmax=high,
                   interpolation='nearest',
                   aspect='equal')
    
    # 축 제거
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # MATLAB 스타일 텍스트 마커
    for i, r in enumerate(reps):
        ax.text(r['col'], r['row'],
                str(i),
                color='green',
                fontsize=12,
                ha='center',
                va='center',
                weight='normal')
    
    # Figure 저장
    plt.tight_layout(pad=0.1)
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    print(f"[info] Image saved: shape={specfin_final.shape}, range=[{low}, {high:.2f}]")

def extract_spectrum_with_background(cube, row, col, args):
    """
    Extract spectrum from a particle location with background subtraction
    """
    int_size = args.get('INTEGRATION_SIZE', 3)
    bg_offset = args.get('BACKGROUND_OFFSET', 7)
    half_size = (int_size - 1) // 2
    
    H, W, L = cube.shape
    
    # Boundary check
    if row < 0 or row >= H or col < 0 or col >= W:
        print(f"[warning] Invalid coordinates: row={row}, col={col}")
        return np.zeros(L)
    
    # Extract particle spectrum
    row_start = max(0, row - half_size)
    row_end = min(H, row + half_size + 1)
    col_start = max(0, col - half_size)
    col_end = min(W, col + half_size + 1)
    
    # Check if region is valid
    if row_start >= row_end or col_start >= col_end:
        print(f"[warning] Invalid region for particle at ({row}, {col})")
        return np.zeros(L)
    
    particle_region = cube[row_start:row_end, col_start:col_end, :]
    if particle_region.size == 0:
        return np.zeros(L)
    
    particle_spec = particle_region.mean(axis=(0, 1))
    
    # Extract background spectrum
    bg_row = min(max(0, row + bg_offset), H - 1)
    bg_row_start = max(0, bg_row - half_size)
    bg_row_end = min(H, bg_row + half_size + 1)
    
    if bg_row_start >= bg_row_end:
        print(f"[warning] Invalid background region, using particle spectrum without background subtraction")
        return np.maximum(particle_spec, 0)
    
    background_region = cube[bg_row_start:bg_row_end, col_start:col_end, :]
    if background_region.size == 0:
        background_spec = np.zeros(L)
    else:
        background_spec = background_region.mean(axis=(0, 1))
    
    # Subtract background
    corrected_spec = particle_spec - background_spec
    corrected_spec = np.maximum(corrected_spec, 0)  # No negative values
    
    # Debug info
    print(f"[debug] Spectrum at ({row},{col}): max={corrected_spec.max():.2f}, "
          f"mean={corrected_spec.mean():.2f}, integrated={corrected_spec.sum():.2f}")
    
    return corrected_spec