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
    
    # Original automatic selection - 수정된 버전
    reps = []
    unique_labels = np.unique(labels)
    print(f"\n[debug] Processing {len(unique_labels)-1} detected regions (excluding background)")
    
    for lab in unique_labels:
        if lab == 0: 
            continue  # Skip background
            
        coords = np.argwhere(labels == lab)
        print(f"\n[debug] Label {lab}: {coords.shape[0]} pixels")
        
        if coords.shape[0] < args["MIN_PIXELS_CLUS"]: 
            print(f"  - Skipped: too few pixels ({coords.shape[0]} < {args['MIN_PIXELS_CLUS']})")
            continue
        
        # 각 픽셀에서 스펙트럼 추출하여 peak 위치 확인
        spectra = []
        peak_positions = []
        peak_intensities = []
        
        for r, c in coords:
            spec = cube[r, c]
            if spec.max() > 0:  # 유효한 스펙트럼인 경우만
                wl_peak, intensity = get_peak(spec, wavelengths)
                spectra.append(spec)
                peak_positions.append(wl_peak)
                peak_intensities.append(intensity)
        
        if len(peak_positions) == 0:
            print(f"  - Skipped: no valid spectra")
            continue
            
        peak_pos_array = np.array(peak_positions)
        peak_std = peak_pos_array.std()
        peak_mean = peak_pos_array.mean()
        
        print(f"  - Peak wavelength: {peak_mean:.1f} ± {peak_std:.1f} nm")
        print(f"  - Peak intensity range: [{min(peak_intensities):.1f}, {max(peak_intensities):.1f}]")
        
        # Peak 위치의 표준편차가 tolerance 이내인지 확인
        if peak_std > args["PEAK_TOL_NM"]: 
            print(f"  - Skipped: peak variation too large ({peak_std:.1f} > {args['PEAK_TOL_NM']})")
            continue
        
        # 대표 픽셀 선택 (가장 강한 intensity를 가진 픽셀)
        ints = np.array(peak_intensities)
        if args["REP_CRITERION"] == "max_int":
            sel = ints.argmax()
        else:
            sel = len(coords) // 2  # 중앙 픽셀
            
        r_sel, c_sel = map(int, coords[sel])
        
        print(f"  - Selected pixel ({r_sel}, {c_sel}) with intensity {ints[sel]:.1f}")
        
        reps.append(dict(
            row=r_sel, 
            col=c_sel, 
            wl_peak=float(peak_positions[sel]), 
            intensity=float(ints[sel]),
            label=int(lab),
            n_pixels=len(coords)
        ))
    
    print(f"\n[info] Selected {len(reps)} representative particles from {len(unique_labels)-1} regions")
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
    Save particle map with enhanced visualization
    Shows particles on a background where they are clearly visible
    """
    # 여러 배경 이미지 옵션 준비
    sum_img = cube.sum(axis=2)
    max_img = cube.max(axis=2)
    
    # 특정 파장에서의 이미지 (중간 파장)
    mid_idx = cube.shape[2] // 2
    mid_wl_img = cube[:, :, mid_idx]
    
    # 가장 contrast가 좋은 이미지 선택
    # (표준편차가 큰 이미지가 일반적으로 feature가 잘 보임)
    images = {'sum': sum_img, 'max': max_img, 'mid_wl': mid_wl_img}
    best_img = max(images.items(), key=lambda x: x[1].std())[1]
    
    # Dynamic range 설정
    vmin, vmax = np.percentile(best_img[best_img > 0], [2, 98]) if np.any(best_img > 0) else (0, 1)
    
    # Figure 생성 - MATLAB 스타일과 향상된 버전 둘 다
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # ---- 1. MATLAB 스타일 (왼쪽) ----
    ax1.imshow(sum_img,
               cmap='gray',
               origin='upper',
               vmin=0,
               vmax=sum_img.max() if sum_img.max() > 0 else 1,
               interpolation='nearest')
    
    ax1.set_title(f'MATLAB Style - {len(reps)} particles', fontsize=14)
    ax1.axis('off')
    
    # 녹색 번호 표시
    for i, r in enumerate(reps):
        ax1.text(r['col'], r['row'],
                str(i),
                color='green',
                fontsize=12,
                ha='center',
                va='center',
                weight='bold')
    
    # ---- 2. 향상된 시각화 (오른쪽) ----
    im = ax2.imshow(best_img,
                    cmap='hot',
                    origin='upper',
                    vmin=vmin,
                    vmax=vmax,
                    interpolation='nearest')
    
    ax2.set_title(f'Enhanced View - {len(reps)} particles', fontsize=14)
    ax2.axis('off')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Intensity', fontsize=10)
    
    # 향상된 마커
    for i, r in enumerate(reps):
        # 흰색 원 with 검은 테두리
        circle1 = plt.Circle((r['col'], r['row']), 
                           radius=2,
                           edgecolor='black',
                           facecolor='none',
                           linewidth=2)
        circle2 = plt.Circle((r['col'], r['row']), 
                           radius=2,
                           edgecolor='white',
                           facecolor='none',
                           linewidth=1)
        ax2.add_patch(circle1)
        ax2.add_patch(circle2)
        
        # 번호 표시 (배경 박스와 함께)
        ax2.text(r['col'] + 3, r['row'] - 3,
                str(i),
                color='white',
                fontsize=6,
                weight='bold',
                bbox=dict(facecolor='black', alpha=0.7, pad=2, edgecolor='white'))
    
    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # 추가 정보 출력
    print(f"\n[info] Markers saved: {out_png}")
    for i, r in enumerate(reps):
        info = f"  Particle {i}: pos=({r['row']}, {r['col']}), λ_peak={r['wl_peak']:.1f}nm"
        if 'n_pixels' in r:
            info += f", pixels={r['n_pixels']}"
        print(info)

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