import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple, Optional, Any, Union

def fit_lorentz(args: Dict[str, Any], y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Fit Lorentzian function to spectrum
    MATLAB과 동일하게 전체 범위에서 fitting 수행
    Returns: y_fit, params, rsq
    """

    fit_min, fit_max = args['FIT_RANGE_NM']
    mask = (x >= fit_min) & (x <= fit_max)
    x_fit = x[mask]
    y_fit = y[mask]
        
    def lorentz_matlab_form(x, a, b, c):
        return (2*a/np.pi) * (c / (4*(x-b)**2 + c**2))
    
    # Check if input data is valid
    if len(y_fit) == 0 or np.all(y_fit == 0) or np.isnan(y_fit).any():
        print("[warning] Invalid spectrum data for fitting")
        return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0}, 0.0
    
    # Initial guess - y_fit 사용!
    idx = int(np.argmax(y_fit))
    if y_fit[idx] <= 0:
        print("[warning] No positive values in spectrum")
        return np.zeros_like(y), {'a': 0, 'b1': 0, 'c1': 0}, 0.0
    
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
        # Set bounds - x_fit 사용!
        bounds = ([0, x_fit.min(), 0], [np.inf, x_fit.max(), np.inf])
        
        # Fit on selected range - x_fit, y_fit 사용!
        popt, pcov = curve_fit(lorentz_matlab_form, x_fit, y_fit, p0=p0, 
                              bounds=bounds, maxfev=8000, 
                              method='trf')
        
        # Generate fit for FULL range (원본 x 사용)
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
                args: Optional[Dict[str, Any]] = None) -> None:
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
        lambda_max = params.get('b1', 0)
        gamma = params.get('c1', 0)
        
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

def save_dfs_particle_map(max_map: np.ndarray, 
                        representatives: List[Dict[str, Any]], 
                        output_path: Path, 
                        sample_name: str) -> None:
    """Save particle map with markers"""
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
    cbar.set_label('Max Intensity', fontsize=12)
    
    # Particle markers
    for i, rep in enumerate(representatives):
        row, col = rep['row'], rep['col']
        
        # White circle marker
        circle_inner = plt.Circle((col, row), 
                                 radius=3,
                                 edgecolor='white',
                                 facecolor='none',
                                 linewidth=2)
        ax.add_patch(circle_inner)
        
        # Particle number
        ax.text(col - 1, row + 3,
                f'{i}',
                color='white',
                fontsize=6,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Wavelength and cluster info
        ax.text(col - 3, row - 3,
                # f'{rep["peak_wl"]:.0f}nm (C{rep["cluster_label"]})',
                f'{rep["peak_wl"]:.0f}nm',
                color='yellow',
                fontsize=6,
                fontweight='bold',
                ha='left')
    
    # Title and axis labels
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