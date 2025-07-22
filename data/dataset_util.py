import numpy as np
from pathlib import Path
from nptdms import TdmsFile
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.ndimage import maximum_filter
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, footprint_rectangle

from typing import List, Dict, Tuple, Optional, Any, Union

def tdms_to_cube(path: Path,
                 image_shape: Optional[Tuple[int, int]] = None):
    """
    TDMS → (H,W,L) cube 변환
    Info 그룹의 wvlths 채널에서 파장 정보 추출
    """
    td = TdmsFile.read(path)
    
    # ── 1) Info 그룹에서 파장 정보 추출 ──
    info_group = td['Info']
    if info_group is None:
        raise RuntimeError("No 'Info' group found in TDMS file")
    
    # wvlths 채널 찾기
    wl_channel = None
    for ch in info_group.channels():
        if ch.name == 'wvlths':
            wl_channel = ch
            break
    
    if wl_channel is None:
        raise RuntimeError("No 'wvlths' channel found in Info group")
    
    wl = wl_channel[:].astype(np.float32)
    print(f"[info] Wavelength array from TDMS: {wl.min():.1f}-{wl.max():.1f} nm, {len(wl)} points")
    
    # ── 2) Spectra 그룹에서 스펙트럼 데이터 추출 ──
    spectra_group = td['Spectra']
    if spectra_group is None:
        raise RuntimeError("No 'Spectra' group found in TDMS file")
    
    specs = list(spectra_group.channels())
    Nspec = len(specs)
    print(f"[info] Found {Nspec} spectrum channels")
    
    if Nspec == 0:
        raise RuntimeError("No spectrum channels found")
    
    # ── 3) 이미지 shape 결정 ──
    if image_shape is not None:
        rows, cols = image_shape
    else:
        # Root properties에서 읽기
        rows = int(td.properties.get('strips', 0))
        top = int(td.properties.get('top pixel', 0))
        bottom = int(td.properties.get('bottom pixel', 0))
        cols = bottom - top + 1
        
        if rows * cols != Nspec:
            # 재계산 시도
            if Nspec == 9261:  # Your specific case
                rows, cols = 49, 189
            elif Nspec == 8000:  # White/Dark reference
                rows, cols = 20, 400
            else:
                raise RuntimeError(f"Cannot determine image shape for {Nspec} channels")
    
    print(f"[info] Image shape: {rows} x {cols}")
    
    # ── 4) 데이터 큐브 생성 ──
    # 각 채널의 데이터를 numpy 배열로
    spectrum_length = len(specs[0][:])
    
    # 파장 배열과 스펙트럼 길이가 일치하는지 확인
    if len(wl) != spectrum_length:
        print(f"[warning] Wavelength array length ({len(wl)}) != spectrum length ({spectrum_length})")
        # 필요하면 interpolation 또는 truncation
        if len(wl) > spectrum_length:
            wl = wl[:spectrum_length]
        else:
            # 이 경우는 문제가 있음
            raise RuntimeError(f"Wavelength array too short: {len(wl)} < {spectrum_length}")
    
    # 데이터 스택
    data_list = []
    for i, ch in enumerate(specs):
        data = ch[:].astype(np.float32)
        data_list.append(data)
        if i % 1000 == 0:
            print(f"  Loading channel {i}/{Nspec}...")
    
    stack = np.vstack(data_list)  # (Nspec, L)
    
    # 3D cube로 재구성 - MATLAB column-major order
    # MATLAB: for c = 1:cols, for r = 1:rows
    # 이는 column이 먼저 변하므로 Fortran order
    cube = stack.reshape(cols, rows, spectrum_length, order='F').transpose(1, 0, 2)
    
    return cube.astype(np.float32), wl.astype(np.float32)

# ------------------- Remaining helpers ----------------------------
def flatfield_correct(cube: np.ndarray,
                      wvl: np.ndarray,
                      white_path: Path,
                      dark_path: Path) -> np.ndarray:
    """
    Flatfield correction - MATLAB과 동일한 방식
    White/Dark의 1340 포인트를 2개씩 평균내어 670 포인트로 변환
    """
    # Load white and dark references
    w_cube, w_wvl = tdms_to_cube(white_path)
    d_cube, d_wvl = tdms_to_cube(dark_path)
    
    print(f"[debug] Sample wavelengths: {wvl.min():.1f}-{wvl.max():.1f} nm ({len(wvl)} points)")
    print(f"[debug] White wavelengths: {w_wvl.min():.1f}-{w_wvl.max():.1f} nm ({len(w_wvl)} points)")
    
    H_w, W_w, L_w = w_cube.shape
    L_sample = len(wvl)
    
    # MATLAB 방식: 2개씩 평균 (1340 -> 670)
    if L_w == 2 * L_sample:
        print(f"[debug] Using MATLAB-style 2-point averaging (1340 -> 670)")
        
        # Reshape and average every 2 points
        w_cube_avg = w_cube.reshape(H_w, W_w, L_sample, 2).mean(axis=3)
        d_cube_avg = d_cube.reshape(H_w, W_w, L_sample, 2).mean(axis=3)
        
        # 파장도 2개씩 평균
        w_wvl_avg = w_wvl.reshape(L_sample, 2).mean(axis=1)
        
        # 평균된 파장이 샘플 파장과 일치하는지 확인
        wvl_diff = np.abs(w_wvl_avg - wvl).mean()
        print(f"[debug] Average wavelength difference after 2-point avg: {wvl_diff:.3f} nm")
        
        w_crop = w_cube_avg
        d_crop = d_cube_avg
        
    else:
        # Fallback: nearest neighbor (기존 방식)
        print(f"[warning] White/Dark not exactly 2x sample size. Using nearest neighbor matching.")
        idxs = []
        for wl_val in wvl:
            idx = np.argmin(np.abs(w_wvl - wl_val))
            idxs.append(idx)
        idxs = np.array(idxs)
        
        w_crop = w_cube[:, :, idxs]
        d_crop = d_cube[:, :, idxs]
    
    # Spatial average to get reference spectra
    w_ref = w_crop.mean(axis=(0, 1))  # (L,)
    d_ref = d_crop.mean(axis=(0, 1))  # (L,)
    
    print(f"[debug] White reference range: [{w_ref.min():.1f}, {w_ref.max():.1f}]")
    print(f"[debug] Dark reference range: [{d_ref.min():.1f}, {d_ref.max():.1f}]")
    
    # Apply correction - MATLAB formula
    numerator = cube - d_ref[None, None, :]
    denominator = (w_ref - d_ref)[None, None, :]
    
    # Avoid division by zero
    denominator = np.where(denominator > 0, denominator, 1.0)
    
    corrected = numerator / denominator
    
    # Clip extreme values
    corrected = np.clip(corrected, 0, 10)  # Reasonable range for corrected data
    
    print(f"[debug] After flatfield: range [{corrected.min():.3f}, {corrected.max():.3f}]")
    
    return corrected

def crop_and_bg(cube, wavelengths, args):

    m = (wavelengths >= args["CROP_RANGE_NM"][0]) & (wavelengths <= args["CROP_RANGE_NM"][1])
    cube = cube[:, :, m]
    wavelengths = wavelengths[m]

    bg_mode = args.get('BACKGROUND_MODE', 'global')

    if bg_mode == 'global':

        percentile = args.get("BACKGROUND_GLOBAL_PERCENTILE", 1) / 100.0\

        if percentile > 0:

            bg = np.quantile(cube, percentile, axis=2, keepdims=True)
            cube = np.maximum(cube - bg, 0)

    return cube.astype(np.float32), wavelengths

def apply_local_background(cube, clusters, representatives, args):
    """
    Local background correction
    각 클러스터 주변에서 가장 어두운 영역을 background로 사용
    """
    print(f"\n[debug] Applying local background correction")
    
    H, W, L = cube.shape
    corrected = cube.copy()
    
    # 각 대표 픽셀에 대해 local background 적용
    for i, rep in enumerate(representatives):
        row, col = rep['row'], rep['col']
        cluster = next(c for c in clusters if c['label'] == rep['label'])
        
        # Local background region 찾기
        search_radius = args.get('BACKGROUND_LOCAL_SEARCH_RADIUS', 20)
        percentile = args.get('BACKGROUND_LOCAL_PERCENTILE', 1)
        
        # 검색 영역 정의 (클러스터 주변)
        row_min = max(0, row - search_radius)
        row_max = min(H, row + search_radius)
        col_min = max(0, col - search_radius)
        col_max = min(W, col + search_radius)
        
        # 검색 영역에서 intensity sum이 낮은 픽셀들 찾기
        search_region = cube[row_min:row_max, col_min:col_max, :]
        intensity_map = search_region.sum(axis=2)
        
        # 클러스터 픽셀은 제외
        mask = np.ones(intensity_map.shape, dtype=bool)
        for coord in cluster['coords']:
            local_r = coord[0] - row_min
            local_c = coord[1] - col_min
            if 0 <= local_r < mask.shape[0] and 0 <= local_c < mask.shape[1]:
                mask[local_r, local_c] = False
        
        # 가장 어두운 픽셀들 선택
        masked_intensity = intensity_map[mask]
        if len(masked_intensity) > 0:
            threshold = np.percentile(masked_intensity, percentile)
            dark_pixels = np.where((intensity_map <= threshold) & mask)
            
            if len(dark_pixels[0]) > 0:
                # Local background spectrum 계산
                background_spectra = search_region[dark_pixels[0], dark_pixels[1], :]
                local_bg = background_spectra.mean(axis=0)
                
                print(f"  Cluster {i}: Using {len(dark_pixels[0])} pixels for local background")
                print(f"    Background range: [{local_bg.min():.1f}, {local_bg.max():.1f}]")
                
                # 클러스터의 모든 픽셀에 local background subtraction 적용
                for coord in cluster['coords']:
                    r, c = coord
                    corrected[r, c, :] = np.maximum(cube[r, c, :] - local_bg, 0)
            else:
                print(f"  Cluster {i}: No suitable background pixels nearby")
        else:
            print(f"  Cluster {i}: No valid pixels in search region")
    
    print(f"[debug] After local background correction: range [{corrected.min():.3f}, {corrected.max():.3f}]")
    
    return corrected

def create_dfs_max_intensity_map(cube, wavelengths, wl_range=(500, 800)):
    """
    Create max intensity projection map for DFS data
    특정 파장 범위에서 최대값 투영
    """
    # 파장 범위 마스크
    mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
    
    if not np.any(mask):
        print(f"[warning] No wavelengths in range {wl_range}, using full range")
        mask = np.ones_like(wavelengths, dtype=bool)
    
    # 해당 범위에서 max projection
    max_map = cube[:, :, mask].max(axis=2)
    
    print(f"[info] Max intensity map created from {mask.sum()} wavelengths "
          f"({wavelengths[mask].min():.1f}-{wavelengths[mask].max():.1f} nm)")
    print(f"[info] Map range: [{max_map.min():.2f}, {max_map.max():.2f}]")
    
    return max_map

def detect_dfs_particles(max_map, args):
    """
    Detect particle clusters from DFS max intensity map
    Enhanced version for low contrast data
    """
    # Check if map has valid data
    if max_map.max() <= max_map.min():
        print("[warning] Max map has no contrast")
        # Try to use raw values without normalization
        mask = max_map > np.percentile(max_map, 90)  # Top 10% of values
    else:
        # Normalize for thresholding
        normalized = (max_map - max_map.min()) / (max_map.max() - max_map.min())
        
        # Adaptive threshold based on data distribution
        threshold = args.get('DFS_INTENSITY_THRESHOLD', 0.1)
        
        # Try Otsu if manual threshold fails
        
        try:
            otsu_val = threshold_otsu(normalized)
            print(f"[debug] Otsu threshold on normalized data: {otsu_val:.3f}")
            # Use lower of manual and Otsu
            threshold = min(threshold, otsu_val * 0.8)  # 80% of Otsu to be more inclusive
        except:
            pass
        
        mask = normalized > threshold
        print(f"[debug] Using threshold: {threshold}")
    
    print(f"[debug] Pixels above threshold: {mask.sum()}")
    
    # If too few pixels, lower threshold
    if mask.sum() < 50:  # Less than 50 pixels
        print("[warning] Too few pixels detected, lowering threshold")
        percentile_threshold = 80  # Top 20% of pixels
        threshold_value = np.percentile(max_map, percentile_threshold)
        mask = max_map > threshold_value
        print(f"[debug] Using percentile {percentile_threshold}: {mask.sum()} pixels above threshold")
    
    # Morphological operations
    
    mask = binary_closing(mask, footprint_rectangle((3, 3)))
    
    # For initial detection, use smaller minimum size
    min_size = max(2, args.get("MIN_PIXELS_CLUS", 4) // 2)
    mask = remove_small_objects(mask, min_size)
    
    # Label connected components
    
    labels, num = ndi.label(mask)
    
    print(f"[info] Found {num} particle clusters (min size: {min_size} pixels)")
    
    # Get cluster info
    clusters = []
    for i in range(1, num + 1):
        coords = np.argwhere(labels == i)
        size = len(coords)
        
        # Apply original size filter here
        if size >= args.get("MIN_PIXELS_CLUS", 4):
            clusters.append({
                'label': i,
                'coords': coords,
                'size': size,
                'center': coords.mean(axis=0),
                'max_intensity': max_map[coords[:, 0], coords[:, 1]].max(),
                'mean_intensity': max_map[coords[:, 0], coords[:, 1]].mean()
            })
            print(f"  Cluster {i}: {size} pixels, center at ({coords.mean(axis=0)[0]:.1f}, {coords.mean(axis=0)[1]:.1f}), "
                  f"max_int={max_map[coords[:, 0], coords[:, 1]].max():.2f}")
        else:
            print(f"  Cluster {i}: {size} pixels (skipped - too small)")
    
    # If no clusters found, try to find peaks
    if len(clusters) == 0:
        print("[warning] No clusters found, trying peak detection")
        
        
        # Find local maxima
        local_max = maximum_filter(max_map, size=5)
        peaks = (max_map == local_max) & (max_map > np.percentile(max_map, 90))
        peak_coords = np.argwhere(peaks)
        
        print(f"[debug] Found {len(peak_coords)} peaks")
        
        # Create clusters from peaks
        for i, (row, col) in enumerate(peak_coords[:20]):  # Limit to 20 peaks
            clusters.append({
                'label': i + 1,
                'coords': np.array([[row, col]]),
                'size': 1,
                'center': np.array([row, col]),
                'max_intensity': max_map[row, col],
                'mean_intensity': max_map[row, col]
            })
    
    print(f"[info] Returning {len(clusters)} valid clusters")
    return labels, clusters

def select_representative_spectra(cube, wavelengths, clusters, args):
    """
    각 클러스터에서 대표 스펙트럼 선택
    - 가장 높은 intensity를 가진 픽셀 선택
    - FWHM tolerance 체크
    """
    representatives = []
    
    for cluster in clusters:
        coords = cluster['coords']
        label = cluster['label']
        
        print(f"\n[Processing Cluster {label}]")
        
        # 모든 픽셀의 스펙트럼과 특성 추출
        spectra_info = []
        for idx, (row, col) in enumerate(coords):
            spectrum = cube[row, col, :]
            
            # Peak 찾기
            peak_idx = np.argmax(spectrum)
            peak_wl = wavelengths[peak_idx]
            peak_intensity = spectrum[peak_idx]
            
            # 간단한 FWHM 추정
            half_max = peak_intensity / 2
            above_half = np.where(spectrum > half_max)[0]
            if len(above_half) > 1:
                fwhm = wavelengths[above_half[-1]] - wavelengths[above_half[0]]
            else:
                fwhm = 0
            
            spectra_info.append({
                'idx': idx,
                'row': row,
                'col': col,
                'spectrum': spectrum,
                'peak_wl': peak_wl,
                'peak_intensity': peak_intensity,
                'fwhm': fwhm,
                'integrated_intensity': spectrum.sum()
            })
        
        # FWHM tolerance 체크
        fwhms = [s['fwhm'] for s in spectra_info if s['fwhm'] > 0]
        if len(fwhms) > 1:
            fwhm_std = np.std(fwhms)
            fwhm_mean = np.mean(fwhms)
            print(f"  FWHM: {fwhm_mean:.1f} ± {fwhm_std:.1f} nm")
            
            if fwhm_std > args['PEAK_TOL_NM']:
                print(f"  [Skipped] FWHM variation ({fwhm_std:.1f}) > tolerance ({args['PEAK_TOL_NM']})")
                continue
        
        # 가장 높은 intensity를 가진 픽셀 선택
        best_idx = max(range(len(spectra_info)), 
                      key=lambda i: spectra_info[i]['peak_intensity'])
        best = spectra_info[best_idx]
        
        print(f"  Selected pixel ({best['row']}, {best['col']})")
        print(f"  - Peak: {best['peak_wl']:.1f} nm @ {best['peak_intensity']:.1f}")
        print(f"  - FWHM: {best['fwhm']:.1f} nm")
        
        representatives.append({
            'label': label,
            'row': best['row'],
            'col': best['col'],
            'spectrum': best['spectrum'],
            'peak_wl': best['peak_wl'],
            'peak_intensity': best['peak_intensity'],
            'fwhm': best['fwhm'],
            'cluster_size': len(coords)
        })
    
    print(f"\n[Summary] {len(representatives)} valid particles from {len(clusters)} clusters")
    return representatives

def select_manual_representatives(self):
    """Manual 좌표에서 대표 스펙트럼 선택"""
    representatives = []
    
    print(f"\n[info] Processing {len(self.args['MANUAL_COORDS'])} manual coordinates...")
    
    for i, (row, col) in enumerate(self.args['MANUAL_COORDS']):
        # 좌표 유효성 검사
        if row < 0 or row >= self.cube.shape[0] or col < 0 or col >= self.cube.shape[1]:
            print(f"[warning] Invalid coordinate ({row}, {col}) - skipping")
            continue
        
        # 스펙트럼 추출
        spectrum = self.cube[row, col, :]
        
        # Peak 정보
        peak_idx = np.argmax(spectrum)
        peak_wl = self.wvl[peak_idx]
        peak_intensity = spectrum[peak_idx]
        
        # FWHM 추정
        half_max = peak_intensity / 2
        above_half = np.where(spectrum > half_max)[0]
        if len(above_half) > 1:
            fwhm = self.wvl[above_half[-1]] - self.wvl[above_half[0]]
        else:
            fwhm = 0
        
        representatives.append({
            'label': i + 1,
            'row': row,
            'col': col,
            'spectrum': spectrum,
            'peak_wl': peak_wl,
            'peak_intensity': peak_intensity,
            'fwhm': fwhm,
            'cluster_size': 1  # Manual은 1픽셀
        })
        
        print(f"  Manual point {i}: ({row},{col}) - λ_peak={peak_wl:.1f}nm, intensity={peak_intensity:.1f}")
    
    return representatives

def save_debug_image(args, img, name, cmap='hot'):
      
    out_dir = Path(args['OUTPUT_DIR']) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if img.ndim == 3:  # RGB
        ax.imshow(img, origin='lower')
        ax.set_title(f"{name} (RGB)")
    else:  # Grayscale
        im = ax.imshow(img, cmap=cmap, origin='lower')
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{name} (range: [{img.min():.2f}, {img.max():.2f}])")
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{args['SAMPLE_NAME']}_{name}.png", dpi=150)
    plt.close()

def save_debug_dfs_detection(args, max_map, labels, clusters):
       
    out_dir = Path(args['OUTPUT_DIR']) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Max intensity map
    im1 = ax1.imshow(max_map, cmap='hot', origin='lower')
    ax1.set_title('Max Intensity Map (500-800nm)')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Binary mask (threshold 적용 후)
    threshold = args.get('DFS_INTENSITY_THRESHOLD', 0.1)
    normalized = (max_map - max_map.min()) / (max_map.max() - max_map.min())
    mask = normalized > threshold
    ax2.imshow(mask, cmap='gray', origin='lower')
    ax2.set_title(f'Binary Mask (threshold={threshold})')
    
    # 3. Labeled clusters
    ax3.imshow(labels, cmap='tab20', origin='lower')
    ax3.set_title(f'Detected Clusters (n={len(clusters)})')
    
    # 클러스터 중심 표시
    for cluster in clusters:
        center = cluster['center']
        ax3.plot(center[1], center[0], 'w+', markersize=10, markeredgewidth=2)
        ax3.text(center[1]+2, center[0]+2, str(cluster['label']), 
                color='white', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{args['SAMPLE_NAME']}_dfs_detection.png", dpi=150)
    plt.close()