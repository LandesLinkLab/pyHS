import numpy as np
from pathlib import Path
from nptdms import TdmsFile
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.ndimage import maximum_filter
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, footprint_rectangle
from skimage.measure import label

from typing import List, Dict, Tuple, Optional, Any, Union

def tdms_to_cube(path: str, 
                image_shape: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
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
                    white_path: str, 
                    dark_path: str) -> np.ndarray:
    """
    MATLAB standardreadindark.m과 동일한 flatfield correction
    Column 방향으로만 평균을 구함
    """
    # Load white and dark references
    w_cube, w_wvl = tdms_to_cube(white_path)
    d_cube, d_wvl = tdms_to_cube(dark_path)
    
    H_w, W_w, L_w = w_cube.shape
    H_d, W_d, L_d = d_cube.shape
    L_sample = len(wvl)
    
    print(f"[debug] White shape: {w_cube.shape}, Dark shape: {d_cube.shape}")
    print(f"[debug] Sample wavelengths: {wvl.min():.1f}-{wvl.max():.1f} nm ({len(wvl)} points)")
    print(f"[debug] White wavelengths: {w_wvl.min():.1f}-{w_wvl.max():.1f} nm ({len(w_wvl)} points)")
    
    # MATLAB: stan = sum(stanim,2)/pcol; % column 방향으로만 평균
    # Python axis: 0=H(row), 1=W(col), 2=L(wavelength)
    # MATLAB dim 2 = Python axis 1
    
    # Column 방향으로 평균 (각 row와 wavelength에 대해 column들을 평균)
    stan = w_cube.mean(axis=1, keepdims=True)  # shape: (H, 1, L)
    dark = d_cube.mean(axis=1, keepdims=True)  # shape: (H, 1, L)
    
    # MATLAB: stan = stan - dark
    stan = stan - dark
    
    # 파장 맞추기 (2:1 비율인 경우)
    if L_w == 2 * L_sample:
        print(f"[debug] Using MATLAB-style 2-point averaging (1340 -> 670)")
        # 2개씩 평균
        stan = stan.reshape(H_w, 1, L_sample, 2).mean(axis=3)
    else:
        # Nearest neighbor matching
        print(f"[warning] White/Dark not exactly 2x sample size. Using nearest neighbor matching.")
        idxs = []
        for wl_val in wvl:
            idx = np.argmin(np.abs(w_wvl - wl_val))
            idxs.append(idx)
        stan = stan[:, :, idxs]
    
    print(f"[debug] Stan shape after wavelength matching: {stan.shape}")
    
    # Sample cube의 각 row에 맞는 flatfield 선택
    H_sample, W_sample, L_sample_check = cube.shape
    
    if H_sample <= H_w:
        # 각 row에 해당하는 flatfield 사용
        corrected = np.zeros_like(cube)
        
        for row in range(H_sample):
            if row < stan.shape[0]:
                # 해당 row의 flatfield 값으로 나누기
                flat_row = stan[row, 0, :]  # (L,)
                # Division 
                denominator = np.where(flat_row > 0, flat_row, 1.0)
                corrected[row, :, :] = cube[row, :, :] / denominator[None, :]
            else:
                # Row index가 범위를 벗어나면 마지막 row 사용
                flat_row = stan[-1, 0, :]
                denominator = np.where(flat_row > 0, flat_row, 1.0)
                corrected[row, :, :] = cube[row, :, :] / denominator[None, :]
    else:
        print(f"[warning] Sample has more rows ({H_sample}) than flatfield ({H_w})")
        # Fallback: 전체 평균 사용
        stan_mean = stan.mean(axis=0, keepdims=True)  # (1, 1, L)
        denominator = np.where(stan_mean > 0, stan_mean, 1.0)
        corrected = cube / denominator
    
    # Clip extreme values
    corrected = np.clip(corrected, 0, 10)
    
    print(f"[debug] After flatfield: range [{corrected.min():.3f}, {corrected.max():.3f}]")
    
    return corrected

def crop_and_bg(args: Dict[str, Any],
                cube: np.ndarray, 
                wavelengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop wavelength range and apply background subtraction
    """
    m = (wavelengths >= args["CROP_RANGE_NM"][0]) & (wavelengths <= args["CROP_RANGE_NM"][1])
    cube = cube[:, :, m]
    wavelengths = wavelengths[m]

    bg_mode = args.get('BACKGROUND_MODE', 'global')

    if bg_mode == 'global':
        # MATLAB 방식의 global background
        backper = args.get('BACKGROUND_PERCENTILE', 0.1)  # 10%
        
        H, W, L = cube.shape
        
        # 모든 파장에 대한 합
        sumnorm = cube.sum(axis=2)
        
        # Normalize (MATLAB 방식)
        sumnorm_min = sumnorm.min()
        sumnorm_range = sumnorm.max() - sumnorm_min
        if sumnorm_range > 0:
            sumnorm = (sumnorm - sumnorm_min) / sumnorm_range
        
        # 가장 어두운 픽셀들 찾기
        smln = int(np.ceil(backper * H * W))  # 전체 픽셀의 10%
        
        # 모든 픽셀값을 정렬
        sorted_values = np.sort(sumnorm.ravel())
        
        # Cutoff value (하위 10%의 상한값)
        if smln < len(sorted_values):
            nthsmlst = sorted_values[smln]
        else:
            nthsmlst = sorted_values[-1]
        
        # 가장 어두운 픽셀들의 위치 찾기
        dark_mask = sumnorm <= nthsmlst
        dark_indices = np.where(dark_mask)
        
        # 실제로 사용할 픽셀 수 (정확히 smln개)
        n_dark = min(len(dark_indices[0]), smln)
        
        print(f"[debug] Using {n_dark} darkest pixels ({backper*100:.1f}% of {H*W} total pixels)")
        
        # Background spectrum 계산
        bkavg = np.zeros(L)
        for i in range(n_dark):
            row = dark_indices[0][i]
            col = dark_indices[1][i]
            bkavg += cube[row, col, :]
        
        bkavg = bkavg / n_dark  # 평균
        
        print(f"[debug] Background range: [{bkavg.min():.2f}, {bkavg.max():.2f}]")
        
        # Background subtraction
        cube = cube - bkavg[None, None, :]
        cube = np.maximum(cube, 0)  # No negative values

    return cube.astype(np.float32), wavelengths

def apply_local_background(args, cube, clusters, representatives):
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
    DFS용 wrapper - RGB 이미지 생성 후 MATLAB 방식 적용
    """
    # DFS max map을 RGB-like 형태로 변환
    # MATLAB의 specrgbnorm처럼 3채널로 만들기
    normalized = (max_map - max_map.min()) / (max_map.max() - max_map.min() + 1e-10)
    
    # 3채널 RGB image 생성 (grayscale을 3채널로 복제)
    rgb_image = np.stack([normalized, normalized, normalized], axis=2)
    
    # MATLAB 방식으로 detection
    return detect_particles_matlab_style(rgb_image, args)

def detect_particles_matlab_style(rgb_image, args):
    """
    MATLAB partident.m과 동일한 방식의 particle detection
    다중 threshold를 시도하며 개별 픽셀을 찾는 방식
    """
    # Parameters from MATLAB
    lower = args.get('PARTICLE_LOWER_BOUND', 0)
    upper = args.get('PARTICLE_UPPER_BOUND', 0.5)
    nhood = args.get('NHOOD_SIZE', 1)  # odd number
    
    # RGB to grayscale (sum of all channels)
    srgb = rgb_image.sum(axis=2)
    H, W = srgb.shape
    
    # Create mask to exclude edges (MATLAB의 picture frame)
    half_nhood = (nhood - 1) // 2
    mask = np.zeros((H, W), dtype=bool)
    mask[half_nhood:H-half_nhood, half_nhood:W-half_nhood] = True
    
    # Collect unique points from all thresholds
    pts = []
    
    # MATLAB: for i=lower:0.001:upper
    thresholds = np.arange(lower, upper + 0.001, 0.001)
    
    print(f"[debug] Testing {len(thresholds)} thresholds from {lower} to {upper}")
    
    for thresh_idx, thresh in enumerate(thresholds):
        # Binary thresholding
        rbs = (rgb_image > thresh).any(axis=2)  # any channel > threshold
        
        # Clear border and apply mask
        rbs = ndi.binary_fill_holes(rbs)
        rbs[~mask] = False
        
        # Find connected components
        labeled, num_features = label(rbs, return_num=True)
        
        if num_features > 0:
            # Get component sizes
            sizes = np.bincount(labeled.ravel())[1:]  # exclude background (0)
            
            # MATLAB logic: remove largest component until only single pixels remain
            while sizes.max() > 1:
                # Remove the largest component
                largest_label = np.argmax(sizes) + 1
                rbs[labeled == largest_label] = False
                
                # Re-label
                labeled, num_features = label(rbs, return_num=True)
                if num_features == 0:
                    break
                sizes = np.bincount(labeled.ravel())[1:]
            
            # Collect remaining single pixels
            single_pixels = np.where(rbs)
            if len(single_pixels[0]) > 0:
                pts.extend(zip(single_pixels[0], single_pixels[1]))
        
        if thresh_idx % 100 == 0:
            print(f"  Processed threshold {thresh:.3f}: found {len(pts)} points so far")
    
    # Remove duplicates (MATLAB: ptu=unique(pts))
    unique_pts = list(set(pts))
    print(f"[debug] Total unique points found: {len(unique_pts)}")
    
    # Convert to clusters format for compatibility
    clusters = []
    labels = np.zeros((H, W), dtype=int)
    
    for i, (row, col) in enumerate(unique_pts):
        clusters.append({
            'label': i + 1,
            'coords': np.array([[row, col]]),
            'size': 1,
            'center': np.array([row, col]),
            'max_intensity': srgb[row, col],
            'mean_intensity': srgb[row, col]
        })
        labels[row, col] = i + 1
    
    print(f"[info] MATLAB-style detection found {len(clusters)} particles")
    
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