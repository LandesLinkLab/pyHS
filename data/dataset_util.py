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

def crop_wavelength(cube: np.ndarray, wavelengths: np.ndarray, wl_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop wavelength range
    """
    m = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
    cube_cropped = cube[:, :, m]
    wavelengths_cropped = wavelengths[m]
    
    print(f"[debug] Wavelength cropped: {wavelengths.min():.1f}-{wavelengths.max():.1f} nm -> "
          f"{wavelengths_cropped.min():.1f}-{wavelengths_cropped.max():.1f} nm")
    
    return cube_cropped, wavelengths_cropped

def flatfield_correct(cube: np.ndarray, 
                    wvl: np.ndarray, 
                    white_path: str, 
                    dark_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB standardreadindark.m과 동일한 flatfield correction
    Column 방향으로만 평균을 구함
    Returns: corrected_cube, white_ref, dark_ref (나중에 background correction에 사용)
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
        dark = dark.reshape(H_d, 1, L_sample, 2).mean(axis=3)
    else:
        # Nearest neighbor matching
        print(f"[warning] White/Dark not exactly 2x sample size. Using nearest neighbor matching.")
        idxs = []
        for wl_val in wvl:
            idx = np.argmin(np.abs(w_wvl - wl_val))
            idxs.append(idx)
        stan = stan[:, :, idxs]
        dark = dark[:, :, idxs]
    
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
    
    # Return corrected cube and references for later use
    return corrected, stan, dark

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
    Detection style에 따라 Python 또는 MATLAB 방식 사용
    """
    detection_style = args.get('PARTICLE_DETECTION_STYLE', 'python')
    
    if detection_style == 'matlab':
        # MATLAB 방식: RGB 이미지 생성 후 다중 threshold
        normalized = (max_map - max_map.min()) / (max_map.max() - max_map.min() + 1e-10)
        rgb_image = np.stack([normalized, normalized, normalized], axis=2)
        return detect_particles_matlab_style(rgb_image, args)
    else:
        # Python 방식: threshold 기반 connected components
        return detect_particles_python_style(max_map, args)

def detect_particles_python_style(max_map, args):
    """
    Python 방식의 particle detection (기존 코드)
    Threshold 기반 connected component 분석
    """
    # Check if map has valid data
    if max_map.max() <= max_map.min():
        print("[warning] Max map has no contrast")
        mask = max_map > np.percentile(max_map, 90)
    else:
        # Normalize for thresholding
        normalized = (max_map - max_map.min()) / (max_map.max() - max_map.min())
        
        # Adaptive threshold
        threshold = args.get('DFS_INTENSITY_THRESHOLD', 0.1)
        
        # Try Otsu if manual threshold fails
        try:
            otsu_val = threshold_otsu(normalized)
            print(f"[debug] Otsu threshold on normalized data: {otsu_val:.3f}")
            threshold = min(threshold, otsu_val * 0.8)
        except:
            pass
        
        mask = normalized > threshold
        print(f"[debug] Using threshold: {threshold}")
    
    print(f"[debug] Pixels above threshold: {mask.sum()}")
    
    # If too few pixels, lower threshold
    if mask.sum() < 50:
        print("[warning] Too few pixels detected, lowering threshold")
        percentile_threshold = 80
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
        
        local_max = maximum_filter(max_map, size=5)
        peaks = (max_map == local_max) & (max_map > np.percentile(max_map, 90))
        peak_coords = np.argwhere(peaks)
        
        print(f"[debug] Found {len(peak_coords)} peaks")
        
        for i, (row, col) in enumerate(peak_coords[:20]):
            clusters.append({
                'label': i + 1,
                'coords': np.array([[row, col]]),
                'size': 1,
                'center': np.array([row, col]),
                'max_intensity': max_map[row, col],
                'mean_intensity': max_map[row, col]
            })
    
    print(f"[info] Python-style detection returning {len(clusters)} valid clusters")
    return labels, clusters

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
    
    print(f"[debug] MATLAB-style: Testing {len(thresholds)} thresholds from {lower} to {upper}")
    
    for thresh_idx, thresh in enumerate(thresholds):
        # Binary thresholding
        rbs = (rgb_image > thresh).any(axis=2)
        
        # Clear border and apply mask
        rbs = ndi.binary_fill_holes(rbs)
        rbs[~mask] = False
        
        # Find connected components
        labeled, num_features = label(rbs, return_num=True)
        
        if num_features > 0:
            # Get component sizes
            sizes = np.bincount(labeled.ravel())[1:]
            
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

def apply_background_correction(cube, wvl, clusters, args, white_ref, dark_ref, raw_cube):
    """
    Apply background correction to cube
    I_corr = {S_raw(λ) / (W(λ) - D(λ))} - {B(λ) / (W(λ) - D(λ))}
    
    Note: cube는 이미 flatfield corrected 상태 {S_raw(λ) / (W(λ) - D(λ))}
    """
    bg_mode = args.get('BACKGROUND_MODE', 'global')
    
    if bg_mode == 'global':
        # MATLAB 방식의 global background
        corrected = apply_global_background(cube, wvl, args, white_ref, dark_ref, raw_cube)
    else:
        # Local background
        corrected = apply_local_background(cube, wvl, clusters, args, white_ref, dark_ref, raw_cube)
    
    return corrected

def apply_global_background(cube, wvl, args, white_ref, dark_ref, raw_cube):
    """
    MATLAB anfunc_lorentz_fit.m과 동일한 global background subtraction
    전체 이미지에서 가장 어두운 픽셀들을 찾아 background로 사용
    """
    # Parameters
    backper = args.get('BACKGROUND_PERCENTILE', 0.1)  # 10%
    
    H, W, L = cube.shape
    
    # 모든 파장에 대한 합 (이미 flatfield corrected)
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
    
    # Background spectrum 계산 - RAW 데이터에서!
    bkavg_raw = np.zeros(L)
    for i in range(n_dark):
        row = dark_indices[0][i]
        col = dark_indices[1][i]
        bkavg_raw += raw_cube[row, col, :]  # raw spectrum 사용!
    
    bkavg_raw = bkavg_raw / n_dark  # 평균
    
    # Background에도 flatfield correction 적용
    # 전체 이미지의 평균 flatfield 사용
    avg_flatfield = (white_ref - dark_ref).mean(axis=0).squeeze()
    denominator = np.where(avg_flatfield > 0, avg_flatfield, 1.0)
    bkavg_corrected = bkavg_raw / denominator
    
    print(f"[debug] Background (raw): [{bkavg_raw.min():.2f}, {bkavg_raw.max():.2f}]")
    print(f"[debug] Background (corrected): [{bkavg_corrected.min():.2f}, {bkavg_corrected.max():.2f}]")
    
    # Background subtraction
    # cube는 이미 {S_raw(λ) / (W(λ) - D(λ))}, bkavg_corrected는 {B(λ) / (W(λ) - D(λ))}
    corrected = cube - bkavg_corrected[None, None, :]
    corrected = np.maximum(corrected, 0)  # No negative values
    
    return corrected

def apply_local_background(cube, wvl, clusters, args, white_ref, dark_ref, raw_cube):
    """
    Local background correction
    각 클러스터 주변에서 가장 어두운 영역을 background로 사용
    """
    print(f"\n[debug] Applying local background correction")
    
    H, W, L = cube.shape
    corrected = cube.copy()
    
    # 각 클러스터에 대해 local background 적용
    for cluster in clusters:
        coords = cluster['coords']
        center = cluster['center']
        
        # Local background region 찾기
        search_radius = args.get('BACKGROUND_LOCAL_SEARCH_RADIUS', 20)
        percentile = args.get('BACKGROUND_LOCAL_PERCENTILE', 1)
        
        # 검색 영역 정의 (클러스터 중심 주변)
        row_center, col_center = int(center[0]), int(center[1])
        row_min = max(0, row_center - search_radius)
        row_max = min(H, row_center + search_radius)
        col_min = max(0, col_center - search_radius)
        col_max = min(W, col_center + search_radius)
        
        # 검색 영역에서 intensity sum이 낮은 픽셀들 찾기
        search_region = cube[row_min:row_max, col_min:col_max, :]
        intensity_map = search_region.sum(axis=2)
        
        # 클러스터 픽셀은 제외
        mask = np.ones(intensity_map.shape, dtype=bool)
        for coord in coords:
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
                # Raw data에서 background 계산
                background_raw = np.zeros(L)
                for i in range(len(dark_pixels[0])):
                    global_r = dark_pixels[0][i] + row_min
                    global_c = dark_pixels[1][i] + col_min
                    background_raw += raw_cube[global_r, global_c, :]  # raw 사용!
                
                background_raw /= len(dark_pixels[0])
                
                # Background에 flatfield correction 적용
                avg_row = (row_min + row_max) // 2
                if avg_row < white_ref.shape[0]:
                    local_flatfield = white_ref[avg_row, 0, :] - dark_ref[avg_row, 0, :]
                else:
                    local_flatfield = white_ref[-1, 0, :] - dark_ref[-1, 0, :]
                
                denominator = np.where(local_flatfield > 0, local_flatfield, 1.0)
                background_corrected = background_raw / denominator
                
                print(f"  Cluster {cluster['label']}: Using {len(dark_pixels[0])} pixels for local background")
                print(f"    Background range: [{background_corrected.min():.1f}, {background_corrected.max():.1f}]")
                
                # 클러스터의 모든 픽셀에 local background subtraction 적용
                for coord in coords:
                    r, c = coord
                    corrected[r, c, :] = np.maximum(cube[r, c, :] - background_corrected, 0)
            else:
                print(f"  Cluster {cluster['label']}: No suitable background pixels nearby")
        else:
            print(f"  Cluster {cluster['label']}: No valid pixels in search region")
    
    print(f"[debug] After local background correction: range [{corrected.min():.3f}, {corrected.max():.3f}]")
    
    return corrected

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
    """Save debug images for particle detection results"""
    out_dir = Path(args['OUTPUT_DIR']) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Detection style에 따라 다른 시각화
    detection_style = args.get('PARTICLE_DETECTION_STYLE', 'python')
    
    if detection_style == 'python':
        # Python 방식: Otsu threshold 과정 시각화
        save_debug_python_detection(args, max_map, labels, clusters, out_dir)
    else:
        # MATLAB 방식: 간단한 결과만
        save_debug_matlab_detection(args, max_map, labels, clusters, out_dir)

def save_debug_python_detection(args, max_map, labels, clusters, out_dir):
    """Python 방식 detection의 상세 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # 1. Original max intensity map
    im1 = axes[0].imshow(max_map, cmap='hot', origin='lower')
    axes[0].set_title('1. Max Intensity Map')
    plt.colorbar(im1, ax=axes[0])
    
    # 2. Normalized map
    normalized = (max_map - max_map.min()) / (max_map.max() - max_map.min() + 1e-10)
    im2 = axes[1].imshow(normalized, cmap='hot', origin='lower')
    axes[1].set_title('2. Normalized Map')
    plt.colorbar(im2, ax=axes[1])
    
    # 3. Otsu threshold visualization
    threshold = args.get('DFS_INTENSITY_THRESHOLD', 0.1)
    try:
        from skimage.filters import threshold_otsu
        otsu_val = threshold_otsu(normalized)
        actual_threshold = min(threshold, otsu_val * 0.8)
        
        # Histogram with thresholds
        axes[2].hist(normalized.ravel(), bins=100, alpha=0.7, color='blue')
        axes[2].axvline(actual_threshold, color='red', linestyle='--', linewidth=2, label=f'Used: {actual_threshold:.3f}')
        axes[2].axvline(otsu_val, color='green', linestyle='--', linewidth=2, label=f'Otsu: {otsu_val:.3f}')
        axes[2].axvline(threshold, color='orange', linestyle='--', linewidth=2, label=f'Manual: {threshold:.3f}')
        axes[2].set_title('3. Histogram with Thresholds')
        axes[2].set_xlabel('Normalized Intensity')
        axes[2].set_ylabel('Count')
        axes[2].legend()
        axes[2].set_yscale('log')
    except:
        actual_threshold = threshold
        axes[2].text(0.5, 0.5, 'Otsu failed', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('3. Histogram (Otsu failed)')
    
    # 4. Binary mask after threshold
    mask = normalized > actual_threshold
    axes[3].imshow(mask, cmap='gray', origin='lower')
    axes[3].set_title(f'4. Binary Mask (threshold={actual_threshold:.3f})')
    pixels_above = mask.sum()
    axes[3].text(0.02, 0.98, f'Pixels: {pixels_above}', transform=axes[3].transAxes, 
                verticalalignment='top', color='yellow', fontsize=10)
    
    # 5. After morphological operations
    from skimage.morphology import binary_closing, remove_small_objects, footprint_rectangle
    mask_morph = binary_closing(mask, footprint_rectangle((3, 3)))
    min_size = max(2, args.get("MIN_PIXELS_CLUS", 4) // 2)
    mask_morph = remove_small_objects(mask_morph, min_size)
    axes[4].imshow(mask_morph, cmap='gray', origin='lower')
    axes[4].set_title(f'5. After Morphology (min_size={min_size})')
    
    # 6. Final labeled clusters
    im6 = axes[5].imshow(labels, cmap='tab20', origin='lower')
    axes[5].set_title(f'6. Final Clusters (n={len(clusters)})')
    
    # Mark cluster centers and info
    for cluster in clusters:
        center = cluster['center']
        axes[5].plot(center[1], center[0], 'w+', markersize=10, markeredgewidth=2)
        axes[5].text(center[1]+2, center[0]+2, f"{cluster['label']}\n{cluster['size']}px", 
                    color='white', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Add grid to all subplots
    for ax in axes:
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'Python-style Particle Detection Debug - {args["SAMPLE_NAME"]}', fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir / f"{args['SAMPLE_NAME']}_python_detection_debug.png", dpi=150)
    plt.close()

def save_debug_matlab_detection(args, max_map, labels, clusters, out_dir):
    """MATLAB 방식 detection의 간단한 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 1. Max intensity map
    im1 = ax1.imshow(max_map, cmap='hot', origin='lower')
    ax1.set_title('Max Intensity Map')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Detected particles
    ax2.imshow(labels, cmap='tab20', origin='lower')
    ax2.set_title(f'MATLAB-style Detection (n={len(clusters)} particles)')
    
    # Mark particles
    for cluster in clusters:
        center = cluster['center']
        ax2.plot(center[1], center[0], 'w+', markersize=8, markeredgewidth=1)
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{args['SAMPLE_NAME']}_matlab_detection.png", dpi=150)
    plt.close()