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
    Flatfield correction with proper wavelength matching
    cube: (H, W, L) - already cropped data
    wvl: (L,) - wavelengths after cropping
    """
    # Load white and dark references
    w_cube, w_wvl = tdms_to_cube(white_path)
    d_cube, d_wvl = tdms_to_cube(dark_path)
    
    print(f"[debug] Sample wavelengths: {wvl.min():.1f}-{wvl.max():.1f} nm ({len(wvl)} points)")
    print(f"[debug] White wavelengths: {w_wvl.min():.1f}-{w_wvl.max():.1f} nm ({len(w_wvl)} points)")
    
    # Find matching wavelength indices
    # White/Dark가 더 많은 포인트를 가지므로 (1340 vs 670)
    # 각 샘플 파장에 대해 가장 가까운 white/dark 파장 찾기
    idxs = []
    for wl_val in wvl:
        idx = np.argmin(np.abs(w_wvl - wl_val))
        idxs.append(idx)
    idxs = np.array(idxs)
    
    print(f"[debug] Wavelength matching: using indices {idxs[0]}-{idxs[-1]} from white/dark")
    
    # Extract matching wavelengths
    w_crop = w_cube[:, :, idxs]
    d_crop = d_cube[:, :, idxs]
    
    # Spatial average to get reference spectra
    w_ref = w_crop.mean(axis=(0, 1))  # (L,)
    d_ref = d_crop.mean(axis=(0, 1))  # (L,)
    
    print(f"[debug] White reference range: [{w_ref.min():.1f}, {w_ref.max():.1f}]")
    print(f"[debug] Dark reference range: [{d_ref.min():.1f}, {d_ref.max():.1f}]")
    
    # Apply correction
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

    if args["BACKGROUND_PERC"] > 0:

        bg = np.quantile(cube, args["BACKGROUND_PERC"], axis=2, keepdims=True)
        cube = np.maximum(cube - bg, 0)

    return cube.astype(np.float32), wavelengths

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

