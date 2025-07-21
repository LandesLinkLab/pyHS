import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from nptdms import TdmsFile
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, footprint_rectangle, binary_closing, remove_small_objects
from scipy import ndimage as ndi

# ------------- Helper to identify wavelength axis -----------------
def _is_lambda_channel(ch):
    # 1) float 배열이고 단조증가(monotonic)하면 λ-axis
    try:
        arr = ch[:]  # numpy array
        if isinstance(arr, np.ndarray) and arr.dtype.kind == 'f' \
           and arr.ndim == 1 and arr.size > 1 \
           and np.all(np.diff(arr) > 0):
            return True
    except:
        pass

    # 2) (기존 로직) 이름/단위 기반 식별
    name = ch.name.lower()
    if any(key in name for key in ("wave", "lambda", "wl", "wavelength", "nm")):
        return True
    unit = str(ch.properties.get("unit_string")
               or ch.properties.get("Unit") or "").lower()
    return "nm" in unit

# -------------------- TDMS -> cube --------------------------------
# dataset_util.py의 tdms_to_cube 함수를 이것으로 교체

def tdms_to_cube(path: Path,
                 image_shape: Optional[Tuple[int, int]] = None):
    """
    MATLAB‑버전과 동일한 로직으로 TDMS → (H,W,L) cube 변환
    파장 축 문제 수정 버전
    """
    td = TdmsFile.read(path)
    ch_all = [ch for g in td.groups() for ch in g.channels()]

    # ── 1) λ‑축 채널 식별 ──────────────────────────────
    def _is_lambda(ch):
        v = ch[:]
        if v.ndim == 1 and v.size > 1:
            # 단조증가 체크를 제거하고 일단 1D 배열이면 후보로
            return True
        name = ch.name.lower()
        if any(k in name for k in ("wave", "lambda", "wl", "wavelength", "nm")):
            return True
        unit = str(ch.properties.get("unit_string")
                   or ch.properties.get("Unit") or "").lower()
        return "nm" in unit

    # 파장 채널 후보들 찾기
    wl_candidates = [c for c in ch_all if _is_lambda(c)]
    
    if not wl_candidates:
        raise RuntimeError("λ‑axis channel not found")
    
    # 파장 후보 중에서 합리적인 범위의 것 선택 (300-1000nm 정도)
    wl_ch = None
    for candidate in wl_candidates:
        vals = candidate[:].astype(np.float32)
        # 파장 범위가 합리적인지 체크
        if vals.min() > 100 and vals.max() < 2000 and vals.size > 100:
            wl_ch = candidate
            wl = vals
            break
    
    # 못 찾았으면 첫 번째 후보 사용하고 스케일 조정
    if wl_ch is None:
        wl_ch = wl_candidates[0]
        wl = wl_ch[:].astype(np.float32)
        
        # 파장이 비정상적으로 크면 스케일 조정 (아마 인덱스일 가능성)
        if wl.max() > 2000:
            print(f"[warning] Wavelength range seems wrong: {wl.min():.1f}-{wl.max():.1f}")
            print("[warning] Creating synthetic wavelength axis 400-900nm")
            # 400-900nm 범위로 재생성
            wl = np.linspace(400, 900, len(wl))

    print(f"[debug] Wavelength axis: {wl.min():.1f}-{wl.max():.1f} nm, {len(wl)} points")

    # ── 2) 스펙트럼 채널 목록 (λ‑축과 길이가 같은 채널) ──
    specs = [c for c in ch_all if len(c) == len(wl) and c is not wl_ch]
    Nspec = len(specs)
    if Nspec == 0:
        raise RuntimeError("No spectrum channels found")

    print(f"[debug] Found {Nspec} spectrum channels")

    # ── 3) (행, 열) 추론 — MATLAB 방식 우선 적용 ──────
    rows = cols = None

    if image_shape is not None:
        rows, cols = image_shape
    else:
        # 3‑A) 각 채널 Property 8,9,10 → pcol, startRow, endRow
        prop8 = specs[0].properties.get(8)   
        prop9 = specs[0].properties.get(9)
        prop10 = specs[0].properties.get(10)

        if prop8 and prop9 is not None and prop10 is not None:
            cols = int(prop8)
            rows = int(prop10 - prop9 + 1)

        # 3‑B) NI_ArrayRow / NI_ArrayColumn 메타
        if rows is None or cols is None:
            row_list = [c.properties.get("NI_ArrayRow") for c in specs
                        if "NI_ArrayRow" in c.properties]
            col_list = [c.properties.get("NI_ArrayColumn") for c in specs
                        if "NI_ArrayColumn" in c.properties]
            if row_list and col_list:
                rows = max(row_list) + 1
                cols = max(col_list) + 1

        # 3‑C) root 'strips', 'top pixel', 'bottom pixel'
        if rows is None or cols is None:
            strips = td.properties.get("strips")
            top = td.properties.get("top pixel")
            bot = td.properties.get("bottom pixel")
            if strips and top is not None and bot is not None:
                cols = int(bot - top + 1)
                rows = int(strips)

        # 3‑D) perfect‑square fallback
        if rows is None or cols is None:
            if int(np.sqrt(Nspec)) ** 2 == Nspec:
                rows = cols = int(np.sqrt(Nspec))

        # 3‑E) 마지막 보루: root Image_Height / Width
        if rows is None or cols is None:
            rows = td.properties.get("Image_Height")
            cols = td.properties.get("Image_Width")

        if rows is None or cols is None or rows * cols != Nspec:
            # 추정 시도
            if Nspec % 49 == 0:  # 49로 나누어떨어지면
                rows = 49
                cols = Nspec // 49
            elif Nspec % 189 == 0:  # 189로 나누어떨어지면
                rows = Nspec // 189
                cols = 189
            else:
                raise RuntimeError(f"Cannot infer image shape for {Nspec} channels")

    print(f"[debug] Image shape: {rows} x {cols}")

    # ── 4) MATLAB 과 동일한 채널 → 픽셀 매핑 ────────────
    def chan_key(ch):
        col = ch.properties.get("NI_ArrayColumn")
        row = ch.properties.get("NI_ArrayRow")
        return (col if col is not None else 0,
                row if row is not None else 0,
                ch.name)

    specs.sort(key=chan_key)

    # 벡터 → cube 변환
    stack = np.vstack([ch[:] for ch in specs])  # (Nspec, L)
    cube = stack.reshape(cols, rows, len(wl), order='C').transpose(1, 0, 2)  # (rows, cols, L)

    return cube.astype(np.float32), wl.astype(np.float32)

# ------------------- Remaining helpers ----------------------------
def flatfield_correct(cube: np.ndarray,
                      wvl: np.ndarray,
                      white_path: Path,
                      dark_path: Path) -> np.ndarray:
    """
    cube:  (H, W, L) — 이미 preprocess()로 crop된 데이터
    wvl:   (L,)       — preprocess() 후 self.wvl
    white_path, dark_path: raw TDMS 파일 경로
    """
    # 1) white/dark 모두 load 후,
    #    sample과 똑같이 파장 축 crop
    w_cube, w_wvl = tdms_to_cube(white_path)
    d_cube, d_wvl = tdms_to_cube(dark_path)

    # wavelength matching
    idxs = [int(np.argmin(np.abs(w_wvl - v))) for v in wvl]
    w_crop = w_cube[:, :, idxs]    # → shape (H_w, W_w, L)
    d_crop = d_cube[:, :, idxs]

    # 2) 공간 방향 평균해서 1D 참조 스펙트럼으로
    w_ref = w_crop.mean(axis=(0, 1))   # shape (L,)
    d_ref = d_crop.mean(axis=(0, 1))   # shape (L,)

    # 3) broadcast 보정
    num = cube - d_ref[None, None, :]
    den = np.clip((w_ref - d_ref)[None, None, :], 1e-9, None)
    return num / den

def crop_and_bg(cube, wavelengths, args):

    m = (wavelengths >= args["CROP_RANGE_NM"][0]) & (wavelengths <= args["CROP_RANGE_NM"][1])
    cube = cube[:, :, m]
    wavelengths = wavelengths[m]

    if args["BACKGROUND_PERC"] > 0:

        bg = np.quantile(cube, args["BACKGROUND_PERC"], axis=2, keepdims=True)
        cube = np.maximum(cube - bg, 0)

    return cube.astype(np.float32), wavelengths

def cube_to_rgb(cube, wavelengths):

    def idx(wl): return int(np.abs(wavelengths - wl).argmin())

    r,g,b = cube[..., idx(650)], cube[..., idx(550)], cube[..., idx(450)]

    return np.clip(np.stack([r,g,b], axis=-1) / np.percentile(cube, 99), 0,1)

# dataset_util.py의 label_particles 함수를 이것으로 교체

def label_particles(rgb, args):
    """
    Detect particles with improved thresholding
    """
    # RGB를 grayscale로 변환
    gray = rgb.mean(axis=2)
    
    # 디버그 정보
    print(f"[debug] Gray image stats: min={gray.min():.3f}, max={gray.max():.3f}, mean={gray.mean():.3f}")
    
    # Otsu threshold 계산
    try:
        thr = threshold_otsu(gray)
        print(f"[debug] Otsu threshold: {thr:.3f}")
    except:
        # Otsu가 실패하면 평균값 사용
        thr = gray.mean()
        print(f"[debug] Otsu failed, using mean: {thr:.3f}")
    
    # Threshold 적용
    threshold_value = thr * args["THRESH_HIGH"]
    print(f"[debug] Final threshold: {threshold_value:.3f} (Otsu * {args['THRESH_HIGH']})")
    
    # Binary mask 생성
    mask = gray > threshold_value
    
    # Morphological operations
    mask = binary_closing(mask, footprint_rectangle((3, 3)))
    mask = remove_small_objects(mask, args["MIN_PIXELS_CLUS"])
    
    # Label connected components
    labels, num = ndi.label(mask)
    
    print(f"[debug] Found {num} regions after filtering")
    
    # Calculate centroids
    if num > 0:
        cents = np.array(ndi.center_of_mass(mask, labels, range(1, num+1)))
        if cents.ndim == 1:
            cents = cents.reshape(1, -1)
    else:
        cents = np.array([])
    
    # 각 레이블의 크기 출력
    for i in range(1, num+1):
        size = np.sum(labels == i)
        print(f"  - Region {i}: {size} pixels")
    
    return labels, cents


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
        from skimage.filters import threshold_otsu
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
    from skimage.morphology import binary_closing, remove_small_objects, footprint_rectangle
    mask = binary_closing(mask, footprint_rectangle((3, 3)))
    
    # For initial detection, use smaller minimum size
    min_size = max(2, args.get("MIN_PIXELS_CLUS", 4) // 2)
    mask = remove_small_objects(mask, min_size)
    
    # Label connected components
    from scipy import ndimage as ndi
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
        from scipy.ndimage import maximum_filter
        
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

def save_dfs_particle_map(max_map, representatives, output_path, sample_name):
    """
    DFS 전용 particle map 저장
    - Max intensity map을 배경으로
    - 검출된 파티클 위치 표시
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Max intensity map 표시
    vmin, vmax = np.percentile(max_map[max_map > 0], [5, 95]) if np.any(max_map > 0) else (0, 1)
    
    im = ax.imshow(max_map,
                   cmap='hot',  # DFS 데이터에 적합한 colormap
                   origin='lower',  # 일반적인 좌표계
                   vmin=vmin,
                   vmax=vmax,
                   interpolation='nearest')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Max Intensity (500-800 nm)', fontsize=12)
    
    # 파티클 마커 표시
    for i, rep in enumerate(representatives):
        row, col = rep['row'], rep['col']
        
        # 흰색 원 with 검은 테두리 (잘 보이도록)
        circle_outer = plt.Circle((col, row), 
                                 radius=5,
                                 edgecolor='black',
                                 facecolor='none',
                                 linewidth=3)
        circle_inner = plt.Circle((col, row), 
                                 radius=5,
                                 edgecolor='white',
                                 facecolor='none',
                                 linewidth=2)
        ax.add_patch(circle_outer)
        ax.add_patch(circle_inner)
        
        # 파티클 번호
        ax.text(col + 7, row + 7,
                f'{i}',
                color='white',
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='black',
                         edgecolor='white',
                         alpha=0.8))
        
        # 추가 정보 (작은 글씨로)
        ax.text(col + 7, row - 7,
                f'{rep["peak_wl"]:.0f}nm',
                color='yellow',
                fontsize=8,
                fontweight='bold',
                ha='left')
    
    # 제목과 축 라벨
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