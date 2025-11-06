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
    Convert TDMS file to 3D hyperspectral cube (H,W,λ)
    
    This function reads TDMS files containing hyperspectral data and converts them
    into a 3D numpy array with dimensions (Height, Width, Wavelength).
    Wavelength information is extracted from the Info group's 'wvlths' channel.
    
    Parameters:
    -----------
    path : str
        Path to the TDMS file
    image_shape : Optional[Tuple[int, int]]
        Image shape as (rows, cols). If None, automatically determined from file properties
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (cube data as float32, wavelength array as float32)
    """
    td = TdmsFile.read(path)
    
    # ── 1) Extract wavelength information from Info group ──
    info_group = td['Info']
    if info_group is None:
        raise RuntimeError("No 'Info' group found in TDMS file")
    
    # Find wavelength channel 'wvlths'
    wl_channel = None
    for ch in info_group.channels():
        if ch.name == 'wvlths':
            wl_channel = ch
            break
    
    if wl_channel is None:
        raise RuntimeError("No 'wvlths' channel found in Info group")
    
    wl = wl_channel[:].astype(np.float32)
    print(f"[info] Wavelength array from TDMS: {wl.min():.1f}-{wl.max():.1f} nm, {len(wl)} points")
    
    # ── 2) Extract spectrum data from Spectra group ──
    spectra_group = td['Spectra']
    if spectra_group is None:
        raise RuntimeError("No 'Spectra' group found in TDMS file")
    
    specs = list(spectra_group.channels())
    Nspec = len(specs)
    print(f"[info] Found {Nspec} spectrum channels")
    
    if Nspec == 0:
        raise RuntimeError("No spectrum channels found")
    
    # ── 3) Determine image shape ──
    if image_shape is not None:
        rows, cols = image_shape
    else:
        # Read from root properties
        rows = int(td.properties.get('strips', 0))
        top = int(td.properties.get('top pixel', 0))
        bottom = int(td.properties.get('bottom pixel', 0))
        cols = bottom - top + 1
        
        # If calculated shape doesn't match, try known configurations
        if rows * cols != Nspec:
            if Nspec == 9261:  # Common configuration
                rows, cols = 49, 189
            elif Nspec == 8000:  # White/Dark reference
                rows, cols = 20, 400
            else:
                raise RuntimeError(f"Cannot determine image shape for {Nspec} channels")
    
    print(f"[info] Image shape: {rows} x {cols}")
    
    # ── 4) Create data cube ──
    # Convert each channel data to numpy array
    spectrum_length = len(specs[0][:])
    
    # Check if wavelength array length matches spectrum length
    if len(wl) != spectrum_length:
        print(f"[warning] Wavelength array length ({len(wl)}) != spectrum length ({spectrum_length})")
        # Truncate wavelength array if longer
        if len(wl) > spectrum_length:
            wl = wl[:spectrum_length]
        else:
            # This is problematic
            raise RuntimeError(f"Wavelength array too short: {len(wl)} < {spectrum_length}")
    
    # Stack all spectrum data
    data_list = []
    for i, ch in enumerate(specs):
        data = ch[:].astype(np.float32)
        data_list.append(data)
        if i % 1000 == 0:
            print(f"  Loading channel {i}/{Nspec}...")
    
    stack = np.vstack(data_list)  # Shape: (Nspec, L)
    
    # Reshape to 3D cube - MATLAB column-major order
    # MATLAB: for c = 1:cols, for r = 1:rows
    # This means column changes first, so use Fortran order
    cube = stack.reshape(cols, rows, spectrum_length, order='F').transpose(1, 0, 2)
    
    return cube.astype(np.float32), wl.astype(np.float32)

def crop_wavelength(cube: np.ndarray, wavelengths: np.ndarray, wl_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop hyperspectral cube to specified wavelength range
    
    Parameters:
    -----------
    cube : np.ndarray
        Input hyperspectral cube (H, W, λ)
    wavelengths : np.ndarray
        Wavelength array
    wl_range : Tuple[float, float]
        Wavelength range as (min_wl, max_wl)
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (cropped cube, cropped wavelengths)
    """
    # Create wavelength mask
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
    Apply flatfield correction identical to MATLAB standardreadindark.m
    
    This function performs flatfield correction by:
    1. Loading white and dark reference cubes
    2. Computing column-wise averages (MATLAB style)
    3. Subtracting dark from white to get flatfield
    4. Matching wavelengths (handles 2:1 ratio)
    5. Applying row-specific flatfield correction
    
    Parameters:
    -----------
    cube : np.ndarray
        Input hyperspectral cube (H, W, λ)
    wvl : np.ndarray
        Wavelength array for the sample
    white_path : str
        Path to white reference TDMS file
    dark_path : str
        Path to dark reference TDMS file

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (corrected_cube, flatfield_ref, dark_ref)
        
        corrected_cube: Flatfield-corrected hyperspectral cube, shape (H, W, L)
        flatfield_ref: (white - dark) already calculated, shape (H, 1, L)
        dark_ref: Dark reference average (for reference only), shape (H, 1, L)
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
    
    # MATLAB: stan = sum(stanim,2)/pcol; % average along column direction only
    # Python axis: 0=H(row), 1=W(col), 2=L(wavelength)
    # MATLAB dim 2 = Python axis 1
    
    # Average along column direction (for each row and wavelength, average across columns)
    stan = w_cube.mean(axis=1, keepdims=True)  # shape: (H, 1, L)
    dark = d_cube.mean(axis=1, keepdims=True)  # shape: (H, 1, L)
    
    # MATLAB: stan = stan - dark
    stan = stan - dark
    
    # Handle wavelength matching (common case: 2:1 ratio)
    if L_w == 2 * L_sample:
        print(f"[debug] Using MATLAB-style 2-point averaging (1340 -> 670)")
        # Average every 2 points
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
    
    # Apply flatfield correction to sample cube
    H_sample, W_sample, L_sample_check = cube.shape
    
    if H_sample <= H_w:
        # Use row-specific flatfield for each sample row
        corrected = np.zeros_like(cube)
        
        for row in range(H_sample):
            if row < stan.shape[0]:
                # Use corresponding row's flatfield values
                flat_row = stan[row, 0, :]  # Shape: (L,)
                # Avoid division by zero
                denominator = np.where(flat_row > 0, flat_row, 1.0)
                corrected[row, :, :] = cube[row, :, :] / denominator[None, :]
            else:
                # If row index exceeds flatfield range, use last row
                flat_row = stan[-1, 0, :]
                denominator = np.where(flat_row > 0, flat_row, 1.0)
                corrected[row, :, :] = cube[row, :, :] / denominator[None, :]
    else:
        print(f"[warning] Sample has more rows ({H_sample}) than flatfield ({H_w})")
        # Fallback: use overall average
        stan_mean = stan.mean(axis=0, keepdims=True)  # Shape: (1, 1, L)
        denominator = np.where(stan_mean > 0, stan_mean, 1.0)
        corrected = cube / denominator
    
    # Clip extreme values to reasonable range
    corrected = np.clip(corrected, 0, 10)
    
    print(f"[debug] After flatfield: range [{corrected.min():.3f}, {corrected.max():.3f}]")
    
    # Return corrected cube and references for later use
    return corrected, stan, dark

def create_dfs_max_intensity_map(cube, wavelengths, wl_range=(500, 800)):
    """
    Create maximum intensity projection map for DFS (Dark Field Scattering) data
    
    This function creates a 2D map by taking the maximum intensity value
    across a specified wavelength range for each pixel.
    
    Parameters:
    -----------
    cube : np.ndarray
        Hyperspectral cube (H, W, λ)
    wavelengths : np.ndarray
        Wavelength array
    wl_range : tuple
        Wavelength range for maximum projection (min_wl, max_wl)
    
    Returns:
    --------
    np.ndarray
        2D maximum intensity map (H, W)
    """
    # Create wavelength range mask
    mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
    
    if not np.any(mask):
        print(f"[warning] No wavelengths in range {wl_range}, using full range")
        mask = np.ones_like(wavelengths, dtype=bool)
    
    # Maximum projection across the specified wavelength range
    max_map = cube[:, :, mask].max(axis=2)
    
    print(f"[info] Max intensity map created from {mask.sum()} wavelengths "
          f"({wavelengths[mask].min():.1f}-{wavelengths[mask].max():.1f} nm)")
    print(f"[info] Map range: [{max_map.min():.2f}, {max_map.max():.2f}]")
    
    return max_map

def detect_dfs_particles(max_map, args):
    """
    Detect particle clusters from DFS maximum intensity map
    
    This function supports two detection styles:
    - 'python': Threshold-based connected component analysis
    - 'matlab': Multi-threshold single-pixel detection (mimics MATLAB partident.m)
    
    Parameters:
    -----------
    max_map : np.ndarray
        2D maximum intensity map
    args : dict
        Configuration dictionary containing detection parameters
    
    Returns:
    --------
    Tuple[np.ndarray, List[dict]]
        (labeled image, list of cluster information)
    """
    detection_style = args.get('PARTICLE_DETECTION_STYLE', 'python')
    
    if detection_style == 'matlab':
        # MATLAB style: RGB image generation followed by multi-threshold detection
        normalized = (max_map - max_map.min()) / (max_map.max() - max_map.min() + 1e-10)
        rgb_image = np.stack([normalized, normalized, normalized], axis=2)
        return detect_particles_matlab_style(rgb_image, args)
    else:
        # Python style: threshold-based connected components
        return detect_particles_python_style(max_map, args)

def detect_particles_python_style(max_map, args):
    """
    Python-style particle detection using threshold-based connected component analysis
    
    This method uses adaptive thresholding, morphological operations, and connected
    component analysis to identify particle clusters.
    
    Parameters:
    -----------
    max_map : np.ndarray
        2D maximum intensity map
    args : dict
        Configuration parameters including:
        - DFS_INTENSITY_THRESHOLD: Manual threshold value
        - MIN_PIXELS_CLUS: Minimum pixels per cluster
    
    Returns:
    --------
    Tuple[np.ndarray, List[dict]]
        (labeled image, cluster information list)
    """
    # Check if map has valid contrast
    if max_map.max() <= max_map.min():
        print("[warning] Max map has no contrast")
        mask = max_map > np.percentile(max_map, 90)
    else:
        # Normalize for thresholding
        normalized = (max_map - max_map.min()) / (max_map.max() - max_map.min())
        
        # Get adaptive threshold
        threshold = args.get('DFS_INTENSITY_THRESHOLD', 0.1)
        
        # Try Otsu threshold as a reference
        try:
            otsu_val = threshold_otsu(normalized)
            print(f"[debug] Otsu threshold on normalized data: {otsu_val:.3f}")
            threshold = min(threshold, otsu_val * 0.8)
        except:
            pass
        
        mask = normalized > threshold
        print(f"[debug] Using threshold: {threshold}")
    
    print(f"[debug] Pixels above threshold: {mask.sum()}")
    
    # If too few pixels detected, lower threshold
    if mask.sum() < 50:
        print("[warning] Too few pixels detected, lowering threshold")
        percentile_threshold = 80
        threshold_value = np.percentile(max_map, percentile_threshold)
        mask = max_map > threshold_value
        print(f"[debug] Using percentile {percentile_threshold}: {mask.sum()} pixels above threshold")
    
    # Morphological operations to clean up the mask
    mask = binary_closing(mask, footprint_rectangle((3, 3)))
    
    # For initial detection, use smaller minimum size
    min_size = max(2, args.get("MIN_PIXELS_CLUS", 4) // 2)
    mask = remove_small_objects(mask, min_size)
    
    # Label connected components
    labels, num = ndi.label(mask)
    
    print(f"[info] Found {num} particle clusters (min size: {min_size} pixels)")
    
    # Extract cluster information
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
    
    # Backup: peak detection if no clusters found
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
    MATLAB-style particle detection identical to partident.m
    
    This method tries multiple thresholds and finds individual pixels
    by removing large connected components at each threshold level.
    
    Parameters:
    -----------
    rgb_image : np.ndarray
        RGB image (H, W, 3)
    args : dict
        Configuration parameters including:
        - PARTICLE_LOWER_BOUND: Lower threshold bound
        - PARTICLE_UPPER_BOUND: Upper threshold bound  
        - NHOOD_SIZE: Neighborhood size for edge exclusion
    
    Returns:
    --------
    Tuple[np.ndarray, List[dict]]
        (labeled image, particle information list)
    """
    # Parameters from MATLAB
    lower = args.get('PARTICLE_LOWER_BOUND', 0)
    upper = args.get('PARTICLE_UPPER_BOUND', 0.5)
    nhood = args.get('NHOOD_SIZE', 1)  # odd number
    
    # RGB to grayscale (sum of all channels)
    srgb = rgb_image.sum(axis=2)
    H, W = srgb.shape
    
    # Create mask to exclude edges (MATLAB's picture frame)
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

def apply_background_correction(cube, wvl, clusters, args, flatfield_ref, raw_cube):
    """
    Apply background correction to hyperspectral cube
    
    This function applies either global or local background correction
    following the MATLAB workflow: Raw → Background subtraction → Flatfield correction
    
    Parameters:
    -----------
    cube : np.ndarray
        Flatfield-corrected hyperspectral cube
    wvl : np.ndarray
        Wavelength array
    clusters : list
        List of detected particle clusters
    args : dict
        Configuration parameters
    flatfield_ref : np.ndarray
        White and dark references from flatfield correction
    raw_cube : np.ndarray
        Original raw cube before flatfield correction
    
    Returns:
    --------
    np.ndarray
        Background-corrected hyperspectral cube
    """
    bg_mode = args.get('BACKGROUND_MODE', 'global')
    
    if bg_mode == 'global':
        # MATLAB-style global background
        corrected = apply_global_background(cube, wvl, args, flatfield_ref, raw_cube)
    elif bg_mode == 'local':
        # Local background
        corrected = apply_local_background(cube, wvl, clusters, args, flatfield_ref, raw_cube)

    else:
        raise ValueError("[error] Wrong bg_mode")
    
    return corrected

def apply_global_background(cube, wvl, args, flatfield_ref, raw_cube):
    """
    Apply global background subtraction identical to MATLAB anfunc_lorentz_fit.m
    
    The MATLAB workflow is:
    1. Find darkest pixel positions using flatfield-corrected data
    2. Calculate background spectrum from raw data at those positions
    3. Subtract background from raw data, then apply flatfield correction
    
    Parameters:
    -----------
    cube : np.ndarray
        Flatfield-corrected cube (for finding dark pixels)
    wvl : np.ndarray
        Wavelength array
    args : dict
        Configuration with BACKGROUND_PERCENTILE
    flatfield_ref : np.ndarray
        Flatfield references
    raw_cube : np.ndarray
        Original raw cube
    
    Returns:
    --------
    np.ndarray
        Background-corrected cube
    """
    # Parameters
    backper = args.get('BACKGROUND_PERCENTILE', 0.1)  # 10%
    
    H, W, L = cube.shape
    
    # Step 1: Find darkest pixel locations using flatfield-corrected data
    # (MATLAB: imnorm=specim./stanim then find darkest pixels)
    sumnorm = cube.sum(axis=2)
    
    # Normalize (MATLAB style)
    sumnorm_min = sumnorm.min()
    sumnorm_range = sumnorm.max() - sumnorm_min
    if sumnorm_range > 0:
        sumnorm = (sumnorm - sumnorm_min) / sumnorm_range
    
    # Find darkest pixels
    smln = int(np.ceil(backper * H * W))  # 10% of total pixels
    
    # Sort all pixel values
    sorted_values = np.sort(sumnorm.ravel())
    
    # Cutoff value (upper bound of bottom 10%)
    if smln < len(sorted_values):
        nthsmlst = sorted_values[smln]
    else:
        nthsmlst = sorted_values[-1]
    
    # Find positions of darkest pixels
    dark_mask = sumnorm <= nthsmlst
    dark_indices = np.where(dark_mask)
    
    # Limit to exactly smln pixels
    n_dark = min(len(dark_indices[0]), smln)
    
    print(f"[debug] Using {n_dark} darkest pixels ({backper*100:.1f}% of {H*W} total pixels)")
    
    # Step 2: Calculate background spectrum from RAW data
    bkavg_raw = np.zeros(L)
    for i in range(n_dark):
        row = dark_indices[0][i]
        col = dark_indices[1][i]
        bkavg_raw += raw_cube[row, col, :]  # Use RAW spectrum!
    
    bkavg_raw = bkavg_raw / n_dark  # Average
    
    print(f"[debug] Background from raw data: [{bkavg_raw.min():.2f}, {bkavg_raw.max():.2f}]")
    
    # Step 3: Process each pixel following MATLAB order
    # MATLAB: specfin = (specim - bkgim) ./ stanim
    corrected = np.zeros_like(cube)
    
    for row in range(H):
        # Select row-specific flatfield
        if row < flatfield_ref.shape[0]:
            flatfield = (flatfield_ref[row, 0, :])
        else:
            flatfield = (flatfield_ref[-1, 0, :])
        
        denominator = np.where(flatfield > 0, flatfield, 1.0)
        
        for col in range(W):
            # Subtract background from raw spectrum
            signal_raw = raw_cube[row, col, :] - bkavg_raw
            
            # Apply flatfield correction
            corrected[row, col, :] = signal_raw / denominator
    
    # Ensure no negative values
    corrected = np.maximum(corrected, 0)
    
    print(f"[debug] After background correction: [{corrected.min():.3f}, {corrected.max():.3f}]")
    
    return corrected

def apply_local_background(cube, wvl, clusters, args, flatfield_ref, raw_cube):
    """
    Apply local background correction around each cluster
    
    This method finds the darkest region around each cluster and uses it as
    local background. Follows MATLAB order: Raw → Background subtraction → Flatfield
    
    Parameters:
    -----------
    cube : np.ndarray
        Flatfield-corrected cube (for finding background regions)
    wvl : np.ndarray
        Wavelength array
    clusters : list
        List of detected particle clusters
    args : dict
        Configuration with local background parameters
    flatfield_ref : np.ndarray
        Flatfield references
    raw_cube : np.ndarray
        Original raw cube
    
    Returns:
    --------
    np.ndarray
        Background-corrected cube
    """
    print(f"\n[debug] Applying local background correction")
    
    H, W, L = cube.shape
    corrected = np.zeros_like(cube)
    
    # Initialize with global background for non-cluster pixels
    corrected = apply_global_background(cube, wvl, args, flatfield_ref, raw_cube)
    
    # Integration size (3x3 region)
    int_size = 3
    int_var = (int_size - 1) // 2
    
    # Apply local background to each cluster
    for cluster in clusters:
        coords = cluster['coords']
        center = cluster['center']
        
        # Define local background search parameters
        search_radius = args.get('BACKGROUND_LOCAL_SEARCH_RADIUS', 20)
        percentile = args.get('BACKGROUND_LOCAL_PERCENTILE', 1)
        
        # Define search region around cluster center
        row_center, col_center = int(center[0]), int(center[1])
        row_min = max(0, row_center - search_radius)
        row_max = min(H, row_center + search_radius)
        col_min = max(0, col_center - search_radius)
        col_max = min(W, col_center + search_radius)
        
        # Find pixels with low intensity sum in search region (using flatfield-corrected data)
        search_region = cube[row_min:row_max, col_min:col_max, :]
        intensity_map = search_region.sum(axis=2)
        
        # Exclude 3x3 cluster region from background calculation
        mask = np.ones(intensity_map.shape, dtype=bool)
        
        # Exclude all pixels in 3x3 region from mask
        for m in range(-int_var, int_var + 1):
            for l in range(-int_var, int_var + 1):
                local_r = row_center + m - row_min
                local_c = col_center + l - col_min
                if 0 <= local_r < mask.shape[0] and 0 <= local_c < mask.shape[1]:
                    mask[local_r, local_c] = False
        
        # Select darkest pixels for background
        masked_intensity = intensity_map[mask]
        if len(masked_intensity) > 0:
            threshold = np.percentile(masked_intensity, percentile)
            dark_pixels = np.where((intensity_map <= threshold) & mask)
            
            if len(dark_pixels[0]) > 0:
                # Calculate background from RAW data
                background_raw = np.zeros(L)
                for i in range(len(dark_pixels[0])):
                    global_r = dark_pixels[0][i] + row_min
                    global_c = dark_pixels[1][i] + col_min
                    background_raw += raw_cube[global_r, global_c, :]  # Use RAW data!
                
                background_raw /= len(dark_pixels[0])
                
                print(f"  Cluster {cluster['label']}: Using {len(dark_pixels[0])} pixels for local background")
                print(f"    Raw background range: [{background_raw.min():.1f}, {background_raw.max():.1f}]")
                
                # Apply same processing to entire 3x3 region
                pixels_corrected = 0
                for m in range(-int_var, int_var + 1):
                    for l in range(-int_var, int_var + 1):
                        r = row_center + m
                        c = col_center + l
                        
                        # Check boundaries
                        if 0 <= r < H and 0 <= c < W:
                            # Subtract background from raw data
                            signal_raw = raw_cube[r, c, :] - background_raw
                            
                            # Apply row-specific flatfield
                            if r < flatfield_ref.shape[0]:
                                flatfield = (flatfield_ref[r, 0, :])
                            else:
                                flatfield = (flatfield_ref[-1, 0, :])
                            
                            denominator = np.where(flatfield > 0, flatfield, 1.0)
                            corrected[r, c, :] = np.maximum(signal_raw / denominator, 0)
                            pixels_corrected += 1
                
                print(f"    Applied to {pixels_corrected} pixels in 3x3 region")
                
            else:
                print(f"  Cluster {cluster['label']}: No suitable background pixels nearby")
        else:
            print(f"  Cluster {cluster['label']}: No valid pixels in search region")
    
    print(f"[debug] After local background correction: range [{corrected.min():.3f}, {corrected.max():.3f}]")
    
    return corrected

def save_debug_image(args, img, name, cmap='hot'):
    """
    Save debug image to output directory
    
    Parameters:
    -----------
    args : dict
        Configuration with OUTPUT_DIR and SAMPLE_NAME
    img : np.ndarray
        Image data to save
    name : str
        Image name (without extension)
    cmap : str
        Matplotlib colormap name
    """
    out_dir = Path(args['OUTPUT_DIR']) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if img.ndim == 3:  # RGB image
        ax.imshow(img, origin='lower')
        ax.set_title(f"{name} (RGB)")
    else:  # Grayscale image
        im = ax.imshow(img, cmap=cmap, origin='lower')
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{name} (range: [{img.min():.2f}, {img.max():.2f}])")
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{args['SAMPLE_NAME']}_{name}.png", dpi=150)
    plt.close()

def save_coordinate_grid_image(args, max_map):
    """
    Save debug image with coordinate grid for manual coordinate selection
    
    This function creates a reference image with coordinate grids that helps
    users manually select particle coordinates for analysis.
    
    Parameters:
    -----------
    args : dict
        Configuration arguments with OUTPUT_DIR and SAMPLE_NAME
    max_map : np.ndarray
        Background-corrected maximum intensity map for better visibility
    """
    out_dir = Path(args['OUTPUT_DIR']) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Display the max intensity map
    im = ax.imshow(max_map, cmap='hot', origin='lower')
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # Get image dimensions
    H, W = max_map.shape
    
    # Set up major and minor grid lines
    # Major grid every 10 pixels
    major_interval = 10
    major_x = np.arange(0, W, major_interval)
    major_y = np.arange(0, H, major_interval)
    
    # Minor grid every 1 pixel
    minor_interval = 1
    minor_x = np.arange(0, W, minor_interval)
    minor_y = np.arange(0, H, minor_interval)
    
    # Draw grid lines
    # Major grid lines (thicker, more visible)
    for x in major_x:
        ax.axvline(x, color='white', linewidth=0.8, alpha=0.5)
    for y in major_y:
        ax.axhline(y, color='white', linewidth=0.8, alpha=0.5)
    
    # Minor grid lines (thinner, less visible)
    for x in minor_x:
        if x % major_interval != 0:  # Skip major grid positions
            ax.axvline(x, color='white', linewidth=0.4, alpha=0.3, linestyle='--')
    for y in minor_y:
        if y % major_interval != 0:
            ax.axhline(y, color='white', linewidth=0.4, alpha=0.3, linestyle='--')
    
    # Set tick labels (show every 10 pixels)
    ax.set_xticks(major_x)
    ax.set_yticks(major_y)
    ax.set_xticklabels(major_x)
    ax.set_yticklabels(major_y)
    
    # Add minor ticks without labels
    ax.set_xticks(minor_x, minor=True)
    ax.set_yticks(minor_y, minor=True)
    
    # Styling for better visibility
    ax.tick_params(axis='both', which='major', labelsize=10, color='white', labelcolor='yellow')
    ax.tick_params(axis='both', which='minor', size=3)
    
    # Set axis limits to show full image
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    
    # Labels and title
    ax.set_xlabel('X (column)', fontsize=12, color='yellow', fontweight='bold')
    ax.set_ylabel('Y (row)', fontsize=12, color='yellow', fontweight='bold')
    ax.set_title(f'{args["SAMPLE_NAME"]} - Coordinate Grid (Background Corrected)', 
                fontsize=14, pad=10)
    
    # Add helper text for coordinate format
    help_text = "Grid: Major lines every 10 pixels, minor lines every 5 pixels\n"
    help_text += "Coordinates format: (row, col)"
    ax.text(0.02, 0.98, help_text, 
           transform=ax.transAxes, 
           verticalalignment='top',
           fontsize=10,
           color='yellow',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    output_path = out_dir / f"{args['SAMPLE_NAME']}_coordinate_grid.png"
    plt.savefig(output_path, dpi=150, facecolor='black')
    plt.close()
    
    print(f"[info] Saved coordinate grid image: {output_path}")

def save_debug_dfs_detection(args, max_map, labels, clusters):
    """
    Save debug images showing particle detection results
    
    Parameters:
    -----------
    args : dict
        Configuration with detection style and output settings
    max_map : np.ndarray
        Maximum intensity map
    labels : np.ndarray
        Labeled image from particle detection
    clusters : list
        List of detected cluster information
    """
    out_dir = Path(args['OUTPUT_DIR']) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Choose visualization based on detection style
    detection_style = args.get('PARTICLE_DETECTION_STYLE', 'python')
    
    if detection_style == 'python':
        # Python style: detailed threshold process visualization
        save_debug_python_detection(args, max_map, labels, clusters, out_dir)
    else:
        # MATLAB style: simple result visualization
        save_debug_matlab_detection(args, max_map, labels, clusters, out_dir)


def save_debug_python_detection(args, max_map, labels, clusters, out_dir):
    """
    Create detailed visualization for Python-style particle detection process
    
    Shows the complete threshold-based detection pipeline including:
    - Original max intensity map
    - Normalized map
    - Histogram with thresholds
    - Binary mask after threshold
    - Result after morphological operations
    - Final labeled clusters
    
    Parameters:
    -----------
    args : dict
        Configuration parameters
    max_map : np.ndarray
        Maximum intensity map
    labels : np.ndarray
        Final labeled image
    clusters : list
        Detected cluster information
    out_dir : Path
        Output directory for debug images
    """
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
    
    # 3. Histogram with threshold visualization
    threshold = args.get('DFS_INTENSITY_THRESHOLD', 0.1)
    try:
        from skimage.filters import threshold_otsu
        otsu_val = threshold_otsu(normalized)
        actual_threshold = min(threshold, otsu_val * 0.8)
        
        # Create histogram with threshold lines
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
    
    # 6. Final labeled clusters with annotations
    im6 = axes[5].imshow(labels, cmap='tab20', origin='lower')
    axes[5].set_title(f'6. Final Clusters (n={len(clusters)})')
    
    # Mark cluster centers and information
    for cluster in clusters:
        center = cluster['center']
        axes[5].plot(center[1], center[0], 'w+', markersize=10, markeredgewidth=2)
        axes[5].text(center[1]+2, center[0]+2, f"{cluster['label']}\n{cluster['size']}px", 
                    color='white', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Add grid to all subplots for better readability
    for ax in axes:
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f'Python-style Particle Detection Debug - {args["SAMPLE_NAME"]}', fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir / f"{args['SAMPLE_NAME']}_python_detection_debug.png", dpi=150)
    plt.close()

def save_debug_matlab_detection(args, max_map, labels, clusters, out_dir):
    """
    Create simple visualization for MATLAB-style particle detection results
    
    Shows only the input and output since MATLAB-style detection is complex
    multi-threshold process that's difficult to visualize step-by-step.
    
    Parameters:
    -----------
    args : dict
        Configuration parameters
    max_map : np.ndarray
        Maximum intensity map
    labels : np.ndarray
        Final labeled image
    clusters : list
        Detected cluster information  
    out_dir : Path
        Output directory
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 1. Original max intensity map
    im1 = ax1.imshow(max_map, cmap='hot', origin='lower')
    ax1.set_title('Max Intensity Map')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Detected particles
    ax2.imshow(labels, cmap='tab20', origin='lower')
    ax2.set_title(f'MATLAB-style Detection (n={len(clusters)} particles)')
    
    # Mark detected particles
    for cluster in clusters:
        center = cluster['center']
        ax2.plot(center[1], center[0], 'w+', markersize=8, markeredgewidth=1)
    
    # Add grid for both subplots
    for ax in [ax1, ax2]:
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_dir / f"{args['SAMPLE_NAME']}_matlab_detection.png", dpi=150)
    plt.close()

def create_manual_clusters(max_map, manual_coords, args):
    """
    Create particle clusters from manually specified coordinates
    
    This function takes user-specified coordinates and automatically creates
    3x3 clusters around each point for consistent analysis.
    
    Parameters:
    -----------
    max_map : np.ndarray
        Maximum intensity map for reference and intensity calculation
    manual_coords : List[Tuple[int, int]]
        List of (row, col) coordinates specified by user
    args : dict
        Configuration arguments
    
    Returns:
    --------
    Tuple[np.ndarray, List[dict]]
        (label map with 3x3 regions, cluster information list)
    """
    H, W = max_map.shape
    labels = np.zeros((H, W), dtype=int)
    clusters = []
    
    print(f"[info] Creating 3x3 clusters from {len(manual_coords)} manual coordinates")
    
    for i, (row, col) in enumerate(manual_coords):
        # Validate that center coordinates are within image bounds
        if not (0 <= row < H and 0 <= col < W):
            print(f"  [warning] Coordinate ({row}, {col}) out of bounds, skipping")
            continue
        
        label = i + 1
        
        # Create 3x3 region around the specified coordinate
        coords = []
        for dr in [-1, 0, 1]:  # Row offsets: -1, 0, +1
            for dc in [-1, 0, 1]:  # Column offsets: -1, 0, +1
                r = row + dr
                c = col + dc
                # Only include pixels within image boundaries
                if 0 <= r < H and 0 <= c < W:
                    labels[r, c] = label
                    coords.append([r, c])
        
        coords = np.array(coords)
        
        # Calculate cluster properties from the max intensity map
        intensities = max_map[coords[:, 0], coords[:, 1]]
        
        # Create cluster information dictionary
        cluster = {
            'label': label,
            'coords': coords,
            'size': len(coords),
            'center': np.array([row, col]),  # Keep original coordinate as center
            'max_intensity': intensities.max(),
            'mean_intensity': intensities.mean(),
            'manual': True  # Flag to indicate this was manually selected
        }
        
        clusters.append(cluster)
        
        print(f"  Manual particle {label}: center=({row}, {col}), "
              f"3x3 region with {len(coords)} pixels, "
              f"max_intensity={intensities.max():.2f}")
    
    print(f"[info] Created {len(clusters)} manual 3x3 clusters")
    
    return labels, clusters