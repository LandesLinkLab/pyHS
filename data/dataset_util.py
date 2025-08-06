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
        (corrected_cube, white_ref, dark_ref) - references saved for background correction
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

def apply_background_correction(cube, wvl, clusters, args, white_ref, dark_ref, raw_cube):
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
    white_ref, dark_ref : np.ndarray
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
        corrected = apply_global_background(cube, wvl, args, white_ref, dark_ref, raw_cube)
    else:
        # Local background
        corrected = apply_local_background(cube, wvl, clusters, args, white_ref, dark_ref, raw_cube)
    
    return corrected

def apply_global_background(cube, wvl, args, white_ref, dark_ref, raw_cube):
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
    white_ref, dark_ref : np.ndarray
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
        if row < white_ref.shape[0]:
            flatfield = (white_ref[row, 0, :] - dark_ref[row, 0, :])
        else:
            flatfield = (white_ref[-1, 0, :] - dark_ref[-1, 0, :])
        
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

def apply_local_background(cube, wvl, clusters, args, white_ref, dark_ref, raw_cube):
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
    white_ref, dark_ref : np.ndarray
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
    corrected = apply_global_background(cube, wvl, args, white_ref, dark_ref, raw_cube)
    
    # Integration size (3x3 region)
    int_size = 3
    int_var = (int_size - 1) // 2
    
    # Apply local background to each cluster
    for cluster in clusters:
        coords = cluster['coords']
        center = cluster['center']
        
        # Define local background search parameters
        search_radius = args.get('BACKGROUND_LOCAL_SEARCH_RADIUS', 20)
        percentile = args.get('BACKGROUND_LOCAL_PERCENTILE',