import os
import numpy as np
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any, Union

from . import dataset_util as du

class Dataset(object):
    """
    Hyperspectral TDMS data loader and preprocessor for DFS (Dark Field Scattering) analysis
    
    This class handles the complete preprocessing pipeline for hyperspectral imaging data:
    1. Loading TDMS files and converting to 3D cubes
    2. Wavelength cropping to region of interest  
    3. Flatfield correction using white/dark references
    4. Particle detection from maximum intensity maps
    5. Background correction (global or local)
    
    The class maintains all intermediate results and provides debug visualizations
    throughout the processing pipeline.
    """
    
    def __init__(self, args: Dict[str, Any]):
        """
        Initialize Dataset with configuration parameters
        
        Parameters:
        -----------
        args : Dict[str, Any]
            Configuration dictionary containing:
            - SAMPLE_NAME: Name of the TDMS file (without extension)
            - DATA_DIR: Directory containing TDMS files
            - WHITE_FILE, DARK_FILE: Reference file names
            - CROP_RANGE_NM: Wavelength range for cropping
            - Various detection and background correction parameters
        """
        self.args = args
        self.sample_name = args['SAMPLE_NAME']
        
        # Main data containers - initialized as None and populated during processing
        self.cube = None   # 3D hyperspectral cube (H,W,λ) after flatfield correction
        self.wvl = None    # 1D wavelength array (λ,)
        self.max_map = None  # 2D maximum intensity projection for DFS analysis
        self.labels = None   # 2D labeled image showing detected particle regions
        self.clusters = None # List of dictionaries containing cluster information
        
        # References and intermediate data for background correction
        self.white_ref = None  # White reference from flatfield correction
        self.dark_ref = None   # Dark reference from flatfield correction  
        self.raw_cube = None   # Original raw cube before any correction
        self.max_map_before_bg = None  # Max map before background correction (for visualization)
        
    def run_dataset(self):
        """
        Execute the complete dataset processing pipeline
        
        This method runs all preprocessing steps in the correct order:
        1. Load TDMS file and convert to hyperspectral cube
        2. Crop to specified wavelength range
        3. Apply flatfield correction using white/dark references
        4. Create maximum intensity projection map for particle detection
        5. Detect particle clusters using specified method
        6. Apply background correction (global or local)
        
        Each step includes debug output and intermediate result saving.
        """
        # Step 1: Load TDMS file into 3D cube format
        self.load_cube()
        
        # Step 2: Crop wavelength range to region of interest
        self.crop_wavelength()
        
        # Step 3: Apply flatfield correction to remove illumination variations
        self.flatfield()
        
        # Step 4: Create 2D maximum intensity map for particle detection
        self.create_dfs_map()
        
        # Step 5: Detect particle clusters from the intensity map
        self.detect_particles_dfs()
        
        # Step 6: Apply background subtraction for improved spectral quality
        self.apply_background()
    
    def load_cube(self):
        """
        Load TDMS file and convert to 3D hyperspectral cube
        
        This method:
        - Constructs the full file path from DATA_DIR and SAMPLE_NAME
        - Calls the TDMS conversion utility function
        - Prints diagnostic information about the loaded data
        - Saves a debug image showing the raw data sum
        """
        # Construct file path
        sample_name = self.args['SAMPLE_NAME'] + ".tdms"
        path = os.path.join(self.args['DATA_DIR'], sample_name)
        print(f"\n[debug] Loading TDMS file: {path}")
        
        # Convert TDMS to cube using utility function
        self.cube, self.wvl = du.tdms_to_cube(path)
        
        # Print diagnostic information
        print(f"[debug] Raw cube loaded:")
        print(f"  - Shape: {self.cube.shape} (H={self.cube.shape[0]}, W={self.cube.shape[1]}, λ={self.cube.shape[2]})")
        print(f"  - Data range: [{self.cube.min():.2f}, {self.cube.max():.2f}]")
        print(f"  - Mean value: {self.cube.mean():.2f}")
        print(f"  - Wavelengths: {self.wvl.min():.1f}-{self.wvl.max():.1f} nm, {len(self.wvl)} points")
        
        # Save debug visualization of raw data (sum across all wavelengths)
        du.save_debug_image(self.args, self.cube.sum(axis=2), "raw_sum", cmap='hot')
    
    def crop_wavelength(self):
        """
        Crop hyperspectral cube to specified wavelength range
        
        This method:
        - Uses the CROP_RANGE_NM parameter to define wavelength limits
        - Calls the cropping utility function
        - Updates both cube and wavelength arrays
        - Saves debug image of cropped data
        """
        print(f"\n[debug] Cropping wavelength range...")
        wl_range = self.args['CROP_RANGE_NM']
        self.cube, self.wvl = du.crop_wavelength(self.cube, self.wvl, wl_range)
        
        print(f"[debug] After cropping:")
        print(f"  - Shape: {self.cube.shape}")
        print(f"  - Wavelengths: {self.wvl.min():.1f}-{self.wvl.max():.1f} nm")
        
        # Save debug image showing cropped data
        du.save_debug_image(self.args, self.cube.sum(axis=2), "cropped_sum", cmap='hot')
    
    def flatfield(self):
        """
        Apply flatfield correction to remove illumination variations
        
        This method:
        - Saves a copy of raw data before correction
        - Constructs paths to white and dark reference files
        - Applies MATLAB-compatible flatfield correction
        - Stores white and dark references for later background correction
        - Saves debug image of corrected data
        """
        print(f"\n[debug] Applying flatfield correction...")

        # Save raw cube before any correction (needed for background correction)
        self.raw_cube = self.cube.copy()
        
        # Construct paths to reference files
        w = os.path.join(self.args['DATA_DIR'], self.args['WHITE_FILE'])
        d = os.path.join(self.args['DATA_DIR'], self.args['DARK_FILE'])
        
        print(f"  - White reference: {w}")
        print(f"  - Dark reference: {d}")
        
        # Apply flatfield correction and save references for background correction
        self.cube, self.white_ref, self.dark_ref = du.flatfield_correct(self.cube, self.wvl, w, d)
        
        print(f"[debug] After flatfield:")
        print(f"  - Data range: [{self.cube.min():.2f}, {self.cube.max():.2f}]")
        print(f"  - Mean value: {self.cube.mean():.2f}")
        
        # Save debug image showing flatfield-corrected data
        du.save_debug_image(self.args, self.cube.sum(axis=2), "flatfield_sum", cmap='hot')
    
    def create_dfs_map(self):
        """
        Create DFS-specific maximum intensity projection map
        
        This method:
        - Uses DFS_WL_RANGE parameter to define wavelength window
        - Creates 2D map by taking maximum intensity across wavelengths
        - This map is used for particle detection
        - Saves debug image of the intensity map
        """
        print("\n[Step] Creating DFS max intensity map...")
        
        # Get wavelength range for DFS analysis (default: 500-800 nm)
        wl_range = self.args.get('DFS_WL_RANGE', (500, 800))
        self.max_map = du.create_dfs_max_intensity_map(self.cube, self.wvl, wl_range)
        
        # Save debug image showing the maximum intensity projection
        du.save_debug_image(self.args, self.max_map, "dfs_max_map", cmap='hot')
    
    def detect_particles_dfs(self):
        """
        Detect particle clusters using DFS-specific methods
        
        This method supports two modes:
        1. Manual mode (USE_MANUAL_COORDS=True): Uses manually specified coordinates
        2. Automatic mode: Uses either Python or MATLAB-style detection algorithms
        
        Results are stored in self.labels (2D labeled image) and 
        self.clusters (list of cluster information dictionaries).
        """

        if self.args.get("USE_MANUAL_COORDS", False):
            # Manual coordinate mode: create 3x3 clusters around specified points
            print("\n[Step] Using manual coordinates for particles...")
            self.labels, self.clusters = du.create_manual_clusters(self.max_map, self.args["MANUAL_COORDS"], self.args)

        else:
            # Automatic detection mode: use specified detection algorithm
            print("\n[Step] Detecting particles from DFS data...")
            self.labels, self.clusters = du.detect_dfs_particles(self.max_map, self.args)
        
        # Save debug images showing detection results
        du.save_debug_dfs_detection(self.args, self.max_map, self.labels, self.clusters)
    
    def apply_background(self):
        """
        Apply background correction to improve spectral quality
        
        This method:
        - Saves the current max map before background correction (for visualization)
        - Applies either global or local background correction based on configuration
        - Updates the cube with background-corrected data
        - Creates new max intensity map after background correction
        - Saves debug images and coordinate grid for manual inspection
        
        The background correction follows MATLAB workflow:
        Raw data → Background subtraction → Flatfield correction
        """
        print("\n[Step] Applying background correction...")

        # Save max map before background correction for comparison
        self.max_map_before_bg = self.max_map.copy()
        
        # Apply background correction using stored references and raw data
        self.cube = du.apply_background_correction(self.cube, self.wvl, self.clusters, self.args, 
                                                   self.white_ref, self.dark_ref, self.raw_cube)
        
        # Update max intensity map with background-corrected data
        wl_range = self.args.get('DFS_WL_RANGE', (500, 800))
        self.max_map = du.create_dfs_max_intensity_map(self.cube, self.wvl, wl_range)
        du.save_debug_image(self.args, self.max_map, "dfs_max_map_bg_corrected", cmap='hot')

        # Save coordinate grid image for manual coordinate selection reference
        du.save_coordinate_grid_image(self.args, self.max_map)