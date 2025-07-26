import os
import numpy as np
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any, Union

from . import dataset_util as du

class Dataset(object):
    """Load hyperspectral TDMS data and preprocess for DFS analysis."""
    
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.sample_name = args['SAMPLE_NAME']
        self.cube = None   # ndarray (H,W,λ)
        self.wvl = None    # ndarray (λ,)
        self.max_map = None  # DFS max intensity map
        self.labels = None
        self.clusters = None
        self.white_ref = None  # White reference for background correction
        self.dark_ref = None   # Dark reference for background correction
        self.raw_cube = None
        
    def run_dataset(self):
        """
        Dataset processing pipeline:
        1. Load TDMS -> cube
        2. Crop wavelength
        3. Flatfield correction
        4. Particle detection
        5. Background removal
        """
        # 1. Load TDMS
        self.load_cube()
        
        # 2. Crop wavelength
        self.crop_wavelength()
        
        # 3. Flatfield correction
        self.flatfield()
        
        # 4. Create max intensity map for detection
        self.create_dfs_map()
        
        # 5. Detect particles/clusters
        self.detect_particles_dfs()
        
        # 6. Apply background correction
        self.apply_background()
    
    def load_cube(self):
        """Load TDMS file and convert to cube"""
        sample_name = self.args['SAMPLE_NAME'] + ".tdms"
        path = os.path.join(self.args['DATA_DIR'], sample_name)
        print(f"\n[debug] Loading TDMS file: {path}")
        
        self.cube, self.wvl = du.tdms_to_cube(path)
        
        print(f"[debug] Raw cube loaded:")
        print(f"  - Shape: {self.cube.shape} (H={self.cube.shape[0]}, W={self.cube.shape[1]}, λ={self.cube.shape[2]})")
        print(f"  - Data range: [{self.cube.min():.2f}, {self.cube.max():.2f}]")
        print(f"  - Mean value: {self.cube.mean():.2f}")
        print(f"  - Wavelengths: {self.wvl.min():.1f}-{self.wvl.max():.1f} nm, {len(self.wvl)} points")
        
        # Save a debug image of raw data
        du.save_debug_image(self.args, self.cube.sum(axis=2), "raw_sum", cmap='hot')
    
    def crop_wavelength(self):
        """Crop to specified wavelength range"""
        print(f"\n[debug] Cropping wavelength range...")
        wl_range = self.args['CROP_RANGE_NM']
        self.cube, self.wvl = du.crop_wavelength(self.cube, self.wvl, wl_range)
        
        print(f"[debug] After cropping:")
        print(f"  - Shape: {self.cube.shape}")
        print(f"  - Wavelengths: {self.wvl.min():.1f}-{self.wvl.max():.1f} nm")
        
        du.save_debug_image(self.args, self.cube.sum(axis=2), "cropped_sum", cmap='hot')
    
    def flatfield(self):
        """Apply flatfield correction"""
        print(f"\n[debug] Applying flatfield correction...")

        self.raw_cube = self.cube.copy()
        
        w = os.path.join(self.args['DATA_DIR'], self.args['WHITE_FILE'])
        d = os.path.join(self.args['DATA_DIR'], self.args['DARK_FILE'])
        
        print(f"  - White reference: {w}")
        print(f"  - Dark reference: {d}")
        
        # Flatfield correction and save references
        self.cube, self.white_ref, self.dark_ref = du.flatfield_correct(self.cube, self.wvl, w, d)
        
        print(f"[debug] After flatfield:")
        print(f"  - Data range: [{self.cube.min():.2f}, {self.cube.max():.2f}]")
        print(f"  - Mean value: {self.cube.mean():.2f}")
        
        du.save_debug_image(self.args, self.cube.sum(axis=2), "flatfield_sum", cmap='hot')
    
    def create_dfs_map(self):
        """Create DFS-specific max intensity map"""
        print("\n[Step] Creating DFS max intensity map...")
        
        wl_range = self.args.get('DFS_WL_RANGE', (500, 800))
        self.max_map = du.create_dfs_max_intensity_map(self.cube, self.wvl, wl_range)
        
        # Debug 이미지 저장
        du.save_debug_image(self.args, self.max_map, "dfs_max_map", cmap='hot')
    
    def detect_particles_dfs(self):
        """Detect particles using DFS-specific method"""
        print("\n[Step] Detecting particles from DFS data...")
        
        self.labels, self.clusters = du.detect_dfs_particles(self.max_map, self.args)
        
        # Debug 이미지 저장
        du.save_debug_dfs_detection(self.args, self.max_map, self.labels, self.clusters)
    
    def apply_background(self):
        """Apply background correction"""
        print("\n[Step] Applying background correction...")
        
        self.cube = du.apply_background_correction(
            self.cube, self.wvl, self.clusters, self.args, 
            self.white_ref, self.dark_ref, self.raw_cube)
        
        # Update max map after background correction
        wl_range = self.args.get('DFS_WL_RANGE', (500, 800))
        self.max_map = du.create_dfs_max_intensity_map(self.cube, self.wvl, wl_range)
        du.save_debug_image(self.args, self.max_map, "dfs_max_map_bg_corrected", cmap='hot')