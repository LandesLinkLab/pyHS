import os
import timeit
import pickle as pkl
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any, Union
from . import dataset_util as du

class Dataset(object):
    
    def __init__(self, 
                args: Dict[str, Any]):
        
        self.args = args
        self.sample_name = args['SAMPLE_NAME']
        self.cube = None   # ndarray (H,W,λ)
        self.wvl = None    # ndarray (λ,)
        self.rgb = None
        self.labels = None
        self.centroids = None
        self.manual_coords = None
        
    def run_dataset(self):
        self.load_cube()
        self.preprocess()
        self.flatfield()
        
        # Check if manual coordinates are provided
        if self.args.get('USE_MANUAL_COORDS', False) and self.args.get('MANUAL_COORDS'):
            self.use_manual_particles()
        else:
            self.detect_particles()
    
    # ---------------- I/O & preprocessing ----------------
    def load_cube(self):
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
        self._save_debug_image(self.cube.sum(axis=2), "raw_sum")
        
    def flatfield(self):
        print(f"\n[debug] Applying flatfield correction...")
        
        w = os.path.join(self.args['DATA_DIR'], self.args['WHITE_FILE'])
        d = os.path.join(self.args['DATA_DIR'], self.args['DARK_FILE'])
        
        print(f"  - White reference: {w}")
        print(f"  - Dark reference: {d}")
        
        cube_before = self.cube.copy()
        self.cube = du.flatfield_correct(self.cube, self.wvl, w, d)
        
        print(f"[debug] After flatfield:")
        print(f"  - Data range: [{self.cube.min():.2f}, {self.cube.max():.2f}]")
        print(f"  - Mean value: {self.cube.mean():.2f}")
        
        # Check if flatfield made things worse
        if np.all(self.cube == 0) or np.isnan(self.cube).any():
            print("[warning] Flatfield correction resulted in zero or NaN values!")
            print("[warning] Skipping flatfield correction")
            self.cube = cube_before
        
        self._save_debug_image(self.cube.sum(axis=2), "flatfield_sum")
        
    def preprocess(self):
        print(f"\n[debug] Preprocessing...")
        print(f"  - Crop range: {self.args['CROP_RANGE_NM']} nm")
        print(f"  - Background percentile: {self.args['BACKGROUND_PERC']}")
        
        self.cube, self.wvl = du.crop_and_bg(self.cube, self.wvl, self.args)
        
        print(f"[debug] After preprocessing:")
        print(f"  - Shape: {self.cube.shape}")
        print(f"  - Data range: [{self.cube.min():.2f}, {self.cube.max():.2f}]")
        print(f"  - Wavelengths: {self.wvl.min():.1f}-{self.wvl.max():.1f} nm")
        
        self.rgb = du.cube_to_rgb(self.cube, self.wvl)
        
        self._save_debug_image(self.cube.sum(axis=2), "preprocessed_sum")
        self._save_debug_image(self.rgb, "rgb_composite")
        
    def detect_particles(self):
        print(f"\n[debug] Detecting particles automatically...")
        print(f"  - Threshold multiplier: {self.args['THRESH_HIGH']}")
        print(f"  - Min cluster size: {self.args['MIN_PIXELS_CLUS']}")
        
        self.labels, self.centroids = du.label_particles(self.rgb, self.args)
        
        print(f"[debug] Detection results:")
        print(f"  - Number of particles found: {len(self.centroids)}")
        if len(self.centroids) > 0:
            print(f"  - Particle locations: {self.centroids}")
        
        # Save debug image with labels
        self._save_debug_labels()
        
    def use_manual_particles(self):
        """Use manually specified coordinates instead of automatic detection"""
        print(f"\n[debug] Using manual coordinates: {len(self.args['MANUAL_COORDS'])} particles")
        
        # Convert manual coordinates to centroids
        self.manual_coords = np.array(self.args['MANUAL_COORDS'])
        self.centroids = self.manual_coords.astype(float)
        
        print(f"  - Manual coordinates: {self.manual_coords}")
        
        # Create fake labels for compatibility
        self.labels = np.zeros(self.cube.shape[:2], dtype=int)
        for i, (row, col) in enumerate(self.manual_coords):
            # Check if coordinates are valid
            if row < 0 or row >= self.cube.shape[0] or col < 0 or col >= self.cube.shape[1]:
                print(f"[warning] Coordinate ({row}, {col}) is out of bounds!")
                continue
                
            # Mark the integration area with the label
            int_size = self.args.get('INTEGRATION_SIZE', 3)
            half_size = (int_size - 1) // 2
            
            row_start = max(0, row - half_size)
            row_end = min(self.labels.shape[0], row + half_size + 1)
            col_start = max(0, col - half_size)
            col_end = min(self.labels.shape[1], col + half_size + 1)
            
            self.labels[row_start:row_end, col_start:col_end] = i + 1
            
            # Show spectrum at this location
            spec = self.cube[row, col, :]
            print(f"  - Particle {i} at ({row},{col}): max intensity = {spec.max():.2f}")
    
    def _save_debug_image(self, img, name):
        """Save debug images"""
        out_dir = Path(self.args['OUTPUT_DIR']) / "debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if img.ndim == 3:  # RGB
            ax.imshow(img, origin='lower')
            ax.set_title(f"{name} (RGB)")
        else:  # Grayscale
            im = ax.imshow(img, cmap='hot', origin='lower')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{name} (range: [{img.min():.2f}, {img.max():.2f}])")
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        plt.tight_layout()
        plt.savefig(out_dir / f"{self.sample_name}_{name}.png", dpi=150)
        plt.close()
    
    def _save_debug_labels(self):
        """Save debug image with detected particles"""
        out_dir = Path(self.args['OUTPUT_DIR']) / "debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show sum image with detected regions
        sum_img = self.cube.sum(axis=2)
        ax1.imshow(sum_img, cmap='gray', origin='lower')
        ax1.set_title('Sum Image')
        
        # Show labels
        ax2.imshow(self.labels, cmap='tab20', origin='lower')
        ax2.set_title(f'Detected Particles (n={len(self.centroids)})')
        
        # Mark centroids
        for i, (row, col) in enumerate(self.centroids):
            ax1.plot(col, row, 'r+', markersize=10, markeredgewidth=2)
            ax1.text(col+2, row+2, str(i), color='red', fontsize=8)
            ax2.plot(col, row, 'w+', markersize=10, markeredgewidth=2)
            ax2.text(col+2, row+2, str(i), color='white', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"{self.sample_name}_particle_detection.png", dpi=150)
        plt.close()