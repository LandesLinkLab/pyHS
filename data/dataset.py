import os
import timeit
import pickle as pkl
import numpy as np
from pathlib import Path
from matplotlib import cm
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
        self.representatives = None
        
    def run_dataset(self):
        self.load_cube()
        self.preprocess()
        if not self.args.get('SKIP_FLATFIELD', False):
            self.flatfield()
        self.create_dfs_map()
        self.detect_particles_dfs()
        self.select_representatives()
    
    # ---- 기본 I/O 메서드들 (누락된 부분) ----
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
        
        self._save_debug_image(self.cube.sum(axis=2), "preprocessed_sum")
    
    # ---- DFS 전용 메서드들 ----
    def create_dfs_map(self):
        """Create DFS-specific max intensity map"""
        print("\n[Step] Creating DFS max intensity map...")
        
        # 500-800nm 범위에서 max intensity projection
        wl_range = self.args.get('DFS_WL_RANGE', (500, 800))
        self.max_map = du.create_dfs_max_intensity_map(self.cube, self.wvl, wl_range)
        
        # Debug 이미지 저장
        self._save_debug_image(self.max_map, "dfs_max_map", cmap='hot')
    
    def detect_particles_dfs(self):
        """Detect particles using DFS-specific method"""
        print("\n[Step] Detecting particles from DFS data...")
        
        self.labels, self.clusters = du.detect_dfs_particles(self.max_map, self.args)
        
        # Debug 이미지 저장
        self._save_debug_dfs_detection()
    
    def select_representatives(self):
        """Select representative spectrum for each particle"""
        print("\n[Step] Selecting representative spectra...")
        
        self.representatives = du.select_representative_spectra(
            self.cube, self.wvl, self.clusters, self.args)
        
        # 호환성을 위해 centroids 형식으로도 저장
        if self.representatives:
            self.centroids = np.array([[r['row'], r['col']] 
                                       for r in self.representatives])
        else:
            self.centroids = np.array([])
    
    # ---- Debug 이미지 저장 메서드들 ----
    def _save_debug_image(self, img, name, cmap='hot'):
        """Save debug images"""
        if not self.args.get('DEBUG', False):
            return
            
        out_dir = Path(self.args['OUTPUT_DIR']) / "debug"
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
        plt.savefig(out_dir / f"{self.sample_name}_{name}.png", dpi=150)
        plt.close()
    
    def _save_debug_dfs_detection(self):
        """Save debug image for DFS particle detection"""
        if not self.args.get('DEBUG', False):
            return
            
        out_dir = Path(self.args['OUTPUT_DIR']) / "debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Max intensity map
        im1 = ax1.imshow(self.max_map, cmap='hot', origin='lower')
        ax1.set_title('Max Intensity Map (500-800nm)')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Binary mask (threshold 적용 후)
        threshold = self.args.get('DFS_INTENSITY_THRESHOLD', 0.1)
        normalized = (self.max_map - self.max_map.min()) / (self.max_map.max() - self.max_map.min())
        mask = normalized > threshold
        ax2.imshow(mask, cmap='gray', origin='lower')
        ax2.set_title(f'Binary Mask (threshold={threshold})')
        
        # 3. Labeled clusters
        ax3.imshow(self.labels, cmap='tab20', origin='lower')
        ax3.set_title(f'Detected Clusters (n={len(self.clusters)})')
        
        # 클러스터 중심 표시
        for cluster in self.clusters:
            center = cluster['center']
            ax3.plot(center[1], center[0], 'w+', markersize=10, markeredgewidth=2)
            ax3.text(center[1]+2, center[0]+2, str(cluster['label']), 
                    color='white', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(out_dir / f"{self.sample_name}_dfs_detection.png", dpi=150)
        plt.close()