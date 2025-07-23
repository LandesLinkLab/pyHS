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
        self.representatives = None

    def run_dataset(self):
        self.load_cube()

        bg_mode = self.args.get('BACKGROUND_MODE', 'global')
        
        if bg_mode == 'global':
            # MATLAB 방식: flatfield 먼저, 그 다음 background
            self.flatfield()
            self.preprocess()  # 여기서 MATLAB global background 적용됨
            self.create_dfs_map()
            self.detect_particles_dfs()
            self.select_representatives()

        elif bg_mode == 'local':
            self.preprocess()
            self.flatfield()
            self.create_dfs_map()
            self.detect_particles_dfs()
            self.select_representatives()

            if self.representatives:
                print("\n[Step] Applying local background correction...")
                self.cube = du.apply_local_background(self.args, self.cube, self.clusters, self.representatives)

                wl_range = self.args.get('DFS_WL_RANGE', (500, 800))
                self.max_map = du.create_dfs_max_intensity_map(self.cube, self.wvl, wl_range)
                du.save_debug_image(self.args, self.max_map, "dfs_max_map_bg_corrected", cmap = 'hot')

                for rep in self.representatives:
                    rep['spectrum'] = self.cube[rep['row'], rep['col'], :]
                    peak_idx = np.argmax(rep['spectrum'])
                    rep['peak_wl'] = self.wvl[peak_idx]
                    rep['peak_intensity'] = rep['spectrum'][peak_idx]

        else:

            raise RuntimeError("[error] Not proper background mode: check bg_mode global or local")


    
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
        du.save_debug_image(self.args, self.cube.sum(axis=2), "raw_sum", cmap = 'hot')


    def flatfield(self):
        print(f"\n[debug] Applying flatfield correction...")
        
        w = os.path.join(self.args['DATA_DIR'], self.args['WHITE_FILE'])
        d = os.path.join(self.args['DATA_DIR'], self.args['DARK_FILE'])

        print(f"  - White reference: {w}")
        print(f"  - Dark reference: {d} (global mode)")
        
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
        
        du.save_debug_image(self.args, self.cube.sum(axis=2), "flatfield_sum", cmap = 'hot')
        
    def preprocess(self):
        print(f"\n[debug] Preprocessing...")
        print(f"  - Crop range: {self.args['CROP_RANGE_NM']} nm")
        print(f"  - Background percentile: {self.args['BACKGROUND_GLOBAL_PERCENTILE']}")
        
        self.cube, self.wvl = du.crop_and_bg(self.args, self.cube, self.wvl)
        
        print(f"[debug] After preprocessing:")
        print(f"  - Shape: {self.cube.shape}")
        print(f"  - Data range: [{self.cube.min():.2f}, {self.cube.max():.2f}]")
        print(f"  - Wavelengths: {self.wvl.min():.1f}-{self.wvl.max():.1f} nm")
        
        du.save_debug_image(self.args, self.cube.sum(axis=2), "preprocessed_sum", cmap = 'hot')
    
    # ---- DFS 전용 메서드들 ----
    def create_dfs_map(self):
        """Create DFS-specific max intensity map"""
        print("\n[Step] Creating DFS max intensity map...")
        
        # 500-800nm 범위에서 max intensity projection
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

    def select_representatives(self):

        print("\n[Step] Selecting representative spectra...")

        if self.args.get('USE_MANUAL_COORDS', False) and self.args.get('MANUAL_COORDS'):
            self.representatives = du.select_manual_representatives()
        else:
            self.representatives = du.select_representative_spectra(
                self.cube, self.wvl, self.clusters, self.args)
        
        # 호환성을 위해 centroids 형식으로도 저장
        if self.representatives:
            self.centroids = np.array([[r['row'], r['col']] 
                                       for r in self.representatives])
        else:
            self.centroids = np.array([])
