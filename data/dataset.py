import os
import numpy as np
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any, Union
from nptdms import TdmsFile

import data.dataset_util as du

class Dataset(object):
    """
    Unified hyperspectral data loader and preprocessor
    
    Supports two analysis modes:
    - 'dfs': Dark Field Scattering spatial analysis (H × W × λ cube)
    - 'echem': Electrochemical time-series analysis (Time × λ array)
    
    The mode is determined by args['ANALYSIS_MODE']
    """
    
    def __init__(self, args: Dict[str, Any]):
        """
        Initialize Dataset with configuration and mode detection
        
        Parameters:
        -----------
        args : Dict[str, Any]
            Configuration dictionary containing:
            - ANALYSIS_MODE: 'dfs' or 'echem'
            - For DFS: SAMPLE_NAME, DFS_WL_RANGE, etc.
            - For EChem: ECHEM_SAMPLE_NAME, ECHEM_CHI_FILE, etc.
        """
        self.args = args
        self.mode = args.get('ANALYSIS_MODE', 'dfs')
        
        # Common attributes
        self.cube = None   # DFS: (H,W,λ), EChem: (Time,λ) or None
        self.wvl = None
        self.white_ref = None
        self.dark_ref = None
        self.raw_cube = None
        
        # Mode-specific initialization
        if self.mode == 'dfs':
            self._init_dfs_mode()
        elif self.mode == 'echem':
            self._init_echem_mode()
        else:
            raise ValueError(f"Unknown ANALYSIS_MODE: {self.mode}. Use 'dfs' or 'echem'")
        
        print(f"[info] Dataset initialized in {self.mode.upper()} mode")
    
    def _init_dfs_mode(self):
        """Initialize DFS-specific attributes"""
        self.sample_name = self.args['SAMPLE_NAME']
        
        # DFS-specific data structures
        self.max_map = None
        self.labels = None
        self.clusters = None
        self.max_map_before_bg = None
        
        print(f"[info] DFS mode - Sample: {self.sample_name}")
    
    def _init_echem_mode(self):
        """Initialize EChem-specific attributes"""
        self.sample_name = self.args['ECHEM_SAMPLE_NAME']
        
        # EChem uses 2D array instead of 3D cube
        self.spectra = None      # (Time × λ) - will reference self.cube
        self.spec_times = None   # Time stamps for spectra
        self.wavelengths = None  # Alias for self.wvl
        
        # Potentiostat data
        self.chi_data = None
        self.voltages = None
        self.background = None
        
        # Processing parameters
        self.lowercut = self.args.get('ECHEM_LOWERCUT', 140)
        self.uppercut = self.args.get('ECHEM_UPPERCUT', 260)
        
        # Setup EChem output directory
        self.echem_output_dir = Path(self.args['OUTPUT_DIR']) / 'echem'
        self.echem_output_dir.mkdir(parents=True, exist_ok=True)
        self.args['ECHEM_OUTPUT_DIR'] = str(self.echem_output_dir)
        
        print(f"[info] EChem mode - Sample: {self.sample_name}")
        print(f"[info] EChem output: {self.echem_output_dir}")
    
    def run_dataset(self):
        """Execute preprocessing pipeline based on mode"""
        if self.mode == 'dfs':
            self._run_dfs_pipeline()
        elif self.mode == 'echem':
            self._run_echem_pipeline()
    
    # ========================================================================
    # DFS PIPELINE
    # ========================================================================
    
    def _run_dfs_pipeline(self):
        """Execute DFS preprocessing pipeline"""
        self.load_cube()
        self.crop_wavelength()
        self.flatfield()
        self.create_dfs_map()
        self.detect_particles_dfs()
        self.apply_background()
    
    def load_cube(self):
        """Load TDMS file and convert to 3D hyperspectral cube (DFS mode)"""
        sample_name = self.args['SAMPLE_NAME'] + ".tdms"
        path = os.path.join(self.args['DATA_DIR'], sample_name)
        print(f"\n[debug] Loading DFS TDMS file: {path}")
        
        self.cube, self.wvl = du.tdms_to_cube(path)
        
        print(f"[debug] Raw cube loaded:")
        print(f"  - Shape: {self.cube.shape} (H={self.cube.shape[0]}, W={self.cube.shape[1]}, λ={self.cube.shape[2]})")
        print(f"  - Data range: [{self.cube.min():.2f}, {self.cube.max():.2f}]")
        print(f"  - Mean value: {self.cube.mean():.2f}")
        print(f"  - Wavelengths: {self.wvl.min():.1f}-{self.wvl.max():.1f} nm, {len(self.wvl)} points")
        
        du.save_debug_image(self.args, self.cube.sum(axis=2), "raw_sum", cmap='hot')
    
    def crop_wavelength(self):
        """Crop wavelength range (DFS mode)"""
        print(f"\n[debug] Cropping wavelength range...")
        wl_range = self.args['CROP_RANGE_NM']
        self.cube, self.wvl = du.crop_wavelength(self.cube, self.wvl, wl_range)
        
        print(f"[debug] After cropping:")
        print(f"  - Shape: {self.cube.shape}")
        print(f"  - Wavelengths: {self.wvl.min():.1f}-{self.wvl.max():.1f} nm")
        
        du.save_debug_image(self.args, self.cube.sum(axis=2), "cropped_sum", cmap='hot')
    
    def flatfield(self):
        """Apply flatfield correction (DFS mode)"""
        print(f"\n[debug] Applying flatfield correction...")
        
        # Save raw cube before any correction
        self.raw_cube = self.cube.copy()
        
        # Construct paths to reference files
        w = os.path.join(self.args['DATA_DIR'], self.args['WHITE_FILE'])
        d = os.path.join(self.args['DATA_DIR'], self.args['DARK_FILE'])
        
        print(f"  - White reference: {w}")
        print(f"  - Dark reference: {d}")
        
        # Apply flatfield correction and save references
        self.cube, self.white_ref, self.dark_ref = du.flatfield_correct(
            self.cube, self.wvl, w, d
        )
        
        print(f"[debug] After flatfield:")
        print(f"  - Data range: [{self.cube.min():.2f}, {self.cube.max():.2f}]")
        print(f"  - Mean value: {self.cube.mean():.2f}")
        
        du.save_debug_image(self.args, self.cube.sum(axis=2), "flatfield_sum", cmap='hot')
    
    def create_dfs_map(self):
        """Create DFS maximum intensity projection map"""
        print("\n[Step] Creating DFS max intensity map...")
        
        wl_range = self.args.get('DFS_WL_RANGE', (500, 800))
        self.max_map = du.create_dfs_max_intensity_map(self.cube, self.wvl, wl_range)
        
        du.save_debug_image(self.args, self.max_map, "dfs_max_map", cmap='hot')
    
    def detect_particles_dfs(self):
        """Detect particle clusters (DFS mode)"""
        if self.args.get("USE_MANUAL_COORDS", False):
            print("\n[Step] Using manual coordinates for particles...")
            self.labels, self.clusters = du.create_manual_clusters(
                self.max_map, self.args["MANUAL_COORDS"], self.args
            )
        else:
            print("\n[Step] Detecting particles from DFS data...")
            self.labels, self.clusters = du.detect_dfs_particles(
                self.max_map, self.args
            )
        
        du.save_debug_dfs_detection(self.args, self.max_map, self.labels, self.clusters)
    
    def apply_background(self):
        """Apply background correction (DFS mode)"""
        print("\n[Step] Applying background correction...")
        
        # Save max map before background correction
        self.max_map_before_bg = self.max_map.copy()
        
        # Apply background correction
        self.cube = du.apply_background_correction(
            self.cube, self.wvl, self.clusters, self.args,
            self.white_ref, self.dark_ref, self.raw_cube
        )
        
        # Update max intensity map with background-corrected data
        wl_range = self.args.get('DFS_WL_RANGE', (500, 800))
        self.max_map = du.create_dfs_max_intensity_map(self.cube, self.wvl, wl_range)
        du.save_debug_image(self.args, self.max_map, "dfs_max_map_bg_corrected", cmap='hot')
        
        # Save coordinate grid for manual inspection
        du.save_coordinate_grid_image(self.args, self.max_map)
    
    # ========================================================================
    # ECHEM PIPELINE
    # ========================================================================
    
    def _run_echem_pipeline(self):
        """Execute EChem preprocessing pipeline"""
        self.load_spectral_tdms()
        self.load_chi_data()
        self.match_voltage_to_spectra()
        self.apply_flatfield_echem()
        self.crop_wavelength_echem()
        self.save_debug_plots_echem()
    
    def load_spectral_tdms(self):
        """Load TDMS file containing time-series spectral data (EChem mode)"""
        tdms_path = os.path.join(self.args['DATA_DIR'], f"{self.sample_name}.tdms")
        print(f"\n[debug] Loading EChem TDMS file: {tdms_path}")
        
        td = TdmsFile.read(tdms_path)
        
        # Extract wavelength array from Info group
        info_group = td['Info']
        if info_group is None:
            raise RuntimeError("No 'Info' group found in EChem TDMS file")
        
        wl_channel = None
        for ch in info_group.channels():
            if ch.name == 'wvlths':
                wl_channel = ch
                break
        
        if wl_channel is None:
            raise RuntimeError("No 'wvlths' channel found in Info group")
        
        self.wvl = wl_channel[:].astype(np.float32)
        self.wavelengths = self.wvl  # Alias for EChem compatibility
        
        # Extract background spectrum
        bg_channel = None
        for ch in info_group.channels():
            if 'bg' in ch.name.lower() or 'background' in ch.name.lower():
                bg_channel = ch
                break
        
        if bg_channel is None:
            print("[warning] No background channel found, will use zeros")
            self.background = np.zeros_like(self.wvl)
        else:
            bg_raw = bg_channel[:].astype(np.float32)
            wvgrp = td.properties.get('wvlth group', 1)
            if wvgrp == 2:
                self.background = bg_raw
            else:
                self.background = bg_raw[::2] + bg_raw[1::2]
        
        # Extract time stamps
        time_channel = None
        for ch in info_group.channels():
            if 'time' in ch.name.lower():
                time_channel = ch
                break
        
        if time_channel is None:
            print("[warning] No time channel found, will use indices")
            self.spec_times = None
        else:
            self.spec_times = time_channel[:].astype(np.float32)
        
        print(f"[info] Wavelength array: {self.wvl.min():.1f}-{self.wvl.max():.1f} nm, "
              f"{len(self.wvl)} points")
        
        # Extract spectra from Spectra group
        spectra_group = td['Spectra']
        if spectra_group is None:
            raise RuntimeError("No 'Spectra' group found in EChem TDMS file")
        
        spec_channels = list(spectra_group.channels())
        n_spectra = len(spec_channels)
        
        print(f"[info] Found {n_spectra} time-point spectra")
        
        # Check wavelength grouping
        wvgrp = td.properties.get('wvlth group', 1)
        
        # Initialize spectra array
        if wvgrp == 2:
            self.spectra = np.zeros((n_spectra, len(self.wvl)), dtype=np.float32)
        else:
            self.spectra = np.zeros((n_spectra, len(self.background)), dtype=np.float32)
        
        # Load each spectrum
        for i, ch in enumerate(spec_channels):
            spec_data = ch[:].astype(np.float32)
            
            if wvgrp == 2:
                self.spectra[i, :] = spec_data
            else:
                # Average adjacent wavelengths
                self.spectra[i, :] = spec_data[::2] + spec_data[1::2]
            
            if i % 50 == 0:
                print(f"  Loading spectrum {i}/{n_spectra}...")
        
        # Create time array if not found in file
        if self.spec_times is None:
            self.spec_times = np.arange(n_spectra, dtype=np.float32)
            print("[info] Created synthetic time array (indices)")
        
        # For EChem, cube is just the 2D spectra array
        self.cube = self.spectra
        
        print(f"[debug] Loaded spectra shape: {self.spectra.shape}")
        print(f"[debug] Time array: {self.spec_times.min():.2f} - {self.spec_times.max():.2f} s")
        print(f"[debug] Spectral data range: [{self.spectra.min():.2f}, {self.spectra.max():.2f}]")
    
    def load_chi_data(self):
        """Load and parse CHI potentiostat data (EChem mode)"""
        # Import here to avoid circular dependency
        from echem import echem_util as eu
        
        chi_path = os.path.join(self.args['DATA_DIR'], 
                               f"{self.args['ECHEM_CHI_FILE']}.txt")
        
        print(f"\n[debug] Loading CHI potentiostat file: {chi_path}")
        
        self.chi_data = eu.parse_chi_file(chi_path)
        
        print(f"[info] Detected technique: {self.chi_data['technique']}")
        print(f"[info] CHI data points: {len(self.chi_data['data'])}")
        
        if len(self.chi_data['data']) > 0:
            chi_times = self.chi_data['data'][:, 0]
            chi_voltages = self.chi_data['data'][:, 1]
            
            print(f"[info] CHI time range: {chi_times.min():.2f} - {chi_times.max():.2f} s")
            print(f"[info] Voltage range: {chi_voltages.min():.3f} - {chi_voltages.max():.3f} V")
    
    def match_voltage_to_spectra(self):
        """Synchronize spectra with voltage values (EChem mode)"""
        from echem import echem_util as eu
        
        print(f"\n[debug] Matching spectra times to voltage values...")
        
        if len(self.chi_data['data']) == 0:
            print("[warning] No CHI data available, using zeros for voltage")
            self.voltages = np.zeros(len(self.spec_times))
            return
        
        chi_times = self.chi_data['data'][:, 0]
        chi_voltages = self.chi_data['data'][:, 1]
        
        self.voltages = eu.match_spectra_to_voltage(
            self.spec_times, chi_times, chi_voltages
        )
        
        print(f"[info] Matched voltages: {self.voltages.min():.3f} - {self.voltages.max():.3f} V")
        print(f"[info] Matched {len(self.voltages)} spectra to voltage values")
    
    def apply_flatfield_echem(self):
        """Apply flatfield correction (EChem mode)"""
        print(f"\n[debug] Applying flatfield correction...")
        
        white_path = os.path.join(self.args['DATA_DIR'], self.args['WHITE_FILE'])
        dark_path = os.path.join(self.args['DATA_DIR'], self.args['DARK_FILE'])
        
        print(f"  White reference: {white_path}")
        print(f"  Dark reference: {dark_path}")
        
        # Load white reference
        td_white = TdmsFile.read(white_path)
        white_group = td_white['Spectra']
        white_channels = list(white_group.channels())
        
        white_spectra = [ch[:].astype(np.float32) for ch in white_channels]
        white_array = np.array(white_spectra)
        
        wvgrp = td_white.properties.get('wvlth group', 1)
        if wvgrp == 2:
            white_avg = white_array.mean(axis=0)
        else:
            white_combined = white_array[:, ::2] + white_array[:, 1::2]
            white_avg = white_combined.mean(axis=0)
        
        # Load dark reference
        td_dark = TdmsFile.read(dark_path)
        dark_group = td_dark['Spectra']
        dark_channels = list(dark_group.channels())
        
        dark_spectra = [ch[:].astype(np.float32) for ch in dark_channels]
        dark_array = np.array(dark_spectra)
        
        wvgrp_dark = td_dark.properties.get('wvlth group', 1)
        if wvgrp_dark == 2:
            dark_avg = dark_array.mean(axis=0)
        else:
            dark_combined = dark_array[:, ::2] + dark_array[:, 1::2]
            dark_avg = dark_combined.mean(axis=0)
        
        # Store references
        self.white_ref = white_avg
        self.dark_ref = dark_avg
        
        # Calculate flatfield
        flatfield = white_avg - dark_avg
        
        # Avoid division by zero
        flatfield = np.where(flatfield > 0, flatfield, 1.0)
        
        # Apply flatfield correction to each spectrum
        print(f"[debug] Applying flatfield to {self.spectra.shape[0]} spectra...")
        
        for i in range(self.spectra.shape[0]):
            # Subtract background, then normalize by flatfield
            self.spectra[i, :] = (self.spectra[i, :] - self.background) / flatfield
        
        # Clip extreme values
        self.spectra = np.clip(self.spectra, 0, 10)
        
        # Update cube reference
        self.cube = self.spectra
        
        print(f"[debug] After flatfield: range [{self.spectra.min():.3f}, {self.spectra.max():.3f}]")
    
    def crop_wavelength_echem(self):
        """Crop wavelength range (EChem mode)"""
        print(f"\n[debug] Cropping wavelength range...")
        print(f"  Lower cut: {self.lowercut} pixels")
        print(f"  Upper cut: {self.uppercut} pixels")
        
        crop_range = self.args.get('CROP_RANGE_NM', 
                                   (self.wavelengths.min(), self.wavelengths.max()))
        print(f"  Wavelength range: {crop_range[0]}-{crop_range[1]} nm")
        
        # Apply pixel trimming
        if self.uppercut > 0:
            self.wavelengths = self.wavelengths[self.lowercut:-self.uppercut]
            self.spectra = self.spectra[:, self.lowercut:-self.uppercut]
        else:
            self.wavelengths = self.wavelengths[self.lowercut:]
            self.spectra = self.spectra[:, self.lowercut:]
        
        # Additional wavelength range cropping
        wl_mask = (self.wavelengths >= crop_range[0]) & (self.wavelengths <= crop_range[1])
        self.wavelengths = self.wavelengths[wl_mask]
        self.spectra = self.spectra[:, wl_mask]
        
        # Update wvl alias and cube reference
        self.wvl = self.wavelengths
        self.cube = self.spectra
        
        print(f"[debug] After cropping:")
        print(f"  Wavelength range: {self.wavelengths.min():.1f} - {self.wavelengths.max():.1f} nm")
        print(f"  Spectra shape: {self.spectra.shape}")
    
    def save_debug_plots_echem(self):
        """Save EChem debug visualizations"""
        output_dir = self.echem_output_dir / "debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Spectral heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(self.spectra.T, aspect='auto', cmap='hot',
                      extent=[self.spec_times.min(), self.spec_times.max(),
                             self.wavelengths.min(), self.wavelengths.max()],
                      origin='lower')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Wavelength (nm)', fontsize=12)
        ax.set_title(f'{self.sample_name} - Spectral Evolution', fontsize=14)
        
        plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
        plt.tight_layout()
        plt.savefig(output_dir / f"{self.sample_name}_spectral_heatmap.png", dpi=150)
        plt.close()
        
        # Plot 2: Voltage trace
        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(self.spec_times, self.voltages, 'r-', linewidth=1.5)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Potential (V)', fontsize=12)
        ax.set_title(f'{self.sample_name} - Applied Potential', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{self.sample_name}_voltage_trace.png", dpi=150)
        plt.close()
        
        # Plot 3: Sample spectra at different times
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot every 10th spectrum or 5 evenly spaced spectra
        n_plot = min(5, self.spectra.shape[0])
        indices = np.linspace(0, self.spectra.shape[0]-1, n_plot, dtype=int)
        
        for idx in indices:
            label = f't={self.spec_times[idx]:.1f}s, V={self.voltages[idx]:.2f}V'
            ax.plot(self.wavelengths, self.spectra[idx, :], label=label, linewidth=2)
        
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Scattering (a.u.)', fontsize=12)
        ax.set_title(f'{self.sample_name} - Sample Spectra', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{self.sample_name}_sample_spectra.png", dpi=150)
        plt.close()
        
        print(f"[info] Saved debug plots to {output_dir}")