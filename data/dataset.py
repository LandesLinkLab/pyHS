import os
import numpy as np
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any, Union
from nptdms import TdmsFile

import data.dataset_util as du
from echem import echem_util as eu

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
            self.white_ref, self.raw_cube
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
        
        # EChem TDMS 구조:
        # - Group 0: 시계열 스펙트럼 (N개 채널, 각 1340 points)
        # - Group 1: 시간 배열 (1개 채널, N points)
        # - Group 2: 배경 스펙트럼 (1개 채널, 1340 points)
        # - Group 3: 파장 배열 (1개 채널, 670 points)
        
        groups = list(td.groups())
        
        print(f"[debug] Found {len(groups)} groups in TDMS file")
        for i, group in enumerate(groups):
            n_channels = len(list(group.channels()))
            print(f"  Group {i}: {n_channels} channels")
        
        # ===== 1. 파장 배열 추출 (Group 3) =====
        if len(groups) < 4:
            raise RuntimeError(f"Expected 4 groups, found {len(groups)}")
        
        wl_group = groups[3]
        wl_channels = list(wl_group.channels())
        if len(wl_channels) != 1:
            raise RuntimeError(f"Expected 1 wavelength channel, found {len(wl_channels)}")
        
        self.wvl = wl_channels[0][:].astype(np.float32)
        self.wavelengths = self.wvl
        
        print(f"[info] Wavelength array: {self.wvl.min():.1f}-{self.wvl.max():.1f} nm, "
              f"{len(self.wvl)} points")
        
        # ===== 2. 배경 스펙트럼 추출 (Group 2) =====
        bg_group = groups[2]
        bg_channels = list(bg_group.channels())
        if len(bg_channels) != 1:
            raise RuntimeError(f"Expected 1 background channel, found {len(bg_channels)}")
        
        bg_raw = bg_channels[0][:].astype(np.float32)
        
        # Wavelength grouping 확인
        wvgrp = td.properties.get('wvlth group', 1)
        print(f"[debug] Wavelength group: {wvgrp}")
        
        if wvgrp == 2:
            self.background = bg_raw
        else:
            # Adjacent wavelength averaging
            self.background = bg_raw[::2] + bg_raw[1::2]
        
        print(f"[info] Background spectrum: {len(self.background)} points")
        
        # ===== 3. 시간 배열 추출 (Group 1) =====
        time_group = groups[1]
        time_channels = list(time_group.channels())
        if len(time_channels) != 1:
            raise RuntimeError(f"Expected 1 time channel, found {len(time_channels)}")
        
        self.spec_times = time_channels[0][:].astype(np.float32)
        
        print(f"[info] Time array: {self.spec_times.min():.2f} - {self.spec_times.max():.2f} s, "
              f"{len(self.spec_times)} points")
        
        # ===== 4. 스펙트럼 데이터 추출 (Group 0) =====
        spec_group = groups[0]
        spec_channels = list(spec_group.channels())
        n_spectra = len(spec_channels)
        
        print(f"[info] Found {n_spectra} time-point spectra")
        
        # 시간 배열과 스펙트럼 개수가 일치하는지 확인
        if len(self.spec_times) != n_spectra:
            print(f"[warning] Time array length ({len(self.spec_times)}) != "
                  f"number of spectra ({n_spectra})")
            print(f"[warning] Using spectrum count as reference")
            self.spec_times = np.arange(n_spectra, dtype=np.float32)
        
        # 스펙트럼 배열 초기화
        if wvgrp == 2:
            self.spectra = np.zeros((n_spectra, len(self.wvl)), dtype=np.float32)
        else:
            self.spectra = np.zeros((n_spectra, len(self.background)), dtype=np.float32)
        
        # 각 스펙트럼 로드
        for i, ch in enumerate(spec_channels):
            spec_data = ch[:].astype(np.float32)
            
            if wvgrp == 2:
                self.spectra[i, :] = spec_data
            else:
                # Average adjacent wavelengths
                self.spectra[i, :] = spec_data[::2] + spec_data[1::2]
            
            if i % 50 == 0:
                print(f"  Loading spectrum {i}/{n_spectra}...")
        
        # For EChem, cube is just the 2D spectra array
        self.cube = self.spectra
        
        print(f"[debug] Loaded spectra shape: {self.spectra.shape}")
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
       
        print(f"\n[debug] Matching spectra times to voltage values...")
        
        if len(self.chi_data['data']) == 0:
            print("[warning] No CHI data available, using zeros for voltage")
            self.voltages = np.zeros(len(self.spec_times))
            return
        
        # CHI data columns: [potential, current, charge, time]
        chi_voltages = self.chi_data['data'][:, 0]  # Column 0 = Potential
        chi_times = self.chi_data['data'][:, 3]     # Column 3 = Time
        
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
        """Save EChem visualizations to echem folder"""
        output_dir = self.echem_output_dir  # debug 폴더 대신 echem 폴더에 직접 저장
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장용 폴더 생성
        data_dir = self.echem_output_dir / "plot_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        num_peaks = self.args.get('NUM_PEAKS', 1)
        
        # ========================================
        # 1. 전체 스펙트럼 범위 heatmap (eV 단위) + voltage
        # ========================================
        print("[debug] Saving full-range spectral heatmap (eV)...")
        
        fig, (ax_heat, ax_volt) = plt.subplots(2, 1, figsize=(12, 8), 
                                                gridspec_kw={'height_ratios': [3, 1]})
        
        # nm → eV 변환
        energy = 1239.842 / self.wavelengths
        
        # 에너지 증가 순으로 정렬
        sort_idx = np.argsort(energy)
        energy_sorted = energy[sort_idx]
        spectra_sorted = self.spectra[:, sort_idx]
        
        # Heatmap
        im = ax_heat.imshow(spectra_sorted.T, aspect='auto', cmap='hot',
                            extent=[self.spec_times.min(), self.spec_times.max(),
                                   energy_sorted.min(), energy_sorted.max()],
                            origin='lower')
        
        ax_heat.set_ylabel('Energy (eV)', fontsize=12, fontweight='bold')
        ax_heat.set_title(f'{self.sample_name} - Spectral Evolution (Full Range)', 
                          fontsize=14, pad=20)
        ax_heat.tick_params(labelsize=10)
        ax_heat.set_xticklabels([])
        ax_heat.set_ylim(energy_sorted.min(), energy_sorted.max())
        
        # Colorbar 상단 배치
        cbar = plt.colorbar(im, ax=ax_heat, location='top', pad=0.02, fraction=0.05, shrink=0.2, anchor=(1.0, 0.0))
        cbar.set_label('Intensity (a.u.)', fontsize=10)
        
        # Voltage trace subplot
        ax_volt.plot(self.spec_times, self.voltages, 'r-', linewidth=1.5)
        ax_volt.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax_volt.set_ylabel('Voltage (V)', fontsize=12, fontweight='bold')
        ax_volt.tick_params(labelsize=10)
        ax_volt.grid(True, alpha=0.3)
        ax_volt.set_xlim(self.spec_times.min(), self.spec_times.max())
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{self.sample_name}_spectral_heatmap_full_eV.png", dpi=150)
        plt.close()
        
        # 데이터 저장 (full heatmap)
        np.savetxt(data_dir / f"{self.sample_name}_spectral_heatmap_full_eV.txt",
                   np.column_stack([self.spec_times, self.voltages]),
                   header=f"Time(s)\tVoltage(V)\nEnergy_range: {energy_sorted.min():.4f}-{energy_sorted.max():.4f} eV\nShape: {spectra_sorted.shape}",
                   delimiter='\t', fmt='%.6f')
        
        # ========================================
        # 2. 각 피크별 개별 heatmap (eV 단위) + voltage - 기본 버전
        # ========================================
        print(f"[debug] Saving individual peak heatmaps ({num_peaks} peaks)...")

        for peak_idx in range(num_peaks):
            # ✓ 모든 피크에 대해 full 범위 사용
            peak_wl_min = self.wavelengths.min()
            peak_wl_max = self.wavelengths.max()
            
            # 해당 범위 마스크
            mask = (self.wavelengths >= peak_wl_min) & (self.wavelengths <= peak_wl_max)
            
            if not np.any(mask):
                print(f"  Peak {peak_idx+1}: No data in range")
                continue
            
            # 추출
            wl_subset = self.wavelengths[mask]
            spectra_subset = self.spectra[:, mask]
            
            # nm → eV
            energy_subset = 1239.842 / wl_subset
            sort_idx_sub = np.argsort(energy_subset)
            energy_subset = energy_subset[sort_idx_sub]
            spectra_subset = spectra_subset[:, sort_idx_sub]
            
            # Plot: 2개 subplot (heatmap + voltage) - 파라미터는 나중에 추가
            fig, (ax_heat, ax_volt) = plt.subplots(2, 1, figsize=(12, 8),
                                                    gridspec_kw={'height_ratios': [3, 1]})
            
            # Heatmap
            im = ax_heat.imshow(spectra_subset.T, aspect='auto', cmap='hot',
                               extent=[self.spec_times.min(), self.spec_times.max(),
                                      energy_subset.min(), energy_subset.max()],
                               origin='lower')
            
            ax_heat.set_ylabel('Energy (eV)', fontsize=12, fontweight='bold')
            ax_heat.set_title(f'{self.sample_name} - Peak {peak_idx+1} Spectral Evolution',
                             fontsize=14, pad=20)
            ax_heat.tick_params(labelsize=10)
            ax_heat.set_xticklabels([])
            ax_heat.set_ylim(energy_subset.min(), energy_subset.max())
            
            # Colorbar 상단
            cbar = plt.colorbar(im, ax=ax_heat, location='top', pad=0.02, fraction=0.05, shrink=0.2, anchor=(1.0, 0.0))
            cbar.set_label('Intensity (a.u.)', fontsize=10)
            
            # Voltage trace subplot
            ax_volt.plot(self.spec_times, self.voltages, 'r-', linewidth=1.5)
            ax_volt.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            ax_volt.set_ylabel('Voltage (V)', fontsize=12, fontweight='bold')
            ax_volt.tick_params(labelsize=10)
            ax_volt.grid(True, alpha=0.3)
            ax_volt.set_xlim(self.spec_times.min(), self.spec_times.max())
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{self.sample_name}_spectral_heatmap_peak{peak_idx+1}_eV_basic.png", dpi=150)
            plt.close()
            
            # 데이터 저장
            np.savetxt(data_dir / f"{self.sample_name}_spectral_heatmap_peak{peak_idx+1}_eV_basic.txt",
                       np.column_stack([self.spec_times, self.voltages]),
                       header=f"Time(s)\tVoltage(V)\nPeak_{peak_idx+1}\nEnergy_range: {energy_subset.min():.4f}-{energy_subset.max():.4f} eV",
                       delimiter='\t', fmt='%.6f')
            
            print(f"  Saved peak {peak_idx+1} basic heatmap: "
                  f"{peak_wl_min:.0f}-{peak_wl_max:.0f} nm ({energy_subset.min():.2f}-{energy_subset.max():.2f} eV)")
        
        # ========================================
        # 3. Voltage trace
        # ========================================
        print("[debug] Saving voltage trace...")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(self.spec_times, self.voltages, 'r-', linewidth=1.5)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Potential (V)', fontsize=12)
        ax.set_title(f'{self.sample_name} - Applied Potential', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{self.sample_name}_voltage_trace.png", dpi=150)
        plt.close()
        
        # 데이터 저장
        np.savetxt(data_dir / f"{self.sample_name}_voltage_trace.txt",
                   np.column_stack([self.spec_times, self.voltages]),
                   header="Time(s)\tVoltage(V)",
                   delimiter='\t', fmt='%.6f')

        # ========================================
        # 4. Sample spectra (Voltage-based sampling)
        # ========================================
        print("[debug] Saving sample spectra (voltage-based sampling)...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 전압 범위 자동 계산
        v_min = np.min(self.voltages)
        v_max = np.max(self.voltages)

        # 7개 전압 포인트 샘플링 (0.1V 간격으로 반올림)
        # 예: -0.4V ~ 0.2V → [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2]
        v_min_rounded = np.round(v_min * 10) / 10  # 0.1V 단위로 반올림
        v_max_rounded = np.round(v_max * 10) / 10
        target_voltages = np.linspace(v_min_rounded, v_max_rounded, 7)

        # 각 목표 전압에 가장 가까운 스펙트럼 찾기
        indices = []
        for target_v in target_voltages:
            # 해당 전압과 가장 가까운 시점 찾기
            voltage_diffs = np.abs(self.voltages - target_v)
            closest_idx = np.argmin(voltage_diffs)
            
            # 중복 방지 (같은 인덱스가 이미 선택된 경우 다음으로 가까운 것 선택)
            while closest_idx in indices and len(voltage_diffs) > len(indices):
                voltage_diffs[closest_idx] = np.inf
                closest_idx = np.argmin(voltage_diffs)
            
            indices.append(closest_idx)

        indices = sorted(set(indices))  # 중복 제거 및 정렬

        # 스펙트럼 플롯
        for idx in indices:
            actual_voltage = self.voltages[idx]
            label = f'V={actual_voltage:.2f}V (t={self.spec_times[idx]:.1f}s)'
            ax.plot(self.wavelengths, self.spectra[idx, :], label=label, linewidth=2)

        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Scattering (a.u.)', fontsize=12)
        ax.set_title(f'{self.sample_name} - Sample Spectra (Voltage Range: {v_min:.2f}V to {v_max:.2f}V)', 
                     fontsize=14)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"{self.sample_name}_sample_spectra.png", dpi=150)
        plt.close()

        # 데이터 저장 (sample spectra)
        print(f"[info] Sampled voltages: {[f'{self.voltages[idx]:.2f}V' for idx in indices]}")
        for idx in indices:
            data = np.column_stack([self.wavelengths, self.spectra[idx, :]])
            np.savetxt(data_dir / f"{self.sample_name}_sample_spectrum_V{self.voltages[idx]:.2f}V.txt",
                       data,
                       header=f"Wavelength(nm)\tIntensity\nVoltage: {self.voltages[idx]:.2f}V, Time: {self.spec_times[idx]:.1f}s",
                       delimiter='\t', fmt='%.6f')