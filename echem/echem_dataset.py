import os
import numpy as np
from pathlib import Path
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any

from . import echem_util as eu


class EChemDataset(object):
    """
    Electrochemical spectroscopy dataset loader and preprocessor
    
    This class handles loading and preprocessing of time-series spectral data
    acquired during electrochemical experiments (CV, CA, CC). Unlike DFS analysis
    which deals with spatial hyperspectral cubes (H × W × λ), EChem data has
    a time-series structure (Time × λ) tracking spectral evolution of a single
    particle or electrode under potential control.
    
    Key differences from DFS Dataset:
    - Data structure: Time × λ instead of H × W × λ
    - Additional data: Synchronized potentiostat measurements
    - No particle detection: Single position tracking
    - Background correction: Similar flatfield approach
    """
    
    def __init__(self, args: Dict[str, Any]):
        """
        Initialize EChemDataset with configuration parameters
        
        Parameters:
        -----------
        args : Dict[str, Any]
            Configuration dictionary containing:
            - ECHEM_SAMPLE_NAME: Name of spectral TDMS file
            - DATA_DIR: Directory containing data files (shared with DFS)
            - ECHEM_CHI_FILE: Potentiostat data file name
            - WHITE_FILE, DARK_FILE: Reference files for flatfield
            - ECHEM_CROP_RANGE_NM: Wavelength range for analysis
            - ECHEM_LOWERCUT, ECHEM_UPPERCUT: Pixel trimming parameters
            - OUTPUT_DIR: Base output directory (EChem saves to OUTPUT_DIR/echem/)
        """
        self.args = args
        self.sample_name = args['ECHEM_SAMPLE_NAME']
        
        # Setup EChem-specific output directory
        self.echem_output_dir = Path(args['OUTPUT_DIR']) / 'echem'
        self.echem_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update args to use echem subdirectory for all outputs
        self.args['ECHEM_OUTPUT_DIR'] = str(self.echem_output_dir)
        
        # Main data containers
        self.spectra = None      # 2D array (Time × λ) - time-series spectra
        self.wavelengths = None  # 1D wavelength array (λ,)
        self.spec_times = None   # 1D time array for spectra acquisition
        
        # Potentiostat data
        self.chi_data = None     # Dict containing voltage, current, charge, time
        self.voltages = None     # Voltage array matched to spectra
        
        # Flatfield references
        self.white_ref = None
        self.dark_ref = None
        self.background = None   # Background spectrum for subtraction
        
        # Processing parameters
        self.lowercut = args.get('ECHEM_LOWERCUT', 140)
        self.uppercut = args.get('ECHEM_UPPERCUT', 260)
        
    def run_dataset(self):
        """
        Execute the complete EChem dataset preprocessing pipeline
        
        Steps:
        1. Load spectral TDMS file (time-series)
        2. Load potentiostat CHI file
        3. Match spectra times to voltage values
        4. Load white/dark references
        5. Apply flatfield correction
        6. Crop wavelength range
        7. Apply background subtraction
        """
        # Step 1: Load spectral TDMS data
        self.load_spectral_tdms()
        
        # Step 2: Load potentiostat data
        self.load_chi_data()
        
        # Step 3: Synchronize spectra with voltage
        self.match_voltage_to_spectra()
        
        # Step 4: Load references and apply flatfield
        self.apply_flatfield()
        
        # Step 5: Crop wavelength range
        self.crop_wavelength()
        
        # Step 6: Background subtraction
        self.apply_background_correction()
        
        # Step 7: Save debug visualizations
        self.save_debug_plots()
    
    def load_spectral_tdms(self):
        """
        Load TDMS file containing time-series spectral data
        
        EChem TDMS structure (different from DFS):
        - Root properties: acquisition parameters
        - Info group: wavelength array, background, time stamps
        - Spectra group: Multiple channels, each = one time point spectrum
        
        This creates a 2D array (Time × λ) instead of 3D cube (H × W × λ)
        """
        # Construct file path - use DATA_DIR (shared with DFS)
        tdms_path = os.path.join(self.args['DATA_DIR'], 
                                f"{self.sample_name}.tdms")
        
        print(f"\n[debug] Loading EChem TDMS file: {tdms_path}")
        
        # Read TDMS file
        td = TdmsFile.read(tdms_path)
        
        # Extract wavelength array from Info group
        info_group = td['Info']
        if info_group is None:
            raise RuntimeError("No 'Info' group found in EChem TDMS file")
        
        # Find wavelength channel
        wl_channel = None
        for ch in info_group.channels():
            if ch.name == 'wvlths':
                wl_channel = ch
                break
        
        if wl_channel is None:
            raise RuntimeError("No 'wvlths' channel found in Info group")
        
        self.wavelengths = wl_channel[:].astype(np.float32)
        
        # Extract background spectrum
        bg_channel = None
        for ch in info_group.channels():
            if 'bg' in ch.name.lower() or 'background' in ch.name.lower():
                bg_channel = ch
                break
        
        if bg_channel is None:
            print("[warning] No background channel found, will use zeros")
            self.background = np.zeros_like(self.wavelengths)
        else:
            bg_raw = bg_channel[:].astype(np.float32)
            # Handle wavelength grouping (MATLAB: wvgrp)
            wvgrp = td.properties.get('wvlth group', 1)
            if wvgrp == 2:
                self.background = bg_raw
            else:
                # Average adjacent wavelengths if grouped differently
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
        
        print(f"[info] Wavelength array: {self.wavelengths.min():.1f}-{self.wavelengths.max():.1f} nm, "
              f"{len(self.wavelengths)} points")
        
        # Extract spectra from Spectra group
        spectra_group = td['Spectra']
        if spectra_group is None:
            raise RuntimeError("No 'Spectra' group found in EChem TDMS file")
        
        spec_channels = list(spectra_group.channels())
        n_spectra = len(spec_channels)
        
        print(f"[info] Found {n_spectra} time-point spectra")
        
        # Check wavelength grouping for spectra
        wvgrp = td.properties.get('wvlth group', 1)
        
        # Initialize spectra array
        if wvgrp == 2:
            self.spectra = np.zeros((n_spectra, len(self.wavelengths)), dtype=np.float32)
        else:
            # If wavelengths are grouped, average them
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
        
        print(f"[debug] Loaded spectra shape: {self.spectra.shape}")
        print(f"[debug] Time array: {self.spec_times.min():.2f} - {self.spec_times.max():.2f} s")
        print(f"[debug] Spectral data range: [{self.spectra.min():.2f}, {self.spectra.max():.2f}]")
    
    def load_chi_data(self):
        """
        Load and parse CHI potentiostat data file
        
        CHI files contain:
        - Header: Technique type (CV/CA/CC), experimental parameters
        - Data: Time, Potential, Current, Charge columns
        """
        chi_path = os.path.join(self.args['DATA_DIR'],
                               f"{self.args['ECHEM_CHI_FILE']}.txt")
        
        print(f"\n[debug] Loading CHI potentiostat file: {chi_path}")
        
        # Parse CHI file using utility function
        self.chi_data = eu.parse_chi_file(chi_path)
        
        print(f"[info] Detected technique: {self.chi_data['technique']}")
        print(f"[info] CHI data points: {len(self.chi_data['data'])}")
        
        # Extract columns
        if len(self.chi_data['data']) > 0:
            chi_times = self.chi_data['data'][:, 0]
            chi_voltages = self.chi_data['data'][:, 1]
            
            print(f"[info] CHI time range: {chi_times.min():.2f} - {chi_times.max():.2f} s")
            print(f"[info] Voltage range: {chi_voltages.min():.3f} - {chi_voltages.max():.3f} V")
    
    def match_voltage_to_spectra(self):
        """
        Synchronize spectral acquisition times with potentiostat voltage values
        
        Since spectra and potentiostat are acquired independently, we need to
        find the voltage value at each spectrum time point using nearest neighbor.
        """
        print(f"\n[debug] Matching spectra times to voltage values...")
        
        if len(self.chi_data['data']) == 0:
            print("[warning] No CHI data available, using zeros for voltage")
            self.voltages = np.zeros(len(self.spec_times))
            return
        
        chi_times = self.chi_data['data'][:, 0]
        chi_voltages = self.chi_data['data'][:, 1]
        
        # Use utility function for matching
        self.voltages = eu.match_spectra_to_voltage(
            self.spec_times, chi_times, chi_voltages
        )
        
        print(f"[info] Matched voltages: {self.voltages.min():.3f} - {self.voltages.max():.3f} V")
        print(f"[info] Matched {len(self.voltages)} spectra to voltage values")
    
    def apply_flatfield(self):
        """
        Apply flatfield correction using white/dark references
        
        Similar to DFS flatfield but simpler since we don't have spatial variation.
        Each spectrum is normalized by (spectrum - dark) / (white - dark)
        """
        print(f"\n[debug] Applying flatfield correction...")
        
        # Construct paths to reference files - use DATA_DIR (shared with DFS)
        white_path = os.path.join(self.args['DATA_DIR'], 
                                 self.args['WHITE_FILE'])
        dark_path = os.path.join(self.args['DATA_DIR'], 
                                self.args['DARK_FILE'])
        
        print(f"  White reference: {white_path}")
        print(f"  Dark reference: {dark_path}")
        
        # Load white reference
        td_white = TdmsFile.read(white_path)
        white_group = td_white['Spectra']
        white_channels = list(white_group.channels())
        
        # Average all white reference spectra
        white_spectra = []
        for ch in white_channels:
            white_spectra.append(ch[:].astype(np.float32))
        white_array = np.array(white_spectra)
        
        # Handle wavelength grouping
        wvgrp = td_white.properties.get('wvlth group', 1)
        if wvgrp == 2:
            white_avg = white_array.mean(axis=0)
        else:
            # Average adjacent wavelengths
            white_combined = white_array[:, ::2] + white_array[:, 1::2]
            white_avg = white_combined.mean(axis=0)
        
        # Load dark reference
        td_dark = TdmsFile.read(dark_path)
        dark_group = td_dark['Spectra']
        dark_channels = list(dark_group.channels())
        
        # Average all dark reference spectra
        dark_spectra = []
        for ch in dark_channels:
            dark_spectra.append(ch[:].astype(np.float32))
        dark_array = np.array(dark_spectra)
        
        # Handle wavelength grouping
        wvgrp_dark = td_dark.properties.get('wvlth group', 1)
        if wvgrp_dark == 2:
            dark_avg = dark_array.mean(axis=0)
        else:
            # Average adjacent wavelengths
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
        
        print(f"[debug] After flatfield: range [{self.spectra.min():.3f}, {self.spectra.max():.3f}]")
    
    def crop_wavelength(self):
        """
        Crop wavelength range and trim pixels from edges
        
        MATLAB uses lowercut and uppercut to remove unreliable pixels
        from blue and red ends of the spectrum.
        Wavelength range uses CROP_RANGE_NM (shared with DFS).
        """
        print(f"\n[debug] Cropping wavelength range...")
        print(f"  Lower cut: {self.lowercut} pixels")
        print(f"  Upper cut: {self.uppercut} pixels")
        
        # Get wavelength range from shared CROP_RANGE_NM parameter
        crop_range = self.args.get('CROP_RANGE_NM', (self.wavelengths.min(), self.wavelengths.max()))
        print(f"  Wavelength range: {crop_range[0]}-{crop_range[1]} nm")
        
        # Apply pixel trimming
        if self.uppercut > 0:
            self.wavelengths = self.wavelengths[self.lowercut:-self.uppercut]
            self.spectra = self.spectra[:, self.lowercut:-self.uppercut]
        else:
            self.wavelengths = self.wavelengths[self.lowercut:]
            self.spectra = self.spectra[:, self.lowercut:]
        
        # Additional wavelength range cropping (if needed)
        wl_mask = (self.wavelengths >= crop_range[0]) & (self.wavelengths <= crop_range[1])
        self.wavelengths = self.wavelengths[wl_mask]
        self.spectra = self.spectra[:, wl_mask]
        
        print(f"[debug] After cropping:")
        print(f"  Wavelength range: {self.wavelengths.min():.1f} - {self.wavelengths.max():.1f} nm")
        print(f"  Spectra shape: {self.spectra.shape}")
    
    def apply_background_correction(self):
        """
        Apply additional background correction if needed
        
        For EChem, background is typically already handled in flatfield,
        but this method is available for additional processing.
        """
        print(f"\n[debug] Background correction already applied in flatfield step")
        # EChem typically doesn't need additional background correction
        # beyond flatfield, but this method can be extended if needed
    
    def save_debug_plots(self):
        """
        Save debug visualizations of loaded EChem data
        """
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