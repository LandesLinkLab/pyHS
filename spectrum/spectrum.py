import os
import sys
import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any, Union

from . import spectrum_util as su

class SpectrumAnalyzer:
    """
    Spectrum analysis class for DFS (Dark Field Scattering) hyperspectral data
    
    This class handles the complete spectral analysis pipeline:
    1. Fitting Lorentzian functions to all detected particle spectra
    2. Quality filtering based on fitting parameters (FWHM, R-squared)
    3. Selecting representative spectra for each cluster
    4. Generating plots and saving results
    5. Computing statistics and creating summary reports
    
    The class works with preprocessed data from the Dataset class and applies
    MATLAB-compatible analysis methods for consistency with existing workflows.
    """
    
    def __init__(self, args: Dict[str, Any], dataset):
        """
        Initialize SpectrumAnalyzer with configuration and preprocessed data
        
        Parameters:
        -----------
        args : Dict[str, Any]
            Configuration dictionary containing analysis parameters:
            - MAX_WIDTH_NM: Maximum allowed FWHM for filtering
            - RSQ_MIN: Minimum R-squared value for fitting quality
            - FIT_RANGE_NM: Wavelength range for Lorentzian fitting
            - Output and visualization parameters
        dataset : Dataset
            Preprocessed dataset object containing hyperspectral cube,
            wavelengths, detected clusters, and reference data
        """
        self.args = args
        self.dataset = dataset
        
        # Results storage containers
        self.cluster_fits = []  # All pixel fitting results organized by cluster
        self.representatives = []  # Representative spectrum for each cluster
        self.rejected_spectra = [] # Rejected spectra with reasons (for debugging)
        
    def run_spectrum(self):
        """
        Execute the complete spectrum analysis pipeline
        
        This method runs all analysis steps in sequence:
        1. Fit Lorentzian functions to all particle spectra
        2. Apply quality filters and select representatives
        3. Generate plots for each valid spectrum
        4. Save particle map with markers
        5. Export spectral data to text files
        6. Create pickle dump of all results
        7. Print comprehensive analysis summary
        """
        self.fit_all_particles()      # Step 1: Fit all detected particles
        self.select_representatives() # Step 2: Select best spectrum per cluster
        self.plot_representatives()   # Step 3: Generate individual spectrum plots
        self.save_dfs_particle_map() # Step 4: Create annotated particle map
        self.save_spectra()          # Step 5: Export data to text files
        self.dump_pkl()              # Step 6: Save all results to pickle file
        self.print_summary()         # Step 7: Print comprehensive summary
    
    def fit_all_particles(self):
        """
        Fit Lorentzian functions to all detected particle clusters
        
        This method:
        - Integrates 3x3 pixel regions around each cluster center
        - Fits Lorentzian functions to the integrated spectra
        - Applies quality filters based on FWHM and R-squared values
        - Computes MATLAB-style Signal-to-Noise ratios
        - Stores both accepted and rejected results for analysis
        
        The fitting uses the background-corrected spectral data and follows
        MATLAB-compatible methods for consistency with existing analysis.
        """
        print("\n[Step] Fitting all particles in clusters...")
        
        if not self.dataset.clusters:
            print("[warning] No clusters found for analysis")
            return
        
        # Integration parameters (MATLAB-compatible 3x3 integration)
        int_size = 3
        int_var = (int_size - 1) // 2  # Half-width of integration region

        # Quality filter parameters
        max_width = self.args.get("MAX_WIDTH_NM", 59)  # Maximum allowed FWHM
        min_r2 = self.args.get("RSQ_MIN", 0.9)         # Minimum R-squared

        # Statistics tracking
        stats = {'total_clusters': len(self.dataset.clusters),
                'rejected_width': 0,
                'rejected_fitting': 0,
                'accepted': 0}
        
        # Process each detected cluster
        for cluster in self.dataset.clusters:
            cluster_results = []
            
            print(f"\n[Fitting Cluster {cluster['label']}] - {cluster['size']} pixels")
            
            # Use cluster center for integration
            center_row = int(cluster['center'][0])
            center_col = int(cluster['center'][1])
            
            # Integrate spectrum over 3x3 pixel region (MATLAB-style)
            H, W, L = self.dataset.cube.shape
            integrated_spectrum = np.zeros(L)
            pixel_count = 0
            
            # Sum spectra from 3x3 region around center
            for m in range(-int_var, int_var + 1):
                for l in range(-int_var, int_var + 1):
                    row = center_row + m
                    col = center_col + l
                    
                    # Check boundaries
                    if 0 <= row < H and 0 <= col < W:
                        integrated_spectrum += self.dataset.cube[row, col, :]
                        pixel_count += 1
            
            # Skip if no valid pixels or very low signal
            if pixel_count == 0 or integrated_spectrum.max() < 0.01:
                continue
            
            # Fit Lorentzian function to integrated spectrum
            y_fit, params, r2 = su.fit_lorentz(self.args, integrated_spectrum, self.dataset.wvl)

            # Extract fitted FWHM for quality filtering
            fitted_width = params.get('c1', 0) # FWHM parameter

            # Quality Filter 1: FWHM limit
            if fitted_width > max_width:
                print(f"  [Rejected - Width] FWHM too large: {fitted_width:.1f} nm (max: {max_width:.1f} nm)")
                stats['rejected_width'] += 1

                # Store rejected spectrum with reason
                self.rejected_spectra.append({
                    'cluster_label': cluster['label'],
                    'row': center_row,
                    'col': center_col,
                    'spectrum': integrated_spectrum,
                    'wavelengths': self.dataset.wvl,
                    'reason': f"Width too large: {fitted_width:.1f} nm",
                    'fitted_width': fitted_width
                })
                continue

            # Quality Filter 2: R-squared limit
            if r2 < min_r2:
                print(f"  [Rejected - Fitting] R² too low: {r2:.3f} (min: {min_r2})")
                stats['rejected_fitting'] += 1
                
                self.rejected_spectra.append({
                    'cluster_label': cluster['label'],
                    'row': center_row,
                    'col': center_col,
                    'spectrum': integrated_spectrum,
                    'wavelengths': self.dataset.wvl,
                    'reason': f"R² too low: {r2:.3f}",
                    'fitted_width': fitted_width
                })
                continue

            # Compute MATLAB-style Signal-to-Noise Ratio
            # 1. Noise: standard deviation of fitting residuals over full range
            resid = np.abs(integrated_spectrum - y_fit)  # Absolute residuals
            noise = np.std(resid)
            
            # 2. Signal: raw data value at resonance wavelength
            resonance_wl = params.get('b1', self.dataset.wvl[np.argmax(integrated_spectrum)])
            resonance_idx = np.argmin(np.abs(self.dataset.wvl - resonance_wl))
            signal = integrated_spectrum[resonance_idx]
            
            # 3. SNR calculation
            snr = signal / noise if noise > 0 else 0

            stats['accepted'] += 1
            
            # Store successful fitting results
            result = {
                'row': center_row,
                'col': center_col,
                'spectrum': integrated_spectrum,
                'fit': y_fit,
                'params': params,
                'r2': r2,
                'snr': snr,
                'peak_wl': params.get('b1', self.dataset.wvl[resonance_idx]),
                'fwhm': params.get('c1', 0),
                'peak_intensity': integrated_spectrum[resonance_idx],
                'raw_peak_intensity': signal,  # Store raw signal value
                'integrated_pixels': pixel_count
            }
            
            cluster_results.append(result)
            
            # Store cluster fitting results
            self.cluster_fits.append({
                'cluster_label': cluster['label'],
                'cluster_size': cluster['size'],
                'fits': cluster_results
            })
            
            # Print fitting summary for this cluster
            print(f"  Integrated {pixel_count} pixels, peak at {result['peak_wl']:.1f} nm")
            print(f"  FWHM: {fitted_width:.1f} nm (max allowed: {max_width:.1f} nm)")
            print(f"  SNR (MATLAB style): {snr:.1f} (raw signal: {signal:.1f}, noise: {noise:.3f})")

        # Print overall filtering statistics
        print(f"\n[Filtering Statistics]")
        print(f"  Total clusters: {stats['total_clusters']}")
        print(f"  Rejected (width > {max_width:.0f} nm): {stats['rejected_width']}")
        print(f"  Rejected (poor fit): {stats['rejected_fitting']}")
        print(f"  Accepted: {stats['accepted']}")
    
    def select_representatives(self):
        """
        Select representative spectrum for each cluster
        
        Since we now use one integrated spectrum per cluster (3x3 integration),
        this method simply selects that single spectrum as the representative.
        In more complex scenarios, this could involve selecting the best spectrum
        from multiple fits per cluster.
        
        The method stores detailed information about each representative including
        spectral parameters, quality metrics, and spatial information.
        """
        print("\n[Step] Selecting representative spectra for each cluster...")
        
        for cluster_fit in self.cluster_fits:
            fits = cluster_fit['fits']
            
            if not fits:
                print(f"  Cluster {cluster_fit['cluster_label']}: No valid fits")
                continue
            
            # Currently only one fit per cluster (3x3 integration)
            best = fits[0]
            
            # Print representative information
            print(f"\n  Cluster {cluster_fit['cluster_label']}:")
            print(f"    Center: ({best['row']}, {best['col']})")
            print(f"    Integrated pixels: {best['integrated_pixels']}")
            print(f"    Peak: {best['peak_wl']:.1f} nm @ {best['peak_intensity']:.1f}")
            print(f"    FWHM: {best['fwhm']:.1f} nm")
            print(f"    S/N: {best['snr']:.1f}, R²: {best['r2']:.3f}")
            
            # Store as representative
            self.representatives.append({
                'cluster_label': cluster_fit['cluster_label'],
                'cluster_size': cluster_fit['cluster_size'],
                'row': best['row'],
                'col': best['col'],
                'spectrum': best['spectrum'],
                'fit': best['fit'],
                'params': best['params'],
                'r2': best['r2'],
                'snr': best['snr'],
                'peak_wl': best['peak_wl'],
                'fwhm': best['fwhm'],
                'peak_intensity': best['peak_intensity'],
                'integrated_pixels': best['integrated_pixels']
            })
        
        print(f"\n[Summary] {len(self.representatives)} valid representatives from {len(self.cluster_fits)} clusters")
    
    def plot_representatives(self):
        """
        Generate individual spectrum plots for each representative
        
        This method:
        - Creates MATLAB-style plots for each valid representative spectrum
        - Shows both experimental data and Lorentzian fit
        - Includes key parameters (wavelength, FWHM, SNR) as text annotations
        - Saves high-resolution plots to the output directory
        - Uses consistent styling and formatting for all plots
        """
        print("\n[Step] Plotting representative spectra...")
        
        out_dir = Path(self.args['OUTPUT_DIR'])
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for i, rep in enumerate(self.representatives):
            # Generate sequential particle numbers for consistent naming
            particle_num = i + 1
            
            # Create individual spectrum plot using utility function
            su.plot_spectrum(
                self.dataset.wvl,           # Wavelength array
                rep['spectrum'],            # Experimental spectrum
                rep['fit'],                 # Lorentzian fit
                f"Particle {particle_num} (Cluster {rep['cluster_label']})", # Title
                out_dir / f"{self.dataset.sample_name}_particle_{particle_num:03d}.png", # Output path
                dpi=self.args["FIG_DPI"],   # Resolution
                params=rep['params'],       # Fitting parameters
                snr=rep['snr'],            # Signal-to-noise ratio
                args=self.args             # Additional configuration
            )
    
    def save_dfs_particle_map(self):
        """
        Save annotated particle map showing all detected and analyzed particles
        
        This method:
        - Uses the max intensity map before background correction for better visibility
        - Marks each representative particle with numbered annotations
        - Shows resonance wavelengths and cluster information
        - Creates a publication-ready figure with proper scaling and coloring
        - Saves to the main output directory for easy access
        """
        if not self.representatives:
            return
        
        out_dir = Path(self.args['OUTPUT_DIR'])
        output_path = out_dir / f"{self.dataset.sample_name}_dfs_markers.png"

        # Use max map before background correction if available (better contrast)
        display_map = self.dataset.max_map_before_bg if hasattr(self.dataset, 'max_map_before_bg') else self.dataset.max_map
        
        # Generate particle map using utility function
        su.save_dfs_particle_map(
            display_map,              # Background intensity map
            self.representatives,     # Representative particles to mark
            output_path,             # Output file path
            self.dataset.sample_name, # Sample name for title
            self.args
        )
    
    def save_spectra(self):
        """
        Export spectral data to tab-separated text files
        
        This method:
        - Creates a 'spectra' subdirectory in the output folder
        - Saves each representative spectrum as a separate text file
        - Includes wavelength, intensity, and fit data in columns
        - Adds metadata headers with particle information and fitting parameters
        - Uses consistent naming convention for easy identification
        """
        print("\n[Step] Saving spectra data...")
        
        out_dir = Path(self.args['OUTPUT_DIR']) / "spectra"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for i, rep in enumerate(self.representatives):
            # Generate sequential particle numbers
            particle_num = i + 1
            
            if output_unit == 'eV':

                energy = 1239.842 / self.dataset.wvl
                energy = energy[::-1]
                spectrum = rep['spectrum'][::-1]
                fit = ref['fit'][::-1]

                data = np.column_stack(energy, spectrum, fit)

                peak_ev = 1239.842 / rep['peak_wl']
                peak_nm = rep['peak_wl']
                fwhm_nm = rep['fwhm']
                fwhm_ev = 1239.842/(peak_nm - fwhm_nm/2) - 1239.842/(peak_nm + fwhm_nm/2)
            
                header = f"Energy(eV)\tIntensity\tFit\n"
                header += f"# Particle {particle_num}, Cluster {rep['cluster_label']}, Position ({rep['row']},{rep['col']})\n"
                header += f"# Peak: {peak_ev:.3f} eV ({rep['peak_wl']:.1f} nm), FWHM: {fwhm_ev:.3f} eV, S/N: {rep['snr']:.1f}"
            
            else:

                # Prepare data array: wavelength, intensity, fit
                data = np.column_stack((self.dataset.wvl, rep['spectrum'], rep['fit']))
                
                # Create informative header with metadata
                header = f"Wavelength(nm)\tIntensity\tFit\n"
                header += f"# Particle {particle_num}, Cluster {rep['cluster_label']}, Position ({rep['row']},{rep['col']})\n"
                header += f"# Peak: {rep['peak_wl']:.1f} nm, FWHM: {rep['fwhm']:.1f} nm, S/N: {rep['snr']:.1f}"
            
            # Save to text file with tab separation
            np.savetxt(
                out_dir / f"{self.dataset.sample_name}_particle_{particle_num:03d}.txt",
                data,
                delimiter='\t',
                header=header,
                fmt='%.3f'  # 3 decimal places
            )
    
    def print_summary(self):
        """
        Print comprehensive analysis summary with statistics
        
        This method provides a detailed summary including:
        - Basic counts of detected, analyzed, and valid particles
        - Statistical analysis of key parameters (wavelength, FWHM, SNR, R²)
        - Quality metrics and filtering results
        - Information about rejected spectra and reasons
        - Cluster size distribution
        
        The summary is formatted for easy reading and interpretation.
        """
        print("\n" + "="*60)
        print("DFS ANALYSIS SUMMARY")
        print("="*60)
        print(f"Sample: {self.dataset.sample_name}")
        print(f"Total clusters detected: {len(self.dataset.clusters) if self.dataset.clusters else 0}")
        print(f"Clusters analyzed: {len(self.cluster_fits)}")
        print(f"Valid representatives: {len(self.representatives)}")
        print(f"Rejected spectra: {len(self.rejected_spectra)}")
        
        if self.representatives:

            output_unit = self.args.get('OUTPUT_UNIT', 'eV')

            if output_unit == 'eV':

                peak_energies = [1239.842/r['peak_wl'] for r in self.representatives]

                fwhm_evs = []
                for r in self.representatives:
                    peak_nm = r['peak_wl']
                    fwhm_nm = r['fwhm']
                    if fwhm_nm > 0:
                        fwhm_ev = 1239.842/(peak_nm - fwhm_nm/2) - 1239.842/(peak_nm + fwhm_nm/2)
                        fwhm_evs.append(abs(fwhm_ev))
                
                print(f"\nResonance energy: {np.mean(peak_energies):.3f} ± {np.std(peak_energies):.3f} eV")
                print(f"  Range: {min(peak_energies):.3f} - {max(peak_energies):.3f} eV")
                
                if fwhm_evs:
                    print(f"\nFWHM: {np.mean(fwhm_evs):.3f} ± {np.std(fwhm_evs):.3f} eV")
                    print(f"  Range: {min(fwhm_evs):.3f} - {max(fwhm_evs):.3f} eV")

            else:

                # Extract parameters for statistical analysis
                wavelengths = [r['peak_wl'] for r in self.representatives]
                fwhms = [r['fwhm'] for r in self.representatives if r['fwhm'] > 0]
                snrs = [r['snr'] for r in self.representatives]
                cluster_sizes = [r['cluster_size'] for r in self.representatives]
                r2s = [r['r2'] for r in self.representatives]
                
                # Resonance wavelength statistics
                print(f"\nResonance wavelength: {np.mean(wavelengths):.1f} ± {np.std(wavelengths):.1f} nm")
                print(f"  Range: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
                
                # FWHM statistics
                if fwhms:
                    print(f"\nFWHM: {np.mean(fwhms):.1f} ± {np.std(fwhms):.1f} nm")
                    print(f"  Range: {min(fwhms):.1f} - {max(fwhms):.1f} nm")
            
            # Signal-to-noise ratio statistics
            print(f"\nS/N ratio: {np.mean(snrs):.1f} ± {np.std(snrs):.1f}")
            print(f"  Range: {min(snrs):.1f} - {max(snrs):.1f}")
            
            # R-squared statistics (fitting quality)
            print(f"\nR² values: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
            print(f"  Range: {min(r2s):.3f} - {max(r2s):.3f}")
            
            # Cluster size statistics
            print(f"\nCluster sizes: {np.mean(cluster_sizes):.1f} ± {np.std(cluster_sizes):.1f} pixels")
            print(f"  Range: {min(cluster_sizes)} - {max(cluster_sizes)} pixels")
        
        # Rejection analysis
        if self.rejected_spectra:
            width_rejected = [r for r in self.rejected_spectra if 'Width too large' in r['reason']]
            if width_rejected:
                widths = [r['fitted_width'] for r in width_rejected]
                print(f"\n[Width Rejections]")
                print(f"  Count: {len(width_rejected)}")
                print(f"  Width range: {min(widths):.1f} - {max(widths):.1f} nm")
        
        print("="*60)
    
    def dump_pkl(self):
        """
        Save all analysis results to a comprehensive pickle file
        
        This method:
        - Creates a complete data package with all results and metadata
        - Includes original configuration parameters for reproducibility
        - Stores both successful and rejected analysis results
        - Adds timestamp and sample information
        - Uses highest protocol for efficient storage
        
        The pickle file serves as a complete record of the analysis and can be
        loaded later for further processing or comparison with other samples.
        """
        out = Path(self.args['OUTPUT_DIR']) / f"{self.dataset.sample_name}_results.pkl"
        
        # Create comprehensive data package
        payload = {
            'sample': self.dataset.sample_name,
            'wavelengths': self.dataset.wvl,
            'cube_shape': self.dataset.cube.shape,
            'max_map': self.dataset.max_map,
            'clusters': self.dataset.clusters,
            'cluster_fits': self.cluster_fits,
            'representatives': self.representatives,
            'rejected_spectra': self.rejected_spectra,
            'config': self.args,
            'output_unit': self.args.get('OUTPUT_UNIT', 'eV'),
            'analysis_date': str(datetime.now().strftime("%m-%d-%Y %H:%M:%S"))
        }
        
        # Save with highest compression
        with open(out, "wb") as f:
            pkl.dump(payload, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        print(f"\n[info] Results saved to: {out}")