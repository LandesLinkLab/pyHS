import os
import sys
import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import curve_fit  # ← 추가!

# Import from spectrum module (reuse Lorentzian fitting)
from spectrum import spectrum_util as su
import echem.echem_util as eu


class EChemAnalyzer:
    """
    Electrochemical spectroscopy analyzer for CV/CA/CC experiments
    
    This class handles spectral analysis during electrochemical experiments:
    1. Fit Lorentzian functions to all time-point spectra
    2. Identify CV cycles or CA/CC steps
    3. Calculate cycle-averaged spectral parameters
    4. Correlate spectral changes with applied potential
    5. Generate plots and export results
    
    The analysis follows MATLAB workflow from cv_analysis_CF_EKS819.m
    """
    
    def __init__(self, args: Dict[str, Any], dataset):
        """
        Initialize EChemAnalyzer with configuration and preprocessed data
        
        Parameters:
        -----------
        args : Dict[str, Any]
            Configuration dictionary containing analysis parameters
        dataset : EChemDataset
            Preprocessed EChem dataset with spectra and voltage data
        """
        self.args = args
        self.dataset = dataset
        
        # Analysis parameters
        self.technique = args.get('ECHEM_TECHNIQUE', 'CV')
        self.scatt_type = args.get('ECHEM_SCATT_TYPE', 'single')
        self.ocp = args.get('ECHEM_OCP', 0.0)
        self.cycle_start = args.get('ECHEM_CYCLE_START', 1)
        self.cycle_backcut = args.get('ECHEM_CYCLE_BACKCUT', 0)
        
        # Results storage
        self.cycle_boundaries = []
        self.fitted_params = []
        self.cycles = []
        self.rejected_fits = []
        
    def run_analysis(self):
        """Execute the complete EChem spectroscopy analysis pipeline"""
        print("\n" + "="*60)
        print("ECHEM SPECTROSCOPY ANALYSIS")
        print("="*60)
        
        self.fit_all_spectra()
        self.identify_cycles()
        self.calculate_cycle_averages()
        self.plot_overview()
        self.plot_cycles()
        self.save_spectra_data()
        self.dump_results()
        self.print_summary()
    
    def fit_lorentz_energy(self, energy: np.ndarray, spectrum: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], float]:
        """
        Fit Lorentzian in energy space
        
        Parameters:
        -----------
        energy : np.ndarray
            Energy array (eV) - MUST be monotonically increasing
        spectrum : np.ndarray
            Intensity array
        
        Returns:
        --------
        Tuple[np.ndarray, Dict[str, float], float]
            - Fitted curve
            - Parameters dict with keys: 'a', 'b1' (peak in eV), 'c1' (FWHM in eV)
            - R-squared value
        """
        
        def lorentz_energy(E, a, E0, gamma):
            """
            Lorentzian in energy space
            E0: peak energy (eV)
            gamma: FWHM (eV)
            """
            return (2*a/np.pi) * (gamma / (4*(E-E0)**2 + gamma**2))
        
        # Validate input
        if len(spectrum) == 0 or np.all(spectrum <= 0) or np.isnan(spectrum).any():
            return np.zeros_like(spectrum), {'a': 0, 'b1': 0, 'c1': 0}, 0.0
        
        # Check if spectrum has sufficient variation
        if spectrum.max() - spectrum.min() < 0.01:
            print(f"[warning] Spectrum has insufficient variation")
            return np.zeros_like(spectrum), {'a': 0, 'b1': 0, 'c1': 0}, 0.0
        
        # Initial parameter guesses
        idx_max = np.argmax(spectrum)
        
        if spectrum[idx_max] <= 0:
            return np.zeros_like(spectrum), {'a': 0, 'b1': 0, 'c1': 0}, 0.0
        
        # Initial guesses
        a0 = spectrum[idx_max] * np.pi / 2
        E0 = energy[idx_max]  # Peak energy
        
        # Estimate FWHM
        half_max = spectrum[idx_max] / 2
        above_half = spectrum > half_max
        if np.sum(above_half) > 1:
            indices = np.where(above_half)[0]
            gamma0 = abs(energy[indices[-1]] - energy[indices[0]])
            gamma0 = max(gamma0, 0.01)  # Minimum FWHM
        else:
            gamma0 = 0.1  # Default FWHM: 0.1 eV
        
        p0 = [a0, E0, gamma0]
        
        try:
            # Set bounds (energy space)
            bounds = (
                [0, energy.min(), 0.001],           # Lower bounds
                [np.inf, energy.max(), 2.0]         # Upper bounds (max FWHM = 2 eV)
            )
            
            # Perform fitting
            popt, pcov = curve_fit(
                lorentz_energy, 
                energy, 
                spectrum, 
                p0=p0,
                bounds=bounds,
                maxfev=8000,
                method='trf'
            )
            
            # Generate fitted curve
            y_fit = lorentz_energy(energy, *popt)
            
            # Calculate R-squared
            ss_res = np.sum((spectrum - y_fit)**2)
            ss_tot = np.sum((spectrum - spectrum.mean())**2)
            
            # Handle edge cases for R²
            if ss_tot < 1e-10:  # Essentially flat spectrum
                r2 = 0.0
            else:
                r2 = 1 - (ss_res / ss_tot)
            
            # Package parameters
            params = {
                'a': popt[0],    # Amplitude
                'b1': popt[1],   # Peak energy (eV)
                'c1': popt[2]    # FWHM (eV)
            }
            
            return y_fit, params, float(r2)
        
        except Exception as e:
            print(f"[warning] Fitting failed: {str(e)}")
            return np.zeros_like(spectrum), {'a': 0, 'b1': 0, 'c1': 0}, 0.0
    
    def fit_all_spectra(self):
        """
        Fit Lorentzian function to all time-point spectra
        
        This uses the same Lorentzian fitting but applied to time-series data
        instead of spatial pixels.
        """
        print("\n[Step] Fitting Lorentzian to all time-point spectra...")
        
        n_spectra = self.dataset.spectra.shape[0]
        wavelengths = self.dataset.wavelengths
        
        # Convert wavelengths to energy (eV) for fitting
        energy = 1239.842 / wavelengths

        if energy[0] > energy[-1]:
            energy = energy[::-1]
            print("[debug] Reversed energy array to ensure monotonic increase")
        
        # Quality filter parameters
        max_width = self.args.get("ECHEM_MAX_WIDTH_EV", 0.15)
        min_r2 = self.args.get("ECHEM_RSQ_MIN", 0.85)
        
        # Track statistics
        stats = {
            'total': n_spectra,
            'rejected_width': 0,
            'rejected_fitting': 0,
            'rejected_negative_r2': 0,
            'accepted': 0
        }
        
        # Fit each spectrum
        for i in range(n_spectra):
            spectrum = self.dataset.spectra[i, :]
            
            # Reverse spectrum if energy was reversed
            if energy[-1] > energy[0]:
                spectrum = spectrum[::-1]
            
            # Skip if spectrum is invalid
            if spectrum.max() < 0.01 or np.any(spectrum < 0):
                print(f"  Spectrum {i}: Skipped (invalid data)")
                self.rejected_fits.append({
                    'index': i,
                    'time': self.dataset.spec_times[i],
                    'voltage': self.dataset.voltages[i],
                    'reason': 'Invalid spectrum data',
                    'fwhm': 0,
                    'r2': 0
                })
                continue
            
            # Fit Lorentzian using energy space
            y_fit, params, r2 = self.fit_lorentz_energy(energy, spectrum)
            
            # Extract FWHM for quality filtering
            fwhm = params.get('c1', 0)  # FWHM in eV
            
            # Quality Filter 1: Negative R² (fitting worse than mean)
            if r2 < 0:
                print(f"  Spectrum {i}: Rejected - R² negative ({r2:.3f})")
                stats['rejected_negative_r2'] += 1
                
                self.rejected_fits.append({
                    'index': i,
                    'time': self.dataset.spec_times[i],
                    'voltage': self.dataset.voltages[i],
                    'reason': f"R² negative: {r2:.3f}",
                    'fwhm': fwhm,
                    'r2': r2
                })
                continue
            
            # Quality Filter 2: FWHM limit
            if fwhm > max_width:
                print(f"  Spectrum {i}: Rejected - FWHM too large ({fwhm:.3f} eV > {max_width:.3f} eV)")
                stats['rejected_width'] += 1
                
                self.rejected_fits.append({
                    'index': i,
                    'time': self.dataset.spec_times[i],
                    'voltage': self.dataset.voltages[i],
                    'reason': f"FWHM too large: {fwhm:.3f} eV",
                    'fwhm': fwhm,
                    'r2': r2
                })
                continue
            
            # Quality Filter 3: R-squared minimum
            if r2 < min_r2:
                print(f"  Spectrum {i}: Rejected - R² too low ({r2:.3f} < {min_r2})")
                stats['rejected_fitting'] += 1
                
                self.rejected_fits.append({
                    'index': i,
                    'time': self.dataset.spec_times[i],
                    'voltage': self.dataset.voltages[i],
                    'reason': f"R² too low: {r2:.3f}",
                    'fwhm': fwhm,
                    'r2': r2
                })
                continue
            
            # Calculate SNR (MATLAB style)
            resid = np.abs(spectrum - y_fit)
            noise = np.std(resid)
            
            peak_energy = params.get('b1', energy[np.argmax(spectrum)])
            peak_idx = np.argmin(np.abs(energy - peak_energy))
            signal = spectrum[peak_idx]
            
            snr = signal / noise if noise > 0 else 0
            
            stats['accepted'] += 1
            
            # Store results
            result = {
                'index': i,
                'time': self.dataset.spec_times[i],
                'voltage': self.dataset.voltages[i],
                'spectrum': spectrum,
                'fit': y_fit,
                'params': params,
                'r2': r2,
                'snr': snr,
                'peakeV1': params.get('b1', 0),
                'FWHMeV1': params.get('c1', 0),
                'area1': params.get('a', 0),
                'peaknm1': 1239.842 / params.get('b1', 1) if params.get('b1', 0) > 0 else 0,
                'FWHMnm1': self.convert_fwhm_ev_to_nm(params.get('b1', 0), params.get('c1', 0))
            }
            
            self.fitted_params.append(result)
            
            if i % 10 == 0:
                print(f"  Fitted spectrum {i}/{n_spectra}: "
                      f"Peak={result['peakeV1']:.3f} eV, "
                      f"FWHM={result['FWHMeV1']:.3f} eV, "
                      f"R²={r2:.3f}, SNR={snr:.1f}")
        
        # Print filtering statistics
        print(f"\n[Filtering Statistics]")
        print(f"  Total spectra: {stats['total']}")
        print(f"  Rejected (negative R²): {stats['rejected_negative_r2']}")
        print(f"  Rejected (FWHM > {max_width:.3f} eV): {stats['rejected_width']}")
        print(f"  Rejected (R² < {min_r2:.3f}): {stats['rejected_fitting']}")
        print(f"  Accepted: {stats['accepted']}")
    
    def convert_fwhm_ev_to_nm(self, peak_ev: float, fwhm_ev: float) -> float:
        """Convert FWHM from eV to nm using MATLAB formula"""
        if peak_ev <= 0 or fwhm_ev <= 0:
            return 0.0
        
        try:
            fwhm_nm = abs(1239.842 / (peak_ev - 0.5 * fwhm_ev) - 1239.842 / (peak_ev + 0.5 * fwhm_ev))
            return fwhm_nm
        except:
            return 0.0
    
    def identify_cycles(self):
        """Identify experimental cycles or steps"""
        print("\n[Step] Identifying experimental cycles...")
        
        if self.technique == 'CV':
            self.identify_cv_cycles()
        elif self.technique in ['CA', 'CC']:
            raise NotImplementedError("CA/CC analysis not yet implemented")
        else:
            raise ValueError(f"Unknown technique: {self.technique}")
    
    def identify_cv_cycles(self):
        """Identify CV cycle boundaries from voltage trace"""
        # Extract voltages for fitted spectra only
        voltages = np.array([p['voltage'] for p in self.fitted_params])
        
        if len(voltages) < 3:
            print("[warning] Not enough data points for cycle detection")
            return
        
        # Determine initial scan direction
        if voltages[5] - voltages[0] > 0:
            scan_dir = 'positive'
        else:
            scan_dir = 'negative'
        
        print(f"[info] Initial scan direction: {scan_dir}")
        
        # Find turning points
        turning_points = eu.find_cv_turning_points(voltages, scan_dir)
        
        print(f"[info] Found {len(turning_points)} turning points")
        print(f"[info] Turning point indices: {turning_points}")
        
        self.cycle_boundaries = turning_points
        
        n_cycles = len(turning_points) - 1
        print(f"[info] Total cycles detected: {n_cycles}")
    
    def calculate_cycle_averages(self):
        """Calculate cycle-averaged spectral parameters"""
        print("\n[Step] Calculating cycle-averaged parameters...")
        
        if len(self.cycle_boundaries) < 2:
            print("[warning] Not enough cycles for averaging")
            return
        
        n_total_cycles = len(self.cycle_boundaries) - 1
        start_cycle = self.cycle_start - 1
        end_cycle = n_total_cycles - self.cycle_backcut
        
        print(f"[info] Analyzing cycles {start_cycle+1} to {end_cycle}")
        
        first_cycle_len = self.cycle_boundaries[start_cycle + 1] - self.cycle_boundaries[start_cycle]
        
        n_cycles = end_cycle - start_cycle
        
        voltage_cycles = np.zeros((first_cycle_len, n_cycles))
        peak_cycles = np.zeros((first_cycle_len, n_cycles))
        fwhm_cycles = np.zeros((first_cycle_len, n_cycles))
        intensity_cycles = np.zeros((first_cycle_len, n_cycles))
        
        delta_peak_cycles = np.zeros((first_cycle_len, n_cycles))
        delta_fwhm_cycles = np.zeros((first_cycle_len, n_cycles))
        delta_intensity_cycles = np.zeros((first_cycle_len, n_cycles))
        
        for i in range(n_cycles):
            cycle_idx = start_cycle + i
            
            start_idx = self.cycle_boundaries[cycle_idx]
            end_idx = self.cycle_boundaries[cycle_idx + 1]
            
            cycle_len = end_idx - start_idx
            
            if cycle_len == first_cycle_len:
                cycle_slice = slice(start_idx, end_idx)
            elif cycle_len == first_cycle_len + 1:
                cycle_slice = slice(start_idx, end_idx - 1)
            elif cycle_len == first_cycle_len - 1:
                cycle_slice = slice(start_idx, end_idx + 1)
            else:
                print(f"[warning] Cycle {i+1} length mismatch: {cycle_len} vs {first_cycle_len}")
                continue
            
            cycle_params = self.fitted_params[cycle_slice]
            
            if len(cycle_params) == 0:
                continue
            
            voltages = np.array([p['voltage'] for p in cycle_params])
            peaks = np.array([p['peakeV1'] for p in cycle_params])
            fwhms = np.array([p['FWHMeV1'] for p in cycle_params])
            intensities = np.array([p['area1'] for p in cycle_params])
            
            actual_len = len(voltages)
            if actual_len < first_cycle_len:
                voltages = np.pad(voltages, (0, first_cycle_len - actual_len), mode='edge')
                peaks = np.pad(peaks, (0, first_cycle_len - actual_len), mode='edge')
                fwhms = np.pad(fwhms, (0, first_cycle_len - actual_len), mode='edge')
                intensities = np.pad(intensities, (0, first_cycle_len - actual_len), mode='edge')
            
            voltage_cycles[:, i] = voltages[:first_cycle_len]
            peak_cycles[:, i] = peaks[:first_cycle_len]
            fwhm_cycles[:, i] = fwhms[:first_cycle_len]
            intensity_cycles[:, i] = intensities[:first_cycle_len]
            
            if i == 0:
                half_cycle = len(voltages) // 2
                ocp_distances = np.abs(voltages[:half_cycle] - self.ocp)
                self.ocp_index = np.argmin(ocp_distances)
                
                print(f"[info] OCP index: {self.ocp_index}, V={voltages[self.ocp_index]:.3f} V")
            
            delta_peak_cycles[:, i] = peaks[:first_cycle_len] - peaks[self.ocp_index]
            delta_fwhm_cycles[:, i] = fwhms[:first_cycle_len] - fwhms[self.ocp_index]
            delta_intensity_cycles[:, i] = (intensities[:first_cycle_len] - intensities[self.ocp_index]) / intensities[self.ocp_index] * 100
        
        voltage_avg = voltage_cycles.mean(axis=1)
        
        peak_avg = peak_cycles.mean(axis=1)
        peak_se = peak_cycles.std(axis=1, ddof=1) / np.sqrt(n_cycles)
        
        fwhm_avg = fwhm_cycles.mean(axis=1)
        fwhm_se = fwhm_cycles.std(axis=1, ddof=1) / np.sqrt(n_cycles)
        
        delta_peak_avg = delta_peak_cycles.mean(axis=1)
        delta_peak_se = delta_peak_cycles.std(axis=1, ddof=1) / np.sqrt(n_cycles)
        
        delta_fwhm_avg = delta_fwhm_cycles.mean(axis=1)
        delta_fwhm_se = delta_fwhm_cycles.std(axis=1, ddof=1) / np.sqrt(n_cycles)
        
        delta_intensity_avg = delta_intensity_cycles.mean(axis=1)
        delta_intensity_se = delta_intensity_cycles.std(axis=1, ddof=1) / np.sqrt(n_cycles)
        
        self.cycles.append({
            'voltage_avg': voltage_avg,
            'peak_avg': peak_avg,
            'peak_se': peak_se,
            'fwhm_avg': fwhm_avg,
            'fwhm_se': fwhm_se,
            'delta_peak_avg': delta_peak_avg,
            'delta_peak_se': delta_peak_se,
            'delta_fwhm_avg': delta_fwhm_avg,
            'delta_fwhm_se': delta_fwhm_se,
            'delta_intensity_avg': delta_intensity_avg,
            'delta_intensity_se': delta_intensity_se,
            'n_cycles': n_cycles,
            'ocp_index': self.ocp_index
        })
        
        print(f"[info] Calculated averages over {n_cycles} cycles")
    
    def plot_overview(self):
        """Generate overview plot showing all time-series data"""
        print("\n[Step] Generating overview plot...")
        
        output_dir = self.dataset.echem_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        times = np.array([p['time'] for p in self.fitted_params])
        voltages = np.array([p['voltage'] for p in self.fitted_params])
        peaks = np.array([p['peakeV1'] for p in self.fitted_params])
        fwhms = np.array([p['FWHMeV1'] for p in self.fitted_params])
        areas = np.array([p['area1'] for p in self.fitted_params])
        
        if hasattr(self, 'ocp_index') and self.ocp_index < len(areas):
            baseline_area = areas[self.ocp_index]
            intensity_change = (areas - baseline_area) / baseline_area * 100
        else:
            baseline_area = areas[0] if len(areas) > 0 else 1.0
            intensity_change = (areas - baseline_area) / baseline_area * 100
        
        fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
        
        im = axes[0].imshow(self.dataset.spectra.T, aspect='auto', cmap='hot',
                          extent=[times.min(), times.max(),
                                 self.dataset.wavelengths.min(), 
                                 self.dataset.wavelengths.max()],
                          origin='lower')
        axes[0].set_ylabel('Wavelength (nm)', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{self.dataset.sample_name} - EChem Analysis', fontsize=14)
        cbar1 = plt.colorbar(im, ax=axes[0])
        cbar1.set_label('Intensity (a.u.)', fontsize=10)
        
        axes[1].scatter(times, peaks, c='red', s=20, marker='d', linewidth=1.4)
        axes[1].set_ylabel('Resonance (eV)', fontsize=12, fontweight='bold')
        axes[1].tick_params(labelsize=10)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].scatter(times, fwhms, c='red', s=17, marker='d', linewidth=1.4)
        axes[2].set_ylabel('FWHM (eV)', fontsize=12, fontweight='bold')
        axes[2].tick_params(labelsize=10)
        axes[2].grid(True, alpha=0.3)
        
        axes[3].scatter(times, intensity_change, c='red', s=20, marker='d', linewidth=1.4)
        axes[3].set_ylabel('Intensity change (%)', fontsize=12, fontweight='bold')
        axes[3].tick_params(labelsize=10)
        axes[3].grid(True, alpha=0.3)
        
        axes[4].plot(times, voltages, 'r-', linewidth=1.4)
        axes[4].set_ylabel('E (V)', fontsize=12, fontweight='bold')
        axes[4].set_xlabel('Elapsed Time (s)', fontsize=12, fontweight='bold')
        axes[4].tick_params(labelsize=10)
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = output_dir / f"{self.dataset.sample_name}_overview.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[info] Saved overview plot: {output_path}")
    
    def plot_cycles(self):
        """Generate cycle-averaged plots with error bars"""
        if len(self.cycles) == 0:
            print("[warning] No cycle data available for plotting")
            return
        
        print("\n[Step] Generating cycle-averaged plots...")
        
        output_dir = Path(self.args['OUTPUT_DIR'])
        
        cycle = self.cycles[0]
        
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        
        voltage = cycle['voltage_avg']
        
        axes[0].errorbar(voltage, cycle['delta_peak_avg'] * 1000,
                        yerr=cycle['delta_peak_se'] * 1000,
                        fmt='o-', color='blue', linewidth=2, markersize=6,
                        capsize=4, capthick=1.5)
        axes[0].set_ylabel('Δ$E_{Res}$ (meV)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(labelsize=11)
        axes[0].axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        
        axes[1].errorbar(voltage, cycle['delta_fwhm_avg'] * 1000,
                        yerr=cycle['delta_fwhm_se'] * 1000,
                        fmt='o-', color='blue', linewidth=2, markersize=6,
                        capsize=4, capthick=1.5)
        axes[1].set_ylabel('ΔΓ (meV)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(labelsize=11)
        axes[1].axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        
        axes[2].errorbar(voltage, cycle['delta_intensity_avg'],
                        yerr=cycle['delta_intensity_se'],
                        fmt='o-', color='blue', linewidth=2, markersize=6,
                        capsize=4, capthick=1.5)
        axes[2].set_ylabel('Int. Change (%)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('$U$ (V)', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].tick_params(labelsize=11)
        axes[2].axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        
        fig.suptitle(f'{self.dataset.sample_name} - Cycle-Averaged Parameters (n={cycle["n_cycles"]})', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        output_path = output_dir / f"{self.dataset.sample_name}_cycle_averaged.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[info] Saved cycle-averaged plot: {output_path}")
    
    def save_spectra_data(self):
        """
        Export spectral data and fitted parameters to text files and plots
        """
        print("\n[Step] Saving spectral data and plots...")
        
        output_dir = Path(self.args['OUTPUT_DIR']) / "spectra"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different plot types
        plots_dir = output_dir / "plots"
        plots_raw_dir = output_dir / "plots_raw"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plots_raw_dir.mkdir(parents=True, exist_ok=True)

        output_unit = self.args.get('OUTPUT_UNIT', 'eV')
        
        # Save individual spectra with fits
        for i, param in enumerate(self.fitted_params):
            
            # ===== Save text data =====
            if output_unit == 'nm':
                # For nm output: reverse arrays for text file
                data = np.column_stack((
                    self.dataset.wavelengths[::-1], 
                    param['spectrum'][::-1], 
                    param['fit'][::-1]
                ))

                header = f"Wavelength (nm)\tIntensity\tFit\n"
                header += f"# Peak: {param['peaknm1']:.1f} nm, FWHM: {param['FWHMnm1']:.1f} nm\n"
                header += f"# Spectrum {i+1}, Time={param['time']:.2f}s, Voltage={param['voltage']:.3f}V\n"
                header += f"# R²: {param['r2']:.4f}, SNR: {param['snr']:.1f}"

            else:
                # For eV output: use energy array
                energy = 1239.842 / self.dataset.wavelengths
                data = np.column_stack((
                    energy,
                    param['spectrum'],
                    param['fit']
                ))

                header = f"Energy (eV)\tIntensity\tFit\n"
                header += f"# Peak: {param['peakeV1']:.3f} eV, FWHM: {param['FWHMeV1']:.3f} eV\n"
                header += f"# Spectrum {i+1}, Time={param['time']:.2f}s, Voltage={param['voltage']:.3f}V\n"
                header += f"# R²: {param['r2']:.4f}, SNR: {param['snr']:.1f}"
            
            # Save text file
            output_file = output_dir / f"{self.dataset.sample_name}_spectrum_{i+1:04d}.txt"
            np.savetxt(output_file, data, delimiter='\t', header=header, 
                      comments='', fmt='%.6f')
            
            # ===== Generate visualization plots =====
            # Create plot title with time and voltage info
            plot_title = f"Spectrum {i+1} | t={param['time']:.1f}s, V={param['voltage']:.3f}V"
            
            # Plot 1: With fitted curve (plots/)
            plot_path = plots_dir / f"{self.dataset.sample_name}_spectrum_{i+1:04d}.png"
            su.plot_spectrum(
                self.dataset.wavelengths,   # Original wavelength array
                param['spectrum'],          # Original spectrum
                param['fit'],              # Original fit
                plot_title,                # Title with time/voltage
                plot_path,                 # Output path
                dpi=self.args.get("FIG_DPI", 300),  # Resolution
                params=param['params'],    # Fitting parameters
                snr=param['snr'],         # Signal-to-noise ratio
                args=self.args,           # Configuration
                show_fit=True             # Show fit curve
            )
            
            # Plot 2: Raw data only (plots_raw/)
            plot_path_raw = plots_raw_dir / f"{self.dataset.sample_name}_spectrum_{i+1:04d}.png"
            su.plot_spectrum(
                self.dataset.wavelengths,   # Original wavelength array
                param['spectrum'],          # Original spectrum
                param['fit'],              # Original fit (not displayed)
                plot_title,                # Title with time/voltage
                plot_path_raw,            # Output path
                dpi=self.args.get("FIG_DPI", 300),  # Resolution
                params=None,               # No parameter annotations
                snr=None,                  # No SNR annotation
                args=self.args,           # Configuration
                show_fit=False            # Hide fit curve
            )
            
            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == len(self.fitted_params):
                print(f"  Saved {i+1}/{len(self.fitted_params)} spectra (text + 2 plots)")
        
        print(f"[info] Saved {len(self.fitted_params)} spectral files to {output_dir}")
        print(f"[info] Saved {len(self.fitted_params)} plots with fit to {plots_dir}")
        print(f"[info] Saved {len(self.fitted_params)} raw-only plots to {plots_raw_dir}")
        
        # Save cycle-averaged data using utility function
        if len(self.cycles) > 0:
            eu.save_echem_cycle_data(self.cycles, output_dir, self.dataset.sample_name)

    
    def dump_results(self):
        """Save all analysis results to pickle file"""
        output_path = Path(self.args['OUTPUT_DIR']) / f"{self.dataset.sample_name}_echem_results.pkl"
        
        payload = {
            'sample': self.dataset.sample_name,
            'technique': self.technique,
            'wavelengths': self.dataset.wavelengths,
            'spectra': self.dataset.spectra,
            'spec_times': self.dataset.spec_times,
            'voltages': self.dataset.voltages,
            'chi_data': self.dataset.chi_data,
            'fitted_params': self.fitted_params,
            'cycles': self.cycles,
            'cycle_boundaries': self.cycle_boundaries if hasattr(self, 'cycle_boundaries') else [],
            'rejected_fits': self.rejected_fits,
            'config': self.args,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_path, 'wb') as f:
            pkl.dump(payload, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        print(f"\n[info] Results saved to: {output_path}")
    
    def print_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*60)
        print("ECHEM ANALYSIS SUMMARY")
        print("="*60)
        print(f"Sample: {self.dataset.sample_name}")
        print(f"Technique: {self.technique}")
        print(f"Total spectra: {self.dataset.spectra.shape[0]}")
        print(f"Successfully fitted: {len(self.fitted_params)}")
        print(f"Rejected fits: {len(self.rejected_fits)}")
        
        # Rejection breakdown
        if len(self.rejected_fits) > 0:
            negative_r2 = len([r for r in self.rejected_fits if 'negative' in r['reason'].lower()])
            width_rejected = len([r for r in self.rejected_fits if 'FWHM' in r['reason']])
            low_r2 = len([r for r in self.rejected_fits if 'too low' in r['reason']])
            
            print(f"\n[Rejection Breakdown]")
            print(f"  Negative R²: {negative_r2}")
            print(f"  FWHM too large: {width_rejected}")
            print(f"  R² too low: {low_r2}")
        
        if len(self.fitted_params) > 0:
            peaks = [p['peakeV1'] for p in self.fitted_params]
            fwhms = [p['FWHMeV1'] for p in self.fitted_params]
            snrs = [p['snr'] for p in self.fitted_params]
            r2s = [p['r2'] for p in self.fitted_params]
            
            print(f"\nResonance energy: {np.mean(peaks):.4f} ± {np.std(peaks):.4f} eV")
            print(f"  Range: {min(peaks):.4f} - {max(peaks):.4f} eV")
            
            print(f"\nFWHM: {np.mean(fwhms):.4f} ± {np.std(fwhms):.4f} eV")
            print(f"  Range: {min(fwhms):.4f} - {max(fwhms):.4f} eV")
            
            print(f"\nS/N ratio: {np.mean(snrs):.1f} ± {np.std(snrs):.1f}")
            print(f"  Range: {min(snrs):.1f} - {max(snrs):.1f}")
            
            print(f"\nR² values: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
            print(f"  Range: {min(r2s):.4f} - {max(r2s):.4f}")
        
        if hasattr(self, 'cycle_boundaries'):
            print(f"\nCycles detected: {len(self.cycle_boundaries) - 1}")
            print(f"Cycles analyzed: {len(self.cycles)}")
        
        print("="*60)