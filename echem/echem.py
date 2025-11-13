import os
import sys
import gc
import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

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
        self.plot_peak_separated_heatmaps()
        self.save_spectra_data()
        self.dump_results()
        self.print_summary()

    def fit_all_spectra(self):
        """
        Fit models to all time-point spectra using BATCH PROCESSING
        
        Key changes:
        - Fit all spectra in a single batch
        - Use PyTorch-based batch fitting
        - Apply quality filters after batch fitting
        - Temporal information is preserved through batch indexing
        """
        print("\n[Step] Fitting all time-point spectra (BATCH MODE)...")
        
        fitting_model = self.args.get('FITTING_MODEL')
        print(f"[info] Using fitting model: {fitting_model}")
        
        n_spectra = self.dataset.spectra.shape[0]
        wavelengths = self.dataset.wavelengths
        
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
        
        # ========================================
        # STEP 1: Validate and prepare spectra
        # ========================================
        print("[info] Step 1: Validating spectra...")
        
        valid_indices = []
        valid_spectra = []
        
        for i in range(n_spectra):
            spectrum = self.dataset.spectra[i, :]
            
            # Skip invalid spectra
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
            
            valid_indices.append(i)
            valid_spectra.append(spectrum)
        
        if len(valid_spectra) == 0:
            print("[warning] No valid spectra to fit")
            return
        
        # Convert to numpy array
        valid_spectra = np.array(valid_spectra)  # shape: (N_valid, N_wavelengths)
        print(f"[info] {len(valid_spectra)} valid spectra prepared for batch fitting")
        
        # ========================================
        # STEP 2: Batch fitting
        # ========================================
        print("[info] Step 2: Batch fitting all valid spectra...")
        
        use_gpu = self.args.get('USE_GPU', False)
        
        if fitting_model == 'fano':
            fitted_spectra, params_list, r2_scores = su.fit_fano_batch(
                self.args, valid_spectra, wavelengths, use_gpu=use_gpu
            )
        elif fitting_model == 'lorentzian':
            fitted_spectra, params_list, r2_scores = su.fit_lorentz_batch(
                self.args, valid_spectra, wavelengths, use_gpu=use_gpu
            )
        else:
            raise ValueError(f"[error] Unknown fitting model: {fitting_model}")
        
        print(f"[info] Batch fitting completed!")
        
        # ========================================
        # STEP 3: Process results and apply quality filters
        # ========================================
        print("[info] Step 3: Applying quality filters...")
        
        for idx_in_batch, original_idx in enumerate(valid_indices):
            spectrum = valid_spectra[idx_in_batch]
            fitted = fitted_spectra[idx_in_batch]
            params = params_list[idx_in_batch]
            r2 = r2_scores[idx_in_batch]
            
            # Extract parameters based on model
            if fitting_model == 'fano':
                fwhm_nm = params.get('bright1_gamma', 0)
                peak_nm = params.get('bright1_lambda', 0)
            elif fitting_model == 'lorentzian':
                fwhm_nm = params.get('c1', 0)
                peak_nm = params.get('b1', 0)
            
            # Convert nm → eV
            if peak_nm > 0 and fwhm_nm > 0:
                fwhm_ev = abs(1239.842/(peak_nm - fwhm_nm/2) - 1239.842/(peak_nm + fwhm_nm/2))
            else:
                fwhm_ev = 0
            
            # Quality Filter 1: Negative R²
            if r2 < 0:
                print(f"  Spectrum {original_idx}: Rejected - R² negative ({r2:.3f})")
                stats['rejected_negative_r2'] += 1
                self.rejected_fits.append({
                    'index': original_idx,
                    'time': self.dataset.spec_times[original_idx],
                    'voltage': self.dataset.voltages[original_idx],
                    'reason': f'R² negative ({r2:.3f})',
                    'fwhm': fwhm_ev,
                    'r2': r2
                })
            
            # Quality Filter 2: Width limit (in eV)
            if fwhm_ev > max_width:
                print(f"  Spectrum {original_idx}: Rejected - Width too large ({fwhm_ev:.3f} eV > {max_width} eV)")
                stats['rejected_width'] += 1
                self.rejected_fits.append({
                    'index': original_idx,
                    'time': self.dataset.spec_times[original_idx],
                    'voltage': self.dataset.voltages[original_idx],
                    'reason': f'Width too large ({fwhm_ev:.3f} eV)',
                    'fwhm': fwhm_ev,
                    'r2': r2
                })
            
            # Quality Filter 3: R² threshold
            if r2 < min_r2:
                print(f"  Spectrum {original_idx}: Rejected - R² too low ({r2:.3f} < {min_r2})")
                stats['rejected_fitting'] += 1
                self.rejected_fits.append({
                    'index': original_idx,
                    'time': self.dataset.spec_times[original_idx],
                    'voltage': self.dataset.voltages[original_idx],
                    'reason': f'R² too low ({r2:.3f})',
                    'fwhm': fwhm_ev,
                    'r2': r2
                })

            else:
                # Quality check 통과
                stats['accepted'] += 1
            
            # Compute SNR
            peak_intensity = spectrum.max()
            noise_region = spectrum[:int(len(spectrum) * 0.1)]
            noise_std = np.std(noise_region)
            snr = peak_intensity / noise_std if noise_std > 0 else 0
            
            # Convert all parameters to eV
            if fitting_model == 'lorentzian':
                num_peaks = self.args.get('NUM_PEAKS', 1)
                for peak_idx in range(1, num_peaks + 1):
                    peak_nm = params.get(f'b{peak_idx}', 0)
                    fwhm_nm = params.get(f'c{peak_idx}', 0)
                    
                    if peak_nm > 0:
                        params[f'peakeV{peak_idx}'] = 1239.842 / peak_nm
                        if fwhm_nm > 0:
                            params[f'FWHMeV{peak_idx}'] = abs(
                                1239.842/(peak_nm - fwhm_nm/2) - 1239.842/(peak_nm + fwhm_nm/2)
                            )
                        else:
                            params[f'FWHMeV{peak_idx}'] = 0
                    else:
                        params[f'peakeV{peak_idx}'] = 0
                        params[f'FWHMeV{peak_idx}'] = 0
            
            elif fitting_model == 'fano':
                num_bright = self.args.get('NUM_BRIGHT_MODES', 1)
                for mode_idx in range(1, num_bright + 1):
                    resonance_nm = params.get(f'bright{mode_idx}_lambda', 0)
                    gamma_nm = params.get(f'bright{mode_idx}_gamma', 0)
                    
                    if resonance_nm > 0:
                        params[f'resonanceeV{mode_idx}'] = 1239.842 / resonance_nm
                        params[f'resonancenm{mode_idx}'] = resonance_nm
                        if gamma_nm > 0:
                            params[f'gammaeV{mode_idx}'] = abs(
                                1239.842/(resonance_nm - gamma_nm/2) - 1239.842/(resonance_nm + gamma_nm/2)
                            )
                            params[f'gammanm{mode_idx}'] = gamma_nm
                        else:
                            params[f'gammaeV{mode_idx}'] = 0
                            params[f'gammanm{mode_idx}'] = 0
                    else:
                        params[f'resonanceeV{mode_idx}'] = 0
                        params[f'resonancenm{mode_idx}'] = 0
                        params[f'gammaeV{mode_idx}'] = 0
                        params[f'gammanm{mode_idx}'] = 0
            
            # Store additional metadata
            params['time'] = self.dataset.spec_times[original_idx]
            params['voltage'] = self.dataset.voltages[original_idx]
            params['r2'] = r2
            params['snr'] = snr
            params['spectrum'] = spectrum
            params['fit'] = fitted
            
            # Accept this fit
            stats['accepted'] += 1

            # 🔧 index와 메타데이터 추가
            params['index'] = original_idx
            params['time'] = self.dataset.spec_times[original_idx]
            params['voltage'] = self.dataset.voltages[original_idx]
            params['r2'] = r2

            self.fitted_params.append(params)
            
            if (original_idx + 1) % 100 == 0:
                print(f"  Processed {original_idx + 1}/{n_spectra} spectra...")
        
        # ========================================
        # Print statistics
        # ========================================
        print("\n" + "="*60)
        print("FITTING STATISTICS")
        print("="*60)
        print(f"Total spectra: {stats['total']}")
        print(f"Good quality fits: {stats['accepted']}")
        print(f"Low quality (negative R²): {stats['rejected_negative_r2']}")
        print(f"Low quality (width): {stats['rejected_width']}")
        print(f"Low quality (R² threshold): {stats['rejected_fitting']}")
        print(f"\nAll {len(self.fitted_params)} fitted spectra will be used for analysis")
        print("="*60)
    
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
            self.identify_ca_steps()
        else:
            print(f"[warning] Unknown technique: {self.technique}")

    def identify_cv_cycles(self):
        """
        Identify CV cycles by detecting voltage direction changes
        
        In CV, voltage sweeps back and forth. A cycle is defined as:
        - One complete forward-backward sweep
        - Detected by voltage direction changes
        """
        voltages = self.dataset.voltages
        
        # Find peaks and valleys (voltage direction changes)
        peaks, _ = find_peaks(voltages, distance=10)
        valleys, _ = find_peaks(-voltages, distance=10)
        
        # Combine and sort turning points
        turning_points = np.sort(np.concatenate([peaks, valleys]))
        
        if len(turning_points) == 0:
            print("[warning] No CV cycles detected")
            return
        
        # Cycle boundaries: start, turning points, end
        self.cycle_boundaries = [0] + list(turning_points) + [len(voltages)-1]
        
        print(f"[info] Detected {len(self.cycle_boundaries)-1} CV half-cycles")
        print(f"[info] Turning points at indices: {turning_points}")

    def identify_ca_steps(self):
        """
        Identify CA/CC steps by detecting voltage plateaus
        
        In CA, voltage is held constant at different levels.
        Steps are detected by finding voltage change points.
        """
        voltages = self.dataset.voltages
        
        # Detect voltage changes (derivative)
        voltage_diff = np.abs(np.diff(voltages))
        threshold = np.std(voltage_diff) * 3
        
        # Find step boundaries
        step_changes = np.where(voltage_diff > threshold)[0]
        
        if len(step_changes) == 0:
            print("[warning] No CA/CC steps detected")
            return
        
        self.cycle_boundaries = [0] + list(step_changes) + [len(voltages)-1]
        
        print(f"[info] Detected {len(self.cycle_boundaries)-1} potential steps")

    def calculate_cycle_averages(self):
        """
        Calculate cycle-averaged spectral parameters
        
        For each cycle:
        - Average resonance energy, FWHM, intensity
        - Track voltage range
        - Calculate relative changes from OCP
        """
        if len(self.cycle_boundaries) < 2:
            print("[warning] Not enough cycles for averaging")
            return
        
        print("\n[Step] Calculating cycle averages...")
        
        fitting_model = self.args.get('FITTING_MODEL', 'lorentzian')
        
        # Determine how many cycles to analyze
        n_cycles = len(self.cycle_boundaries) - 1
        cycle_start_idx = self.cycle_start - 1  # Convert to 0-indexed
        cycle_end_idx = n_cycles - self.cycle_backcut
        
        if cycle_start_idx < 0:
            cycle_start_idx = 0
        if cycle_end_idx > n_cycles:
            cycle_end_idx = n_cycles
        
        print(f"[info] Analyzing cycles {self.cycle_start} to {cycle_end_idx} "
              f"(total: {n_cycles}, start: {self.cycle_start}, backcut: {self.cycle_backcut})")
        
        for cycle_idx in range(cycle_start_idx, cycle_end_idx):
            start_idx = self.cycle_boundaries[cycle_idx]
            end_idx = self.cycle_boundaries[cycle_idx + 1]
            
            # Get parameters for this cycle
            cycle_params = [p for p in self.fitted_params 
                          if start_idx <= p['index'] < end_idx]
            
            if len(cycle_params) == 0:
                print(f"  Cycle {cycle_idx+1}: No valid spectra")
                continue
            
            # Extract voltage for this cycle
            cycle_voltage = self.dataset.voltages[start_idx:end_idx]
            
            # ✅ 수정: 모델에 따라 다른 키 사용
            if fitting_model == 'fano':
                # Fano 모델: resonanceeV1, gammaeV1, c1 사용
                peak_key = 'resonanceeV1'
                fwhm_key = 'gammaeV1'
                area_key = 'c1'
            else:
                # Lorentzian 모델: peakeV1, FWHMeV1, area1 사용
                peak_key = 'peakeV1'
                fwhm_key = 'FWHMeV1'
                area_key = 'area1'
            
            # Calculate relative changes from OCP
            ocp_idx, delta_peaks = eu.compute_ocp_baseline(cycle_voltage, cycle_params, self.ocp, peak_key)
            
            _, delta_fwhm = eu.compute_ocp_baseline(cycle_voltage, cycle_params, self.ocp, fwhm_key)
            
            # Store cycle data
            cycle_data = {
                'cycle_number': cycle_idx + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'n_spectra': len(cycle_params),
                'voltage': cycle_voltage,
                'ocp_idx': ocp_idx,
                'delta_resonance': delta_peaks,
                'delta_fwhm': delta_fwhm,
                'params': cycle_params
            }
            
            self.cycles.append(cycle_data)
            
            print(f"  Cycle {cycle_idx+1}: {len(cycle_params)} spectra, "
                  f"V range: {cycle_voltage.min():.3f} to {cycle_voltage.max():.3f} V")

    def plot_peak_separated_heatmaps(self):
        """Generate peak-separated spectral heatmaps with fitted parameters"""
        print("\n[Step] Generating peak-separated heatmaps with parameters...")
        
        output_dir = self.dataset.echem_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data_dir = self.dataset.echem_output_dir / "plot_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        fitting_model = self.args.get('FITTING_MODEL', 'lorentzian')
        
        # ✅ 수정: 모델에 따라 피크 개수 다르게 설정
        if fitting_model == 'fano':
            num_peaks = self.args.get('NUM_BRIGHT_MODES', 1)
        else:
            num_peaks = self.args.get('NUM_PEAKS', 1)
        
        # ========================================
        # 각 피크별 heatmap + 파라미터 추적 그래프
        # ========================================
        for peak_idx in range(num_peaks):
            print(f"  Creating heatmap for Peak {peak_idx+1}...")
            
            peak_wl_min = self.dataset.wavelengths.min()
            peak_wl_max = self.dataset.wavelengths.max()
            
            mask = (self.dataset.wavelengths >= peak_wl_min) & (self.dataset.wavelengths <= peak_wl_max)
            
            if not np.any(mask):
                print(f"  Peak {peak_idx+1}: No data in range")
                continue
            
            wl_subset = self.dataset.wavelengths[mask]
            
            # nm → eV
            energy_subset = 1239.842 / wl_subset
            sort_idx_sub = np.argsort(energy_subset)
            energy_subset = energy_subset[sort_idx_sub]
            
            n_spectra = len(self.fitted_params)
            n_wavelengths = len(self.dataset.wavelengths)
            
            # 피크별 재구성 스펙트럼
            peak_spectra_reconstructed = np.zeros((n_spectra, n_wavelengths))
            
            # ✅✅✅ 수정: 피팅 모델에 따라 다른 재구성 함수 사용
            if fitting_model == 'lorentzian':
                # Lorentzian 모델 재구성
                def lorentz_single_peak(x, a, b, c):
                    """Single Lorentzian peak in wavelength space"""
                    return (2*a/np.pi) * (c / (4*(x-b)**2 + c**2))
                
                for i, param in enumerate(self.fitted_params):
                    a = param.get(f'a{peak_idx+1}', 0)
                    b = param.get(f'b{peak_idx+1}', 0)
                    c = param.get(f'c{peak_idx+1}', 0)
                    
                    if b > 0:  # Valid peak
                        peak_spectra_reconstructed[i, :] = lorentz_single_peak(self.dataset.wavelengths, a, b, c)
            
            elif fitting_model == 'fano':
                # Fano 모델 재구성 - Bright mode만
                def fano_single_bright(x, c, lam, gamma):
                    """Single Fano bright mode in wavelength space"""
                    A = c * (gamma/2) / (x - lam + 1j*gamma/2)
                    I = np.abs(A)**2
                    return I
                
                for i, param in enumerate(self.fitted_params):
                    c = param.get(f'bright{peak_idx+1}_c', 0)
                    lam = param.get(f'bright{peak_idx+1}_lambda', 0)
                    gamma = param.get(f'bright{peak_idx+1}_gamma', 0)
                    
                    if lam > 0 and gamma > 0:  # Valid bright mode
                        peak_spectra_reconstructed[i, :] = fano_single_bright(self.dataset.wavelengths, c, lam, gamma)
            
            else:
                raise ValueError(f"Unknown fitting model: {fitting_model}")
            
            # ✅ 디버그: 재구성 스펙트럼 강도 확인
            max_intensity = peak_spectra_reconstructed.max()
            mean_intensity = peak_spectra_reconstructed.mean()
            print(f"    Peak {peak_idx+1} reconstructed: max={max_intensity:.6f}, mean={mean_intensity:.6f}")
            
            if max_intensity < 1e-8:
                print(f"    ⚠️ WARNING: Peak {peak_idx+1} has very low intensity!")
            
            # 재구성 스펙트럼을 에너지 축으로 올바르게 변환
            energy_full = 1239.842 / self.dataset.wavelengths
            sort_idx_full = np.argsort(energy_full)
            
            peak_spectra_sorted = peak_spectra_reconstructed[:, sort_idx_full]
            
            energy_full_sorted = energy_full[sort_idx_full]
            mask_subset = (energy_full_sorted >= energy_subset.min()) & (energy_full_sorted <= energy_subset.max())
            
            peak_spectra_ev = peak_spectra_sorted[:, mask_subset]
            
            # ========================================
            # Plot: 5개 subplot
            # ========================================
            fig, axes = plt.subplots(5, 1, figsize=(12, 15),
                                    gridspec_kw={'height_ratios': [1.5, 0.5, 1, 1, 1]})
            
            # Subplot 1: Heatmap (피크별 재구성)
            ax_heat = axes[0]
            im = ax_heat.imshow(peak_spectra_ev.T, aspect='auto', cmap='hot',
                               extent=[self.dataset.spec_times.min(), self.dataset.spec_times.max(),
                                      energy_subset.min(), energy_subset.max()],
                               origin='lower')
            
            ax_heat.set_ylabel('Energy (eV)', fontsize=12, fontweight='bold')
            
            # ✅ 타이틀에 모델명 추가
            if fitting_model == 'fano':
                title_suffix = f'(Bright Mode {peak_idx+1})'
            else:
                title_suffix = f'(Peak {peak_idx+1})'
            ax_heat.set_title(f'{self.dataset.sample_name} - {title_suffix}',
                             fontsize=14, pad=20)
            ax_heat.tick_params(labelsize=10)
            ax_heat.set_xticklabels([])
            ax_heat.set_ylim(energy_subset.min(), energy_subset.max())
            
            cbar = plt.colorbar(im, ax=ax_heat, location='top', pad=0.02, fraction=0.05, shrink=0.3, anchor=(1.0, 0.0))
            cbar.set_label('Intensity (a.u.)', fontsize=10)
            
            # Subplot 2: Voltage trace
            ax_volt = axes[1]
            ax_volt.plot(self.dataset.spec_times, self.dataset.voltages, 'r-', linewidth=1.5)
            ax_volt.set_ylabel('Voltage (V)', fontsize=10, fontweight='bold')
            ax_volt.tick_params(labelsize=9)
            ax_volt.grid(True, alpha=0.3)
            ax_volt.set_xticklabels([])
            ax_volt.set_xlim(self.dataset.spec_times.min(), self.dataset.spec_times.max())
            
            # Subplot 3: Resonance
            ax_res = axes[2]
            
            # ✅ 수정: 모델에 따라 다른 키 사용
            if fitting_model == 'fano':
                peak_key = f'resonanceeV{peak_idx+1}'  # Fano는 resonance
            else:
                peak_key = f'peakeV{peak_idx+1}'      # Lorentzian은 peak
            
            peaks = [p.get(peak_key, np.nan) for p in self.fitted_params]
            times = [p.get('time', np.nan) for p in self.fitted_params]
            
            # ✅ 수정: valid_mask를 peaks와 times 둘 다 체크
            valid_mask = (~np.isnan(peaks)) & (~np.isnan(times))
            times_valid = [t for t, v in zip(times, valid_mask) if v]
            peaks_valid = [pk for pk, v in zip(peaks, valid_mask) if v]
            
            if len(peaks_valid) > 0:
                ax_res.plot(times_valid, peaks_valid, 'b-', linewidth=1.5, marker='o', markersize=3)
            else:
                print(f"    ⚠️ WARNING: No valid resonance data for peak {peak_idx+1}")
            
            if fitting_model == 'fano':
                ylabel = f'Bright {peak_idx+1}\nResonance (eV)'
            else:
                ylabel = f'Peak {peak_idx+1}\nResonance (eV)'
            ax_res.set_ylabel(ylabel, fontsize=10, fontweight='bold')
            ax_res.tick_params(labelsize=9)
            ax_res.grid(True, alpha=0.3)
            ax_res.set_xticklabels([])
            ax_res.set_xlim(self.dataset.spec_times.min(), self.dataset.spec_times.max())
            
            # Subplot 4: FWHM/Gamma
            ax_fwhm = axes[3]
            
            # ✅✅✅ 수정: 모델에 따라 다른 키 사용
            if fitting_model == 'fano':
                fwhm_key = f'gammaeV{peak_idx+1}'   # Fano의 경우 gamma (linewidth)
            else:
                fwhm_key = f'FWHMeV{peak_idx+1}'     # Lorentzian의 경우
            
            fwhms = [p.get(fwhm_key, np.nan) for p in self.fitted_params]
            times = [p.get('time', np.nan) for p in self.fitted_params]
            
            valid_mask = (~np.isnan(fwhms)) & (~np.isnan(times))
            times_valid = [t for t, v in zip(times, valid_mask) if v]
            fwhms_valid = [f for f, v in zip(fwhms, valid_mask) if v]
            
            if len(fwhms_valid) > 0:
                ax_fwhm.plot(times_valid, fwhms_valid, 'g-', linewidth=1.5, marker='s', markersize=3)
            else:
                print(f"    ⚠️ WARNING: No valid gamma/FWHM data for peak {peak_idx+1}")
            
            if fitting_model == 'fano':
                ylabel = f'Bright {peak_idx+1}\nγ (eV)'
            else:
                ylabel = f'Peak {peak_idx+1}\nFWHM (eV)'
            ax_fwhm.set_ylabel(ylabel, fontsize=10, fontweight='bold')
            ax_fwhm.tick_params(labelsize=9)
            ax_fwhm.grid(True, alpha=0.3)
            ax_fwhm.set_xticklabels([])
            ax_fwhm.set_xlim(self.dataset.spec_times.min(), self.dataset.spec_times.max())
            
            # Subplot 5: Intensity Change (%)
            ax_int = axes[4]
            
            # ✅✅✅ 수정: 모델에 따라 다른 키 사용
            if fitting_model == 'fano':
                # Fano의 경우 c 파라미터 (coupling strength)를 intensity로 사용
                area_key = f'c{peak_idx+1}'
            else:
                # Lorentzian의 경우 area 사용
                area_key = f'area{peak_idx+1}'
            
            areas = [p.get(area_key, 0) for p in self.fitted_params]
            times_all = [p.get('time', np.nan) for p in self.fitted_params]
            
            # ✅ 수정: nan 제거
            valid_mask = (~np.isnan(times_all)) & (np.array(areas) != 0)
            times_valid = [t for t, v in zip(times_all, valid_mask) if v]
            areas_valid = [a for a, v in zip(areas, valid_mask) if v]
            
            if len(areas_valid) > 0 and areas_valid[0] != 0:
                baseline_area = areas_valid[0]
                intensity_changes = [(a - baseline_area) / baseline_area * 100 for a in areas_valid]
                ax_int.plot(times_valid, intensity_changes, 'm-', linewidth=1.5, marker='^', markersize=3)
            else:
                print(f"    ⚠️ WARNING: No valid intensity data for peak {peak_idx+1}")
            
            if fitting_model == 'fano':
                ylabel = f'Bright {peak_idx+1}\nCoupling Δ (%)'
            else:
                ylabel = f'Peak {peak_idx+1}\nIntensity Δ (%)'
            ax_int.set_ylabel(ylabel, fontsize=10, fontweight='bold')
            ax_int.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
            ax_int.tick_params(labelsize=9)
            ax_int.grid(True, alpha=0.3)
            ax_int.set_xlim(self.dataset.spec_times.min(), self.dataset.spec_times.max())
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{self.dataset.sample_name}_spectral_heatmap_peak{peak_idx+1}_eV.png", dpi=150)
            plt.close()
            
            # ========================================
            # 데이터 저장
            # ========================================
            param_data = []
            for i, p in enumerate(self.fitted_params):
                row = [
                    p.get('time'),
                    self.dataset.voltages[i] if i < len(self.dataset.voltages) else np.nan,
                    peaks[i] if i < len(peaks) else np.nan,
                    fwhms[i] if i < len(fwhms) else np.nan,
                    areas[i] if i < len(areas) else 0
                ]
                param_data.append(row)
            
            param_data = np.array(param_data)
            
            # ✅ 헤더도 모델에 따라 다르게
            if fitting_model == 'fano':
                header = f"Time(s)\tVoltage(V)\tResonance(eV)\tGamma(eV)\tCoupling(a.u.)\nBright_{peak_idx+1}"
            else:
                header = f"Time(s)\tVoltage(V)\tResonance(eV)\tFWHM(eV)\tArea(a.u.)\nPeak_{peak_idx+1}"
            
            np.savetxt(data_dir / f"{self.dataset.sample_name}_peak{peak_idx+1}_parameters.txt",
                       param_data,
                       header=header,
                       delimiter='\t', fmt='%.6f')
            
            print(f"  Saved peak {peak_idx+1} heatmap with parameters")
        
        # ========================================
        # ✅✅✅ 추가: Dark modes 그래프 생성 (Fano 모델인 경우)
        # ========================================
        if fitting_model == 'fano':
            num_dark = self.args.get('NUM_DARK_MODES', 0)
            
            for dark_idx in range(num_dark):
                print(f"  Creating heatmap for Dark Mode {dark_idx+1}...")
                
                # Full 범위 사용
                peak_wl_min = self.dataset.wavelengths.min()
                peak_wl_max = self.dataset.wavelengths.max()
                
                mask = (self.dataset.wavelengths >= peak_wl_min) & (self.dataset.wavelengths <= peak_wl_max)
                
                if not np.any(mask):
                    print(f"  Dark Mode {dark_idx+1}: No data in range")
                    continue
                
                wl_subset = self.dataset.wavelengths[mask]
                
                # nm → eV
                energy_subset = 1239.842 / wl_subset
                sort_idx_sub = np.argsort(energy_subset)
                energy_subset = energy_subset[sort_idx_sub]
                
                # Dark mode 재구성 스펙트럼
                n_spectra = len(self.fitted_params)
                n_wavelengths = len(self.dataset.wavelengths)
                dark_spectra_reconstructed = np.zeros((n_spectra, n_wavelengths))
                
                def fano_single_dark(x, d, lam, Gamma, theta):
                    """Single Fano dark mode in wavelength space"""
                    A = d * np.exp(1j * theta) * (Gamma/2) / (x - lam + 1j*Gamma/2)
                    I = np.abs(A)**2
                    return I
                
                for i, param in enumerate(self.fitted_params):
                    d = param.get(f'dark{dark_idx+1}_d', 0)
                    lam = param.get(f'dark{dark_idx+1}_lambda', 0)
                    Gamma = param.get(f'dark{dark_idx+1}_Gamma', 0)
                    theta = param.get(f'dark{dark_idx+1}_theta', 0)
                    
                    if lam > 0 and Gamma > 0:  # Valid dark mode
                        dark_spectra_reconstructed[i, :] = fano_single_dark(self.dataset.wavelengths, d, lam, Gamma, theta)
                
                max_intensity = dark_spectra_reconstructed.max()
                mean_intensity = dark_spectra_reconstructed.mean()
                print(f"    Dark {dark_idx+1} reconstructed: max={max_intensity:.6f}, mean={mean_intensity:.6f}")
                
                if max_intensity < 1e-8:
                    print(f"    ⚠️ WARNING: Dark Mode {dark_idx+1} has very low intensity!")
                
                # 재구성 스펙트럼을 에너지 축으로 변환
                energy_full = 1239.842 / self.dataset.wavelengths
                sort_idx_full = np.argsort(energy_full)
                
                dark_spectra_sorted = dark_spectra_reconstructed[:, sort_idx_full]
                energy_full_sorted = energy_full[sort_idx_full]
                mask_subset = (energy_full_sorted >= energy_subset.min()) & (energy_full_sorted <= energy_subset.max())
                
                dark_spectra_ev = dark_spectra_sorted[:, mask_subset]
                
                # Plot: 5개 subplot
                fig, axes = plt.subplots(5, 1, figsize=(12, 15),
                                        gridspec_kw={'height_ratios': [1.5, 0.5, 1, 1, 1]})
                
                # Subplot 1: Heatmap
                ax_heat = axes[0]
                im = ax_heat.imshow(dark_spectra_ev.T, aspect='auto', cmap='hot',
                                   extent=[self.dataset.spec_times.min(), self.dataset.spec_times.max(),
                                          energy_subset.min(), energy_subset.max()],
                                   origin='lower')
                
                ax_heat.set_ylabel('Energy (eV)', fontsize=12, fontweight='bold')
                ax_heat.set_title(f'{self.dataset.sample_name} - (Dark Mode {dark_idx+1})',
                                 fontsize=14, pad=20)
                ax_heat.tick_params(labelsize=10)
                ax_heat.set_xticklabels([])
                ax_heat.set_ylim(energy_subset.min(), energy_subset.max())
                
                cbar = plt.colorbar(im, ax=ax_heat, location='top', pad=0.02, fraction=0.05, shrink=0.3, anchor=(1.0, 0.0))
                cbar.set_label('Intensity (a.u.)', fontsize=10)
                
                # Subplot 2: Voltage
                ax_volt = axes[1]
                ax_volt.plot(self.dataset.spec_times, self.dataset.voltages, 'r-', linewidth=1.5)
                ax_volt.set_ylabel('Voltage (V)', fontsize=10, fontweight='bold')
                ax_volt.tick_params(labelsize=9)
                ax_volt.grid(True, alpha=0.3)
                ax_volt.set_xticklabels([])
                ax_volt.set_xlim(self.dataset.spec_times.min(), self.dataset.spec_times.max())
                
                # Subplot 3: Resonance
                ax_res = axes[2]

                resonances = []
                times = []
                for p in self.fitted_params:
                    lam_nm = p.get(f'dark{dark_idx+1}_lambda', np.nan)
                    time_val = p.get('time', np.nan)
                    
                    if not np.isnan(lam_nm) and lam_nm > 0:
                        resonances.append(1239.842 / lam_nm)  # nm → eV
                    else:
                        resonances.append(np.nan)
                    
                    times.append(time_val)
                
                valid_mask = (~np.isnan(resonances)) & (~np.isnan(times))
                times_valid = [t for t, v in zip(times, valid_mask) if v]
                resonances_valid = [r for r, v in zip(resonances, valid_mask) if v]
                
                if len(resonances_valid) > 0:
                    ax_res.plot(times_valid, resonances_valid, 'b-', linewidth=1.5, marker='o', markersize=3)
                else:
                    print(f"    ⚠️ WARNING: No valid resonance data for dark mode {dark_idx+1}")
                
                ax_res.set_ylabel(f'Dark {dark_idx+1}\nResonance (eV)', fontsize=10, fontweight='bold')
                ax_res.tick_params(labelsize=9)
                ax_res.grid(True, alpha=0.3)
                ax_res.set_xticklabels([])
                ax_res.set_xlim(self.dataset.spec_times.min(), self.dataset.spec_times.max())
                
                # Subplot 4: Linewidth (Gamma)
                ax_gamma = axes[3]
                
                gammas = []
                times = []
                for p in self.fitted_params:
                    Gamma_nm = p.get(f'dark{dark_idx+1}_Gamma', np.nan)
                    lam_nm = p.get(f'dark{dark_idx+1}_lambda', 700)
                    time_val = p.get('time', np.nan)
                    
                    if not np.isnan(Gamma_nm) and Gamma_nm > 0 and lam_nm > 0:
                        # Γ를 eV로 변환: Γ_eV ≈ (1239.842 / λ²) × Γ_nm
                        gamma_eV = (1239.842 / (lam_nm ** 2)) * Gamma_nm
                        gammas.append(gamma_eV)
                    else:
                        gammas.append(np.nan)
                    
                    times.append(time_val)

                valid_mask = (~np.isnan(gammas)) & (~np.isnan(times))
                times_valid = [t for t, v in zip(times, valid_mask) if v]
                gammas_valid = [g for g, v in zip(gammas, valid_mask) if v]
                
                if len(gammas_valid) > 0:
                    ax_gamma.plot(times_valid, gammas_valid, 'g-', linewidth=1.5, marker='s', markersize=3)
                else:
                    print(f"    ⚠️ WARNING: No valid gamma data for dark mode {dark_idx+1}")
                
                ax_gamma.set_ylabel(f'Dark {dark_idx+1}\nΓ (eV)', fontsize=10, fontweight='bold')
                ax_gamma.tick_params(labelsize=9)
                ax_gamma.grid(True, alpha=0.3)
                ax_gamma.set_xticklabels([])
                ax_gamma.set_xlim(self.dataset.spec_times.min(), self.dataset.spec_times.max())
                
                # Subplot 5: Amplitude Change (%)
                ax_amp = axes[4]
                
                amplitudes = [p.get(f'dark{dark_idx+1}_d', 0) for p in self.fitted_params]
                times_all = [p.get('time', np.nan) for p in self.fitted_params]
                
                valid_mask = (~np.isnan(times_all)) & (np.array(amplitudes) != 0)
                times_valid = [t for t, v in zip(times_all, valid_mask) if v]
                amplitudes_valid = [a for a, v in zip(amplitudes, valid_mask) if v]
                
                if len(amplitudes_valid) > 0 and amplitudes_valid[0] != 0:
                    baseline_amp = amplitudes_valid[0]
                    amplitude_changes = [(a - baseline_amp) / baseline_amp * 100 for a in amplitudes_valid]
                    ax_amp.plot(times_valid, amplitude_changes, 'm-', linewidth=1.5, marker='^', markersize=3)
                else:
                    print(f"    ⚠️ WARNING: No valid amplitude data for dark mode {dark_idx+1}")
                
                ax_amp.set_ylabel(f'Dark {dark_idx+1}\nAmplitude Δ (%)', fontsize=10, fontweight='bold')
                ax_amp.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
                ax_amp.tick_params(labelsize=9)
                ax_amp.grid(True, alpha=0.3)
                ax_amp.set_xlim(self.dataset.spec_times.min(), self.dataset.spec_times.max())
                
                plt.tight_layout()
                plt.savefig(output_dir / f"{self.dataset.sample_name}_spectral_heatmap_dark{dark_idx+1}_eV.png", dpi=150)
                plt.close()
                
                # 데이터 저장
                param_data = []
                for i, p in enumerate(self.fitted_params):
                    row = [
                        p.get('time', np.nan),
                        self.dataset.voltages[i] if i < len(self.dataset.voltages) else np.nan,
                        resonances[i] if i < len(resonances) else np.nan,
                        gammas[i] if i < len(gammas) else np.nan,
                        amplitudes[i] if i < len(amplitudes) else 0
                    ]
                    param_data.append(row)
                
                param_data = np.array(param_data)
                
                header = f"Time(s)\tVoltage(V)\tResonance(eV)\tGamma(eV)\tAmplitude(a.u.)\nDark_{dark_idx+1}"
                
                np.savetxt(data_dir / f"{self.dataset.sample_name}_dark{dark_idx+1}_parameters.txt",
                           param_data,
                           header=header,
                           delimiter='\t', fmt='%.6f')
                
                print(f"  Saved dark mode {dark_idx+1} heatmap with parameters")
        
        print(f"[info] Saved {num_peaks} peak-separated heatmaps with parameters")
        if fitting_model == 'fano':
            print(f"[info] Saved {num_dark} dark mode heatmaps with parameters")
        print(f"[info] Saved plot data to {data_dir}")

    def save_spectra_data(self):
        """Export spectral data and fitted parameters to text files and plots"""
        print("\n[Step] Saving spectral data and plots...")
        
        output_dir = Path(self.args['OUTPUT_DIR']) / "spectra"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots_dir = output_dir / "plots"
        plots_raw_dir = output_dir / "plots_raw"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plots_raw_dir.mkdir(parents=True, exist_ok=True)

        output_unit = self.args.get('OUTPUT_UNIT', 'eV')
        fitting_model = self.args.get('FITTING_MODEL', 'lorentzian')
        
        # ✅ 모델에 따라 피크 개수 다르게 설정
        if fitting_model == 'fano':
            num_peaks = self.args.get('NUM_BRIGHT_MODES', 1)
        else:
            num_peaks = self.args.get('NUM_PEAKS', 1)
        
        for i, param in enumerate(self.fitted_params):
            
            # ===== Save text data =====
            if output_unit == 'eV':
                # ✅ 수정: 제대로 정렬해서 변환
                energy = 1239.842 / self.dataset.wavelengths
                
                # energy 증가 순서로 정렬 (argsort 사용)
                sort_idx = np.argsort(energy)
                energy_sorted = energy[sort_idx]
                spectrum_sorted = param['spectrum'][sort_idx]
                fit_sorted = param['fit'][sort_idx]
                
                data = np.column_stack((energy_sorted, spectrum_sorted, fit_sorted))

                header = f"Energy(eV)\tIntensity\tFit\n"
                
                # ✅ 모델별로 다른 헤더 정보 추가
                if fitting_model == 'fano':
                    # Fano 모델: Bright modes 정보
                    for peak_idx in range(1, num_peaks + 1):
                        resonance_ev = param.get(f'resonanceeV{peak_idx}', 0)
                        gamma_ev = param.get(f'gammaeV{peak_idx}', 0)
                        coupling = param.get(f'c{peak_idx}', 0)
                        
                        if resonance_ev > 0:
                            resonance_nm = 1239.842 / resonance_ev
                            if gamma_ev > 0:
                                gamma_nm = abs(1239.842/(resonance_ev - gamma_ev/2) - 
                                              1239.842/(resonance_ev + gamma_ev/2))
                            else:
                                gamma_nm = 0
                            header += f"# Bright {peak_idx}: {resonance_ev:.3f} eV ({resonance_nm:.1f} nm), γ: {gamma_ev:.3f} eV ({gamma_nm:.1f} nm), c: {coupling:.2f}\n"
                else:
                    # Lorentzian 모델: Peak 정보
                    for peak_idx in range(1, num_peaks + 1):
                        peak_ev = param.get(f'peakeV{peak_idx}', 0)
                        fwhm_ev = param.get(f'FWHMeV{peak_idx}', 0)
                        
                        if peak_ev > 0:
                            peak_nm = 1239.842 / peak_ev
                            if fwhm_ev > 0:
                                fwhm_nm = abs(1239.842/(peak_ev - fwhm_ev/2) - 1239.842/(peak_ev + fwhm_ev/2))
                            else:
                                fwhm_nm = 0
                            header += f"# Peak {peak_idx}: {peak_ev:.3f} eV ({peak_nm:.1f} nm), FWHM: {fwhm_ev:.3f} eV ({fwhm_nm:.1f} nm)\n"
                
                header += f"# Spectrum {i+1}, Time={param['time']:.2f}s, Voltage={param['voltage']:.3f}V\n"
                header += f"# R²: {param['r2']:.4f}, SNR: {param['snr']:.1f}"

            else:
                # nm 단위일 때는 원본 wavelength 사용
                data = np.column_stack((
                    self.dataset.wavelengths, 
                    param['spectrum'], 
                    param['fit']
                ))

                header = f"Wavelength(nm)\tIntensity\tFit\n"
                
                # ✅ 모델별로 다른 헤더 정보 추가
                if fitting_model == 'fano':
                    # Fano 모델: Bright modes 정보
                    for peak_idx in range(1, num_peaks + 1):
                        resonance_nm = param.get(f'resonancenm{peak_idx}', 0)
                        gamma_nm = param.get(f'gammanm{peak_idx}', 0)
                        coupling = param.get(f'c{peak_idx}', 0)
                        if resonance_nm > 0:
                            header += f"# Bright {peak_idx}: {resonance_nm:.1f} nm, γ: {gamma_nm:.1f} nm, c: {coupling:.2f}\n"
                else:
                    # Lorentzian 모델: Peak 정보
                    for peak_idx in range(1, num_peaks + 1):
                        peak_nm = param.get(f'peaknm{peak_idx}', 0)
                        fwhm_nm = param.get(f'FWHMnm{peak_idx}', 0)
                        if peak_nm > 0:
                            header += f"# Peak {peak_idx}: {peak_nm:.1f} nm, FWHM: {fwhm_nm:.1f} nm\n"
                
                header += f"# Spectrum {i+1}, Time={param['time']:.2f}s, Voltage={param['voltage']:.3f}V\n"
                header += f"# R²: {param['r2']:.4f}, SNR: {param['snr']:.1f}"
            
            # Save text file
            output_file = output_dir / f"{self.dataset.sample_name}_spectrum_{i+1:04d}.txt"
            np.savetxt(output_file, data, delimiter='\t', header=header, 
                      comments='', fmt='%.6f')
            
            # ===== Generate plots =====
            plot_title = f"Spectrum {i+1} | t={param['time']:.1f}s, V={param['voltage']:.3f}V"
            plot_path = plots_dir / f"{self.dataset.sample_name}_spectrum_{i+1:04d}.png"
            
            # Plot with fit
            su.plot_spectrum(
                self.dataset.wavelengths,  # nm 단위로 플롯 (eV는 plot_spectrum 내부에서 처리)
                param['spectrum'],
                param['fit'],
                plot_title,
                plot_path,
                dpi=self.args.get("FIG_DPI", 300),
                params=param,
                snr=param['snr'],
                args=self.args,
                show_fit=True
            )
            
            # Plot raw only
            plot_path_raw = plots_raw_dir / f"{self.dataset.sample_name}_spectrum_{i+1:04d}.png"
            su.plot_spectrum(
                self.dataset.wavelengths,
                param['spectrum'],
                param['fit'],
                plot_title,
                plot_path_raw,
                dpi=self.args.get("FIG_DPI", 300),
                params=None,
                snr=None,
                args=self.args,
                show_fit=False
            )
            
            if (i + 1) % 10 == 0 or (i + 1) == len(self.fitted_params):
                print(f"  Saved {i+1}/{len(self.fitted_params)} spectra (text + 2 plots)")

            if (i + 1) % 10 == 0:
                gc.collect()  # 가비지 컬렉션 실행
                plt.close('all')  # 모든 matplotlib figure 닫기                
        
        print(f"[info] Saved {len(self.fitted_params)} spectral files to {output_dir}")
        print(f"[info] Saved {len(self.fitted_params)} plots with fit to {plots_dir}")
        print(f"[info] Saved {len(self.fitted_params)} raw-only plots to {plots_raw_dir}")

    def dump_results(self):
        """Save analysis results to pickle file"""
        print("\n[Step] Saving analysis results...")
        
        # 🔧 수정: 스펙트럼 데이터 제거 (메모리 절약)
        fitted_params_light = []
        for p in self.fitted_params:
            # spectrum과 fit 배열 제외
            p_light = {k: v for k, v in p.items() if k not in ['spectrum', 'fit']}
            fitted_params_light.append(p_light)
        
        results = {
            'args': self.args,
            'fitted_params': fitted_params_light,  # 🔧 가벼운 버전
            'rejected_fits': self.rejected_fits,
            'cycles': self.cycles,
            'cycle_boundaries': self.cycle_boundaries,
            'sample_name': self.dataset.sample_name,
            'technique': self.technique,
            'ocp': self.ocp
        }
        
        output_path = self.dataset.echem_output_dir / f"{self.dataset.sample_name}_echem_results.pkl"
        
        with open(output_path, 'wb') as f:
            pkl.dump(results, f)
        
        print(f"[info] Saved results to {output_path}")

    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Sample: {self.dataset.sample_name}")
        print(f"Technique: {self.technique}")

        fitting_model = self.args.get('FITTING_MODEL', 'lorentzian')
        print(f"Fitting model: {fitting_model}")

        if fitting_model == 'fano':
            num_bright = self.args.get('NUM_BRIGHT_MODES', 0)
            num_dark = self.args.get('NUM_DARK_MODES', 0)
            print(f"  Bright modes: {num_bright}")
            print(f"  Dark modes: {num_dark}")

        elif fitting_model == 'lorentzian':
            print(f"  Lorentzian peaks: {self.args.get('NUM_PEAKS', 1)}")

        else:
            raise ValueError("[error] Wrong fitting model")
        
        peak_guess = self.args.get('PEAK_INITIAL_GUESS', 'auto')
        if peak_guess != 'auto':
            print(f"Initial guess: {peak_guess} nm")
        else:
            print(f"Initial guess: auto-detect")
        
        # Print fitting constraint settings
        if fitting_model == 'fano':
            bright_tol = self.args.get('BRIGHT_POSITION_TOLERANCE', None)
            dark_tol = self.args.get('DARK_POSITION_TOLERANCE', None)
            
            if bright_tol is not None:
                if isinstance(bright_tol, (list, tuple)):
                    print(f"Bright position tolerance: {bright_tol} nm (per mode)")
                else:
                    print(f"Bright position tolerance: ±{bright_tol} nm")
            
            if dark_tol is not None:
                if isinstance(dark_tol, (list, tuple)):
                    print(f"Dark position tolerance: {dark_tol} nm (per mode)")
                else:
                    print(f"Dark position tolerance: ±{dark_tol} nm")
        else:
            peak_tol = self.args.get('PEAK_POSITION_TOLERANCE', None)
            if peak_tol is not None:
                if isinstance(peak_tol, (list, tuple)):
                    print(f"Peak position tolerance: {peak_tol} nm (per peak)")
                else:
                    print(f"Peak position tolerance: ±{peak_tol} nm")
            else:
                print(f"Peak position tolerance: None (unconstrained)")
        
        # Print retry fitting settings
        max_attempts = self.args.get('FIT_MAX_ATTEMPTS', 1)
        if max_attempts > 1:
            retry_strategy = self.args.get('FIT_RETRY_STRATEGY', 'broaden_bounds')
            retry_factor = self.args.get('FIT_RETRY_FACTOR', 1.5)
            print(f"Fitting attempts: {max_attempts} (strategy: {retry_strategy}, factor: {retry_factor}x)")
        else:
            print(f"Fitting attempts: 1 (no retry)")
        
        print(f"\nTotal spectra: {self.dataset.spectra.shape[0]}")
        print(f"Successfully fitted: {len(self.fitted_params)}")
        print(f"Rejected fits: {len(self.rejected_fits)}")
        
        # Rejection breakdown
        if len(self.rejected_fits) > 0:
            negative_r2 = len([r for r in self.rejected_fits if 'negative' in r['reason'].lower()])
            width_rejected = len([r for r in self.rejected_fits if 'FWHM' in r['reason'] or 'Width' in r['reason']])
            low_r2 = len([r for r in self.rejected_fits if 'too low' in r['reason']])
            
            print(f"\n[Rejection Breakdown]")
            print(f"  Negative R²: {negative_r2}")
            print(f"  FWHM too large: {width_rejected}")
            print(f"  R² too low: {low_r2}")
        
        if len(self.fitted_params) > 0:
            if fitting_model == 'fano':
                # Fano 모델 통계
                resonances = [p.get('resonanceeV1', 0) for p in self.fitted_params if p.get('resonanceeV1', 0) > 0]
                gammas = [p.get('gammaeV1', 0) for p in self.fitted_params if p.get('gammaeV1', 0) > 0]
                couplings = [p.get('c1', 0) for p in self.fitted_params if p.get('c1', 0) != 0]
                r2s = [p['r2'] for p in self.fitted_params]
                
                if len(resonances) > 0:
                    print(f"\nBright Mode 1 Resonance: {np.mean(resonances):.4f} ± {np.std(resonances):.4f} eV")
                    print(f"  Range: {min(resonances):.4f} - {max(resonances):.4f} eV")
                
                if len(gammas) > 0:
                    print(f"\nBright Mode 1 Gamma: {np.mean(gammas):.4f} ± {np.std(gammas):.4f} eV")
                    print(f"  Range: {min(gammas):.4f} - {max(gammas):.4f} eV")
                
                if len(couplings) > 0:
                    print(f"\nBright Mode 1 Coupling: {np.mean(couplings):.2f} ± {np.std(couplings):.2f}")
                    print(f"  Range: {min(couplings):.2f} - {max(couplings):.2f}")
                
            else:
                # Lorentzian 모델 통계
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