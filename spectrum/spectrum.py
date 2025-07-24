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
    def __init__(self, args: Dict[str, Any], dataset):
        self.args = args
        self.dataset = dataset
        self.cluster_fits = []  # 모든 픽셀의 fitting 결과
        self.representatives = []  # 클러스터별 대표 스펙트럼
        
    def run_spectrum(self):
        """
        Spectrum analysis pipeline:
        1. Fit all particles in clusters
        2. Select representative spectra per cluster
        3. Plot results
        4. Save marking image and spectra
        """
        self.fit_all_particles()
        self.select_representatives()
        self.plot_representatives()
        self.save_dfs_particle_map()
        self.save_spectra()
        self.dump_pkl()
        self.print_summary()
    
    def fit_all_particles(self):
        """Fit Lorentzian to all particles in all clusters"""
        print("\n[Step] Fitting all particles in clusters...")
        
        if not self.dataset.clusters:
            print("[warning] No clusters found for analysis")
            return
        
        # 각 클러스터의 모든 픽셀에 대해 fitting
        for cluster in self.dataset.clusters:
            cluster_results = []
            
            print(f"\n[Fitting Cluster {cluster['label']}] - {cluster['size']} pixels")
            
            for coord in cluster['coords']:
                row, col = coord
                
                # Extract spectrum
                spectrum = self.dataset.cube[row, col, :]
                
                # Skip if spectrum is too weak
                if spectrum.max() < 0.01:
                    continue
                
                # Lorentzian fitting
                y_fit, params, r2 = su.fit_lorentz(self.args, spectrum, self.dataset.wvl)
                
                # Calculate S/N
                resid = spectrum - y_fit
                noise = np.std(resid) if np.std(resid) > 0 else 1.0
                peak_idx = np.argmax(spectrum)
                snr = spectrum[peak_idx] / noise
                
                # Store results
                result = {
                    'row': row,
                    'col': col,
                    'spectrum': spectrum,
                    'fit': y_fit,
                    'params': params,
                    'r2': r2,
                    'snr': snr,
                    'peak_wl': params.get('b1', self.dataset.wvl[peak_idx]),
                    'fwhm': params.get('c1', 0),
                    'peak_intensity': spectrum[peak_idx]
                }
                
                cluster_results.append(result)
            
            self.cluster_fits.append({
                'cluster_label': cluster['label'],
                'cluster_size': cluster['size'],
                'fits': cluster_results
            })
            
            print(f"  Successfully fitted {len(cluster_results)} pixels")
    
    def select_representatives(self):
        """Select representative spectrum for each cluster based on fitting results"""
        print("\n[Step] Selecting representative spectra for each cluster...")
        
        for cluster_fit in self.cluster_fits:
            fits = cluster_fit['fits']
            
            if not fits:
                print(f"  Cluster {cluster_fit['cluster_label']}: No valid fits")
                continue
            
            # Check FWHM tolerance
            fwhms = [f['fwhm'] for f in fits if f['fwhm'] > 0]
            if len(fwhms) > 1:
                fwhm_std = np.std(fwhms)
                fwhm_mean = np.mean(fwhms)
                print(f"\n  Cluster {cluster_fit['cluster_label']}:")
                print(f"    FWHM: {fwhm_mean:.1f} ± {fwhm_std:.1f} nm")
                
                if fwhm_std > self.args['PEAK_TOL_NM']:
                    print(f"    [Skipped] FWHM variation ({fwhm_std:.1f}) > tolerance ({self.args['PEAK_TOL_NM']})")
                    continue
            
            # Select best pixel - always use max intensity
            # (FWHM tolerance already checked above)
            best_idx = max(range(len(fits)), key=lambda i: fits[i]['peak_intensity'])
            
            best = fits[best_idx]
            
            print(f"    Selected pixel ({best['row']}, {best['col']})")
            print(f"    - Peak: {best['peak_wl']:.1f} nm @ {best['peak_intensity']:.1f}")
            print(f"    - FWHM: {best['fwhm']:.1f} nm")
            print(f"    - S/N: {best['snr']:.1f}, R²: {best['r2']:.3f}")
            
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
                'peak_intensity': best['peak_intensity']
            })
        
        print(f"\n[Summary] {len(self.representatives)} valid representatives from {len(self.cluster_fits)} clusters")
    
    def plot_representatives(self):
        """Plot representative spectra"""
        print("\n[Step] Plotting representative spectra...")
        
        out_dir = Path(self.args['OUTPUT_DIR'])
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for i, rep in enumerate(self.representatives):
            # Plot spectrum
            su.plot_spectrum(
                self.dataset.wvl,
                rep['spectrum'],
                rep['fit'],
                f"Particle {i} (Cluster {rep['cluster_label']})",
                out_dir / f"{self.dataset.sample_name}_particle_{i:03d}.png",
                dpi=self.args["FIG_DPI"],
                params=rep['params'],
                snr=rep['snr'],
                args=self.args
            )
    
    def save_dfs_particle_map(self):
        """Save particle map with markers"""
        if not self.representatives:
            return
        
        out_dir = Path(self.args['OUTPUT_DIR'])
        output_path = out_dir / f"{self.dataset.sample_name}_dfs_markers.png"
        
        su.save_dfs_particle_map(
            self.dataset.max_map,
            self.representatives,
            output_path,
            self.dataset.sample_name
        )
    
    def save_spectra(self):
        """Save spectra data to text files"""
        print("\n[Step] Saving spectra data...")
        
        out_dir = Path(self.args['OUTPUT_DIR']) / "spectra"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for i, rep in enumerate(self.representatives):
            # Save spectrum data
            data = np.column_stack((self.dataset.wvl, rep['spectrum'], rep['fit']))
            header = f"Wavelength(nm)\tIntensity\tFit\n"
            header += f"# Particle {i}, Cluster {rep['cluster_label']}, Position ({rep['row']},{rep['col']})\n"
            header += f"# Peak: {rep['peak_wl']:.1f} nm, FWHM: {rep['fwhm']:.1f} nm, S/N: {rep['snr']:.1f}"
            
            np.savetxt(
                out_dir / f"{self.dataset.sample_name}_particle_{i:03d}.txt",
                data,
                delimiter='\t',
                header=header,
                fmt='%.3f'
            )
    
    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("DFS ANALYSIS SUMMARY")
        print("="*60)
        print(f"Sample: {self.dataset.sample_name}")
        print(f"Total clusters detected: {len(self.dataset.clusters) if self.dataset.clusters else 0}")
        print(f"Clusters analyzed: {len(self.cluster_fits)}")
        print(f"Valid representatives: {len(self.representatives)}")
        
        if self.representatives:
            # Statistics
            wavelengths = [r['peak_wl'] for r in self.representatives]
            fwhms = [r['fwhm'] for r in self.representatives if r['fwhm'] > 0]
            snrs = [r['snr'] for r in self.representatives]
            cluster_sizes = [r['cluster_size'] for r in self.representatives]
            r2s = [r['r2'] for r in self.representatives]
            
            print(f"\nResonance wavelength: {np.mean(wavelengths):.1f} ± {np.std(wavelengths):.1f} nm")
            print(f"  Range: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
            
            if fwhms:
                print(f"\nFWHM: {np.mean(fwhms):.1f} ± {np.std(fwhms):.1f} nm")
                print(f"  Range: {min(fwhms):.1f} - {max(fwhms):.1f} nm")
            
            print(f"\nS/N ratio: {np.mean(snrs):.1f} ± {np.std(snrs):.1f}")
            print(f"  Range: {min(snrs):.1f} - {max(snrs):.1f}")
            
            print(f"\nR² values: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
            print(f"  Range: {min(r2s):.3f} - {max(r2s):.3f}")
            
            print(f"\nCluster sizes: {np.mean(cluster_sizes):.1f} ± {np.std(cluster_sizes):.1f} pixels")
            print(f"  Range: {min(cluster_sizes)} - {max(cluster_sizes)} pixels")
        
        print("="*60)
    
    def dump_pkl(self):
        """Save all results to pickle file"""
        out = Path(self.args['OUTPUT_DIR']) / f"{self.dataset.sample_name}_results.pkl"
        
        payload = {
            'sample': self.dataset.sample_name,
            'wavelengths': self.dataset.wvl,
            'cube_shape': self.dataset.cube.shape,
            'max_map': self.dataset.max_map,
            'clusters': self.dataset.clusters,
            'cluster_fits': self.cluster_fits,
            'representatives': self.representatives,
            'config': self.args,
            'analysis_date': str(datetime.now().strftime("%m-%d-%Y %H:%M:%S"))
        }
        
        with open(out, "wb") as f:
            pkl.dump(payload, f, protocol=pkl.HIGHEST_PROTOCOL)
        
        print(f"\n[info] Results saved to: {out}")