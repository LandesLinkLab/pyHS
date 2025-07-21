import os
import sys
import timeit
import pickle as pkl
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any, Union

from . import spectrum_util as su

# spectrum.py의 수정된 부분

class SpectrumAnalyzer:
    def __init__(self, args: Dict[str, Any], dataset):
        self.args = args
        self.dataset = dataset
        self.results = []
        
    def run_spectrum(self):
        self.fit_and_plot()
        self.save_dfs_particle_map()
        self.dump_pkl()
        self.print_summary()
        
    def fit_and_plot(self):
        """Fit and plot spectra for DFS particles"""
        out_dir = Path(self.args['OUTPUT_DIR'])
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.dataset.representatives:
            print("[warning] No particles found for analysis")
            return
        
        print(f"\n[info] Analyzing {len(self.dataset.representatives)} DFS particles...")
        
        for i, rep in enumerate(self.dataset.representatives):
            # 대표 스펙트럼 사용
            y_raw = rep['spectrum']
            
            # Lorentzian fitting
            y_fit, params, r2 = su.fit_lorentz(y_raw, self.dataset.wvl, self.args)
            
            # S/N 계산
            resid = y_raw - y_fit
            noise = np.std(resid) if np.std(resid) > 0 else 1.0
            snr = rep['peak_intensity'] / noise
            
            # 스펙트럼 플롯
            su.plot_spectrum(
                self.dataset.wvl,
                y_raw,
                y_fit,
                f"Particle {i}",
                out_dir / f"{self.dataset.sample_name}_particle_{i:03d}.png",
                dpi=self.args["FIG_DPI"],
                params=params,
                snr=snr
            )
            
            # 결과 저장
            self.results.append({
                'index': i,
                'coord': (rep['row'], rep['col']),
                'wl_peak': params.get('b1', rep['peak_wl']),
                'fwhm': params.get('c1', rep['fwhm']),
                'intensity': rep['peak_intensity'],
                'integrated_intensity': rep['spectrum'].sum(),
                'cluster_size': rep['cluster_size'],
                'params': params,
                'rsq': r2,
                'snr': snr,
                'spectrum': y_raw,  # 원본 스펙트럼도 저장
                'fit': y_fit
            })
            
            print(f"  Particle {i}: λ_max={params.get('b1', 0):.1f}nm, "
                  f"FWHM={params.get('c1', 0):.1f}nm, S/N={snr:.1f}, "
                  f"R²={r2:.3f}, cluster_size={rep['cluster_size']}")
    
    def save_dfs_particle_map(self):
        """Save DFS-specific particle map"""
        if not self.dataset.representatives:
            return
            
        out_dir = Path(self.args['OUTPUT_DIR'])
        output_path = out_dir / f"{self.dataset.sample_name}_dfs_markers.png"
        
        du.save_dfs_particle_map(
            self.dataset.max_map,
            self.dataset.representatives,
            output_path,
            self.dataset.sample_name
        )
    
    def print_summary(self):
        """Print DFS analysis summary"""
        print("\n" + "="*60)
        print("DFS ANALYSIS SUMMARY")
        print("="*60)
        print(f"Sample: {self.dataset.sample_name}")
        print(f"Total clusters detected: {len(self.dataset.clusters) if self.dataset.clusters else 0}")
        print(f"Valid particles analyzed: {len(self.results)}")
        
        if self.results:
            # 통계
            wavelengths = [r['wl_peak'] for r in self.results]
            fwhms = [r['fwhm'] for r in self.results if r['fwhm'] > 0]
            snrs = [r['snr'] for r in self.results]
            cluster_sizes = [r['cluster_size'] for r in self.results]
            
            print(f"\nResonance wavelength: {np.mean(wavelengths):.1f} ± {np.std(wavelengths):.1f} nm")
            print(f"  Range: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
            
            if fwhms:
                print(f"\nFWHM: {np.mean(fwhms):.1f} ± {np.std(fwhms):.1f} nm")
                print(f"  Range: {min(fwhms):.1f} - {max(fwhms):.1f} nm")
            
            print(f"\nS/N ratio: {np.mean(snrs):.1f} ± {np.std(snrs):.1f}")
            print(f"  Range: {min(snrs):.1f} - {max(snrs):.1f}")
            
            print(f"\nCluster sizes: {np.mean(cluster_sizes):.1f} ± {np.std(cluster_sizes):.1f} pixels")
            print(f"  Range: {min(cluster_sizes)} - {max(cluster_sizes)} pixels")
        
        print("="*60)

    def dump_pkl(self):
    """Save analysis results to pickle file"""
    out = Path(self.args['OUTPUT_DIR']) / f"{self.dataset.sample_name}_results.pkl"
    
    payload = {
        'sample': self.dataset.sample_name,
        'wavelengths': self.dataset.wvl,
        'particles': self.results,
        'config': self.args,
        'cube_shape': self.dataset.cube.shape,
        'max_map': self.dataset.max_map,
        'clusters': self.dataset.clusters,
        'representatives': self.dataset.representatives,
        'analysis_date': str(pd.Timestamp.now()) if 'pd' in globals() else None
    }
    
    with open(out, "wb") as f:
        pkl.dump(payload, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    print(f"\n[info] Results saved to: {out}")
