import os
import sys
import timeit
import pickle as pkl
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any, Union
from . import spectrum_util as su

class SpectrumAnalyzer:
    def __init__(self, 
                args: Dict[str, Any],
                dataset):
        self.args = args
        self.dataset = dataset
        self.reps = []
        self.results = []
        
    def run_spectrum(self):
        self.select_representatives()
        self.fit_and_plot()
        self.dump_pkl()
        self.print_summary()
        
    def select_representatives(self):
        self.reps = su.pick_representatives(self.dataset.cube, 
                                           self.dataset.labels, 
                                           self.dataset.wvl, 
                                           self.args)
        
    def fit_and_plot(self):
        out_dir = Path(self.args['OUTPUT_DIR'])
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[info] Analyzing {len(self.reps)} particles...")
        
        for i, r in enumerate(self.reps):
            row, col = r["row"], r["col"]
            
            # Extract spectrum with background subtraction if manual mode
            if self.args.get('USE_MANUAL_COORDS', False):
                y_raw = su.extract_spectrum_with_background(
                    self.dataset.cube, row, col, self.args)
            else:
                y_raw = self.dataset.cube[row, col]
            
            # 1) Lorentzian fitting
            y_fit, params, r2 = su.fit_lorentz(y_raw, self.dataset.wvl, self.args)
            
            # 2) Calculate S/N from residuals
            resid = y_raw - y_fit
            noise = np.std(resid)
            
            # Peak intensity at resonance wavelength
            lambda_max = params.get("b1", params.get("x0", 0))
            if lambda_max > 0:
                idx_max = np.argmin(np.abs(self.dataset.wvl - lambda_max))
                signal = y_raw[idx_max]
            else:
                signal = params.get("a", 0)
                
            snr = signal / noise if noise > 0 else 0
            
            # 3) Plot spectrum
            su.plot_spectrum(self.dataset.wvl,
                           y_raw,
                           y_fit,
                           f"Particle {i}",
                           out_dir / f"{self.dataset.sample_name}_particle_{i:03d}.png",
                           dpi=self.args["FIG_DPI"],
                           params=params,
                           snr=snr)
            
            # Save results
            self.results.append(dict(
                index=i, 
                coord=(int(row), int(col)), 
                wl_peak=params.get("b1", r["wl_peak"]),
                fwhm=params.get("c1", 0),
                intensity=r["intensity"], 
                params=params, 
                rsq=r2, 
                snr=snr
            ))
            
            print(f"  Particle {i}: λ_max={params.get('b1', 0):.1f}nm, "
                  f"FWHM={params.get('c1', 0):.1f}nm, S/N={snr:.1f}, R²={r2:.3f}")
        
        # 4) Save image with markers
        su.save_markers(self.dataset.cube,
                       self.reps,
                       out_dir / f"{self.dataset.sample_name}_markers.png",
                       dpi=self.args["FIG_DPI"])
        
    def dump_pkl(self):
        out = Path(self.args['OUTPUT_DIR']) / f"{self.dataset.sample_name}_results.pkl"
        
        payload = dict(
            sample=self.dataset.sample_name,
            wavelengths=self.dataset.wvl,
            particles=self.results,
            config=self.args,
            cube_shape=self.dataset.cube.shape,
            manual_mode=self.args.get('USE_MANUAL_COORDS', False)
        )
        
        with open(out, "wb") as f:
            pkl.dump(payload, f, protocol=pkl.HIGHEST_PROTOCOL)
            
        print(f"\n[info] Results saved to: {out}")
        
    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Sample: {self.dataset.sample_name}")
        print(f"Mode: {'Manual coordinates' if self.args.get('USE_MANUAL_COORDS', False) else 'Automatic detection'}")
        print(f"Number of particles: {len(self.results)}")
        
        if self.results:
            # Statistics
            wavelengths = [r['wl_peak'] for r in self.results]
            fwhms = [r['fwhm'] for r in self.results if r['fwhm'] > 0]
            snrs = [r['snr'] for r in self.results]
            
            print(f"\nResonance wavelength: {np.mean(wavelengths):.1f} ± {np.std(wavelengths):.1f} nm")
            if fwhms:
                print(f"FWHM: {np.mean(fwhms):.1f} ± {np.std(fwhms):.1f} nm")
            print(f"S/N ratio: {np.mean(snrs):.1f} ± {np.std(snrs):.1f}")
        
        print("="*60)