import os
import sys
import timeit
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any, Union

import spectrum_util as su

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
        self.dump_pickle()

    def select_representatives(self):

        self.reps = su.pick_representatives(self.dataset.cube, self.dataset.labels, self.dataset.wvl, self.args)

    def fit_and_plot(self):

        out_dir = Path(self.args['OUTPUT_DIR'])
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, r in enumerate(self.reps):

            r_, c_ = r["row"], r["col"]
            y_raw = self.dataset.cube[r_, c_]
            y_fit, params, r2 = su.fit_lorentz(y_raw, self.dataset.wvl, self.args)

            su.plot_spectrum(self.dataset.wvl, 
                            y_raw, 
                            y_fit,
                            f"{self.dataset.sample_name} #{i}  R²={r2:.3f}",
                            out_dir / f"{self.dataset.sample_name}_{i:03}.png",
                            dpi=self.args['FIG_DPI'])

            self.results.append(dict(index=i,
                                    coord=(int(r_), int(c_)),
                                    wl_peak=r["wl_peak"],
                                    intensity=r["intensity"],
                                    params=params,
                                    rsq=r2))

        su.save_markers(self.dataset.rgb, 
                        self.reps,
                        out_dir / f"{self.dataset.sample_name}_markers.png")

    def dump_pkl(self):

        out = Path(self.args['OUTPUT_DIR']) / f"{self.dataset.sample_name}.pkl"
        
        payload = dict(sample=self.dataset.sample_name,
                       wavelengths=self.dataset.wvl,
                       reps=self.results,
                       config=dict(vars(self.args)))
        
        with open(out, "wb") as f:
        
            pkl.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
