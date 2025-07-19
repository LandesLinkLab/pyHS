import os
import sys
import timeit
import pickle  as pkl
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

    def select_representatives(self):

        self.reps = su.pick_representatives(self.dataset.cube, self.dataset.labels, self.dataset.wvl, self.args)

    def fit_and_plot(self):
        out_dir = Path(self.args['OUTPUT_DIR'])
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, r in enumerate(self.reps):
            row, col = r["row"], r["col"]
            y_raw = self.dataset.cube[row, col]
            # 1) 로렌츠 피팅
            y_fit, params, r2 = su.fit_lorentz(y_raw, self.dataset.wvl, self.args)

            # 2) residual로부터 noise 계산 → S/N
            resid = y_raw - y_fit
            noise = np.std(resid)
            snr = params.get("a", 0) / noise if noise > 0 else 0

            # 3) 스펙트럼 플롯 (제목에는 R² 대신 index만, 나머지는 우측 상단 텍스트)
            su.plot_spectrum(self.dataset.wvl,
                        y_raw,
                        y_fit,
                        f"{self.dataset.sample_name} #{i}",
                        out_dir / f"{self.dataset.sample_name}_{i:03}.png",
                        dpi=self.args["FIG_DPI"],
                        params=params,
                        snr=snr)

            # 결과 저장
            self.results.append(dict(index=i, coord=(int(row), int(col)), wl_peak=r["wl_peak"], intensity=r["intensity"], params=params, rsq=r2, snr=snr))

        # 4) max-intensity projection 위에 마커 저장
        su.save_markers(self.dataset.cube,
                        self.reps,
                        out_dir / f"{self.dataset.sample_name}_markers.png",
                        dpi=self.args["FIG_DPI"])

        su.dump_mapping_diagnostic(self.dataset.cube,
                                self.reps,
                                out_dir / f"{self.dataset.sample_name}_mapping_dbg.png")

    def dump_pkl(self):

        out = Path(self.args['OUTPUT_DIR']) / f"{self.dataset.sample_name}.pkl"
        
        payload = dict(sample=self.dataset.sample_name,
                       wavelengths=self.dataset.wvl,
                       reps=self.results,
                       config=self.args)
        
        with open(out, "wb") as f:
        
            pkl.dump(payload, f, protocol=pkl.HIGHEST_PROTOCOL)
