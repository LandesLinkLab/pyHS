
from pathlib import Path
import gzip, pickle
from . import spectrum_util as su

class SpectrumAnalyzer:
    def __init__(self, dataset, cfg):
        self.ds = dataset
        self.cfg = cfg
        self.reps = []
        self.results = []

    def select_representatives(self):
        self.reps = su.pick_representatives(
            self.ds.cube, self.ds.labels, self.ds.wvl, self.cfg)

    def fit_and_plot(self):
        out_dir = Path(self.cfg.OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(self.reps):
            r_, c_ = r["row"], r["col"]
            y_raw = self.ds.cube[r_, c_]
            y_fit, params, r2 = su.fit_lorentz(y_raw, self.ds.wvl, self.cfg)
            su.plot_spectrum(self.ds.wvl, y_raw, y_fit,
                             f"{self.ds.sample_name} #{i}  R²={r2:.3f}",
                             out_dir / f"{self.ds.sample_name}_{i:03}.png",
                             dpi=self.cfg.FIG_DPI)
            self.results.append(dict(
                index=i,
                coord=(int(r_), int(c_)),
                wl_peak=r["wl_peak"],
                intensity=r["intensity"],
                params=params,
                rsq=r2))
        su.save_markers(self.ds.rgb, self.reps,
                        out_dir / f"{self.ds.sample_name}_markers.png")

    def dump_pickle(self):
        out = Path(self.cfg.OUTPUT_DIR) / f"{self.ds.sample_name}.pkl.gz"
        payload = dict(sample=self.ds.sample_name,
                       wavelengths=self.ds.wvl,
                       reps=self.results,
                       cfg=dict(vars(self.cfg)))
        with gzip.open(out, "wb", compresslevel=5) as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
