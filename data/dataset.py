
from pathlib import Path
import numpy as np
from . import dataset_util as du

class Dataset:
    """Load hyperspectral TDMS data and preprocess."""

    def __init__(self, sample_name: str, cfg, image_shape=None):
        self.sample_name = sample_name
        self.cfg = cfg
        self.image_shape = image_shape

        self.cube = None   # ndarray (H,W,λ)
        self.wvl = None    # ndarray (λ,)
        self.rgb = None
        self.labels = None
        self.centroids = None

    # ---------------- I/O & preprocessing ----------------
    def load_cube(self):
        path = Path(self.cfg.DATA_DIR) / f"{self.sample_name}.tdms"
        self.cube, self.wvl = du.tdms_to_cube(path, self.image_shape)

    def flatfield(self):
        w = Path(self.cfg.DATA_DIR) / self.cfg.WHITE_FILE
        d = Path(self.cfg.DATA_DIR) / self.cfg.DARK_FILE
        self.cube = du.flatfield_correct(self.cube, w, d)

    def preprocess(self):
        self.cube, self.wvl = du.crop_and_bg(self.cube, self.wvl, self.cfg)
        self.rgb = du.cube_to_rgb(self.cube, self.wvl)

    def detect_particles(self):
        self.labels, self.centroids = du.label_particles(self.rgb, self.cfg)
