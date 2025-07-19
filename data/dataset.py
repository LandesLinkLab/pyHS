import os
import timeit
import pickle as pkl
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any, Union

from . import dataset_util as du

class Dataset(object):
    """Load hyperspectral TDMS data and preprocess."""

    def __init__(self, 
                args: Dict[str, Any]):
        
        self.args = args
        self.sample_name = args['SAMPLE_NAME']

        self.cube = None   # ndarray (H,W,λ)
        self.wvl = None    # ndarray (λ,)
        self.rgb = None
        self.labels = None
        self.centroids = None

    def run_dataset(self):

        self.load_cube()
        self.preprocess()
        self.flatfield()
        self.detect_particles()

    # ---------------- I/O & preprocessing ----------------
    def load_cube(self):

        sample_name = self.args['SAMPLE_NAME'] + ".tdms"
        
        path = os.path.join(self.args['DATA_DIR'], sample_name)

        self.cube, self.wvl = du.tdms_to_cube(path)

    def flatfield(self):

        w = os.path.join(self.args['DATA_DIR'], self.args['WHITE_FILE'])
        d = os.path.join(self.args['DATA_DIR'], self.args['DARK_FILE'])

        self.cube = du.flatfield_correct(self.cube, self.wvl, w, d)

    def preprocess(self):

        self.cube, self.wvl = du.crop_and_bg(self.cube, self.wvl, self.args)
        self.rgb = du.cube_to_rgb(self.cube, self.wvl)

    def detect_particles(self):

        self.labels, self.centroids = du.label_particles(self.rgb, self.args)



