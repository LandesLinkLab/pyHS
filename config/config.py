import os
import sys
from pathlib import Path

home = str(Path.home())
args = dict()

args['DATA_DIR'] = os.path.join(home, 'dataset/pyhs/raw')
args['WHITE_FILE'] = os.path.join(DATA_DIR, "wc1.tdms")
args['DARK_FILE'] = os.path.join(DATA_DIR, "dc1.tdms")
args['OUTPUT_DIR'] = os.path.join(DATA_DIR, "results")

args['CROP_RANGE_NM'] = (450, 800)
args['BACKGROUND_PERC'] = 0.01

args['THRESH_HIGH'] = 0.25
args['MIN_PIXELS_CLUS'] = 4
args['PEAK_TOL_NM'] = 10.0
args['REP_CRITERION'] = "max_int"

args['MAX_PEAKS'] = 1
args['RSQ_MIN'] = 0.90
args['FIG_DPI'] = 300
