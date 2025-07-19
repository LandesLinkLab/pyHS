import os
import sys
from pathlib import Path

home = str(Path.home())
args = dict()

# File information
args['SAMPLE_NAME'] = 'AuNR_PMMA' # Scanned HS file name
args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/raw') # Raw data dir
args['WHITE_FILE'] = "wc.tdms" # White correction file name
args['DARK_FILE'] = "dc.tdms" # Dark correction file name
args['OUTPUT_DIR'] = os.path.join(home, "research", "pyHS") # Output dir

# Dataset information
args['CROP_RANGE_NM'] = (500, 800) # wavelength for spectrum analysis
args['BACKGROUND_PERC'] = 0.01 # ??

# Spectrum analysis information
args['THRESH_HIGH'] = 0.25 # Otsu threshold
args['MIN_PIXELS_CLUS'] = 4 # Threshold for minimun pixels per NP
args['PEAK_TOL_NM'] = 10.0 # Gamma difference threshold for SP/dimer detection
args['REP_CRITERION'] = "max_int" # "first"
args['FIG_DPI'] = 300 # figure dpi

# Not implemented yet
args['MAX_PEAKS'] = 1 # number of max peaks (for multiple peaks)
args['RSQ_MIN'] = 0.90 # minimum R^2 value for fitting

