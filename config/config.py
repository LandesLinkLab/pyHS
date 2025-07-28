import os
import sys
from pathlib import Path
home = str(Path.home())
args = dict()

# Basic settings
args['SAMPLE_NAME'] = 'AuNR_PMMA'
args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/raw_jmkim')
args['WHITE_FILE'] = "wc.tdms"
args['DARK_FILE'] = "dc.tdms"
args['OUTPUT_DIR'] = os.path.join(home, "research", "pyHS_python_global")

# DFS-specific settings
args['DFS_WL_RANGE'] = (500, 850)  # DFS max intensity map을 위한 범위

# Preprocessing
args['CROP_RANGE_NM'] = (500, 850)  

# Background mode - MATLAB 방식의 global background
args['BACKGROUND_MODE'] = 'global'  # 'global' (MATLAB style), 'local'
args['BACKGROUND_PERCENTILE'] = 0.1  # MATLAB의 10%
args['BACKGROUND_LOCAL_SEARCH_RADIUS'] = 20 
args['BACKGROUND_LOCAL_PERCENTILE'] = 1

# Particle detection style
args['PARTICLE_DETECTION_STYLE'] = 'python'  # 'python' or 'matlab'

# Python style detection parameters
args['DFS_INTENSITY_THRESHOLD'] = 0.01  # Python 방식 particle detection threshold
args['MIN_PIXELS_CLUS'] = 1  # 최소 클러스터 크기 (Python & MATLAB 공통)

# MATLAB style detection parameters  
args['PARTICLE_LOWER_BOUND'] = 0      # MATLAB의 lower bound
args['PARTICLE_UPPER_BOUND'] = 0.5    # MATLAB의 upper bound
args['NHOOD_SIZE'] = 1                # MATLAB의 nhood (odd number)

# Representative selection parameters (공통)
args['PEAK_TOL_NM'] = 10.0  # FWHM tolerance for representative selection
args['FIT_RANGE_NM'] = (500, 850)  # Lorentzian fitting range
args['FIG_DPI'] = 300
args['RSQ_MIN'] = 0.90

# Manual mode (currently not used in new workflow)
args['USE_MANUAL_COORDS'] = True
args['MANUAL_COORDS'] = [
(29, 30),    #1
(27, 42),    #2
(38, 42),    #3
(55, 33),    #4
(73, 33),    #5
(92, 15),    #6
(92, 18),    #7
(112, 40),    #8
(130, 13),    #9
(134, 14),    #10
(148, 11),    #11
(152, 15),    #12
(162, 19),    #13
(145, 26),    #14
(142, 29),    #15
(140, 33),    #16
(168, 30),    #17
(136, 39),    #18
(134, 42),    #19
(188, 21),    #20
(176, 23),    #21
(196, 31),    #22
]  # example

# Debug mode
args['DEBUG'] = True