import os
import sys
from pathlib import Path
home = str(Path.home())
args = dict()

# Basic settings
args['SAMPLE_NAME'] = 'AuNR_PMMA'
args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/raw_0716')
args['WHITE_FILE'] = "wc.tdms"
args['DARK_FILE'] = "dc.tdms"
args['OUTPUT_DIR'] = os.path.join(home, "research", "pyHS_jaekak_python_global_manual_0716")

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
args['MIN_PIXELS_CLUS'] = 3  # 최소 클러스터 크기 (Python & MATLAB 공통)

# MATLAB style detection parameters  
args['PARTICLE_LOWER_BOUND'] = 0      # MATLAB의 lower bound
args['PARTICLE_UPPER_BOUND'] = 0.5    # MATLAB의 upper bound
args['NHOOD_SIZE'] = 1                # MATLAB의 nhood (odd number)

# Representative selection parameters
args['PEAK_TOL_NM'] = 3.0  # FWHM tolerance for representative selection
args['MAX_WIDTH_NM'] = 59
args['RSQ_MIN'] = 0.90
args['FIT_RANGE_NM'] = (500, 850)  # Lorentzian fitting range
args['FIG_DPI'] = 300


# Manual mode (currently not used in new workflow)
args['USE_MANUAL_COORDS'] = True
args['MANUAL_COORDS'] = [
(23, 60),    #1
(35, 57),    #2
(36, 68),    #3
(27, 86),    #4
(29, 114),    #5
(13, 124),    #6
(15, 124),    #7
(39, 142),    #8
(13, 162),    #9
(13, 166),    #10
(12, 180),    #11
(16, 184),    #12
(20, 193),    #13
(27, 176),    #14
(29, 172),    #15
(33, 171),    #16
(32, 199),    #17
(38, 166),    #18
(42, 164),    #19
(24, 219),    #20
(25, 309),    #21
(34, 227)    #22
]  # (Row, Col) format

# Debug mode
args['DEBUG'] = True