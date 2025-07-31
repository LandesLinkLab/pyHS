import os
import sys
from pathlib import Path
home = str(Path.home())
args = dict()

# Basic settings
args['SAMPLE_NAME'] = 'AuNR_PMMA'
args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/raw_0714')
args['WHITE_FILE'] = "wc.tdms"
args['DARK_FILE'] = "dc.tdms"
args['OUTPUT_DIR'] = os.path.join(home, "research", "pyHS_jaekak_python_global_auto")

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
args['USE_MANUAL_COORDS'] = False
args['MANUAL_COORDS'] = [
(28, 9),    #1
(38, 8),    #2
(39, 19),    #3
(29, 35),    #4
(30, 53),    #5
(12, 73),    #6
(14, 73),    #7
(36, 92),    #8
(9, 110),    #9
(9, 114),    #10
(6, 128),    #11
(10, 132),    #12
(14, 142),    #13
(21, 125),    #14
(24, 122),    #15
(28, 121),    #16
(25, 149),    #17
(34, 116),    #18
(37, 114),    #19
(16, 168),    #20
(17, 156),    #21
(25, 176)    #22
]  # (Row, Col) format

# Debug mode
args['DEBUG'] = True