import os
import sys
from pathlib import Path
home = str(Path.home())
args = dict()

#================== dataset.py settings ==================
# Basic settings
args['SAMPLE_NAME'] = 'AuNR_PMMA'
args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/raw')
args['WHITE_FILE'] = "wc.tdms"
args['DARK_FILE'] = "dc.tdms"
args['OUTPUT_DIR'] = os.path.join(home, "research", "pyHS")

# DFS-specific settings
args['DFS_WL_RANGE'] = (600, 850)  # DFS max intensity map을 위한 범위
args['DFS_INTENSITY_THRESHOLD'] = 0.05  # Python 방식 particle detection threshold

# Preprocessing
args['CROP_RANGE_NM'] = (500, 850)  

# Background mode - MATLAB 방식의 global background
args['BACKGROUND_MODE'] = 'global'  # 'global' (MATLAB style), 'local'
args['BACKGROUND_PERCENTILE'] = 0.1  # MATLAB의 10%
args['BACKGROUND_LOCAL_SEARCH_RADIUS'] = 20 
args['BACKGROUND_LOCAL_PERCENTILE'] = 1


#================== dataset.py settings ==================
# Particle detection style
args['PARTICLE_DETECTION_STYLE'] = 'python'  # 'python' or 'matlab'

# Python style detection parameters
args['DFS_INTENSITY_THRESHOLD'] = 0.05  # Threshold for Python style
args['MIN_PIXELS_CLUS'] = 3  # 최소 클러스터 크기 (Python & MATLAB 공통)

# MATLAB style detection parameters  
args['PARTICLE_LOWER_BOUND'] = 0      # MATLAB의 lower bound
args['PARTICLE_UPPER_BOUND'] = 0.5    # MATLAB의 upper bound
args['NHOOD_SIZE'] = 1                # MATLAB의 nhood (odd number)

# Representative selection parameters (공통)
args['PEAK_TOL_NM'] = 20.0  # FWHM tolerance for representative selection


#================== spectrum.py settings ==================
# Analysis settings
args['FIT_RANGE_NM'] = (500, 850)
args['REP_CRITERION'] = "max_int"  # 대표 픽셀 선택 기준
args['INTEGRATION_SIZE'] = 3  # 스펙트럼 추출 시 통합 크기
args['BACKGROUND_OFFSET'] = 7  # 배경 추출 오프셋

# Output settings
args['FIG_DPI'] = 300
args['RSQ_MIN'] = 0.90

# Manual mode
args['USE_MANUAL_COORDS'] = False
args['MANUAL_COORDS'] = [
(10, 50), 
(20, 100),
(35, 150)
]  # example

# Debug mode
args['DEBUG'] = True