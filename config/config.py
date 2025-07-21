import os
import sys
from pathlib import Path

home = str(Path.home())
args = dict()

args['SAMPLE_NAME'] = 'AuNR_PMMA'
args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/raw')
args['WHITE_FILE'] = "wc.tdms"
args['DARK_FILE'] = "dc.tdms"
args['OUTPUT_DIR'] = os.path.join(home, "research", "pyHS")
args['CROP_RANGE_NM'] = (450, 800)
args['BACKGROUND_PERC'] = 0.01
args['THRESH_HIGH'] = 0.1
args['MIN_PIXELS_CLUS'] = 4
args['PEAK_TOL_NM'] = 10.0
args['REP_CRITERION'] = "max_int"
args['MAX_PEAKS'] = 1
args['RSQ_MIN'] = 0.90
args['FIG_DPI'] = 300

# Manual particle selection mode
# Set to True to use manual coordinates instead of automatic detection
args['USE_MANUAL_COORDS'] = False

# Manual coordinates (row, col) - 이미지 크기에 맞게 조정하세요
# 이미지가 (49, 189) 크기라면, 중앙 부근의 좌표들을 시도
args['MANUAL_COORDS'] = [
    (24, 94),   # 중앙
    (10, 50),   # 왼쪽 위
    (35, 150),  # 오른쪽 아래
    (20, 100),  # 중앙 근처
    (30, 80),   # 또 다른 위치
]

# Integration size for spectrum extraction (3x3 or 5x5 pixels)
args['INTEGRATION_SIZE'] = 3

# Background offset (pixels away from particle for background estimation)
args['BACKGROUND_OFFSET'] = 7

# Visualization settings (MATLAB-compatible)
args['VIZ_MATLAB_STYLE'] = True  # Use MATLAB-style visualization
args['VIZ_SAVE_EXTRA'] = False   # Save additional visualizations

# Debug mode
args['DEBUG'] = True  # Enable debug output