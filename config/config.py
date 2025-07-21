import os
import sys
from pathlib import Path
home = str(Path.home())
args = dict()

# Basic settings
args['SAMPLE_NAME'] = 'AuNR_PMMA'
args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/raw')
args['WHITE_FILE'] = "wc.tdms"
args['DARK_FILE'] = "dc.tdms"
args['OUTPUT_DIR'] = os.path.join(home, "research", "pyHS")

# DFS-specific settings
# 실제 측정 범위가 388-897nm이지만, 관심 영역은 500-1000nm
args['DFS_WL_RANGE'] = (600, 850)  # DFS max intensity map을 위한 범위
args['DFS_INTENSITY_THRESHOLD'] = 0.05  # 낮춰서 더 많은 파티클 검출

# Preprocessing
# TDMS 파일의 전체 범위가 388-897nm이므로, 500-850nm로 crop
args['CROP_RANGE_NM'] = (500, 850)  
args['BACKGROUND_PERC'] = 0.01
args['SKIP_FLATFIELD'] = False  # Flatfield 사용

args['DC_MODE'] = "local" # local
args['DC_LOCAL_SEARCH_RADIUS'] = 20 # searching radius
args['DC_LOCAL_PERCENTILE'] = 1 

# Particle detection
args['MIN_PIXELS_CLUS'] = 3  # 최소 클러스터 크기 (더 작게)
args['PEAK_TOL_NM'] = 20.0  # FWHM tolerance

# Analysis settings
args['FIT_RANGE_NM'] = (500, 800)
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