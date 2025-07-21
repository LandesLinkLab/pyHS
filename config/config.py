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
args['DFS_WL_RANGE'] = (500, 800)  # 파장 범위 for max intensity map
args['DFS_INTENSITY_THRESHOLD'] = 0.1  # 상대적 intensity threshold (0-1)

# Preprocessing
args['CROP_RANGE_NM'] = (450, 850)  # 전체 분석 범위
args['BACKGROUND_PERC'] = 0.01
args['SKIP_FLATFIELD'] = False  # Flatfield 사용 여부

# Particle detection
args['MIN_PIXELS_CLUS'] = 4  # 최소 클러스터 크기
args['PEAK_TOL_NM'] = 10.0  # FWHM tolerance

# Analysis settings
args['REP_CRITERION'] = "max_int"  # 대표 픽셀 선택 기준
args['INTEGRATION_SIZE'] = 3  # 스펙트럼 추출 시 통합 크기
args['BACKGROUND_OFFSET'] = 7  # 배경 추출 오프셋

# Output settings
args['FIG_DPI'] = 300
args['RSQ_MIN'] = 0.90

# Manual mode (비활성화)
args['USE_MANUAL_COORDS'] = False
args['MANUAL_COORDS'] = []

# Debug mode
args['DEBUG'] = True