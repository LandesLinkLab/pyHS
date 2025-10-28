import os
import sys
import numpy as np
from pathlib import Path

# Get user home directory for cross-platform compatibility
home = str(Path.home())

# Initialize configuration dictionary
args = dict()

# ============================================================================
# ANALYSIS MODE
# ============================================================================
args['ANALYSIS_MODE'] = 'dfs'

# ============================================================================
# BASIC FILE AND DIRECTORY SETTINGS
# ============================================================================
args['SAMPLE_NAME'] = 'AuNR_PMMA'  # Name of the main TDMS file (without .tdms extension)
args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/raw_jmkim')  # Directory containing TDMS files
args['WHITE_FILE'] = "wc.tdms"     # White reference file name for flatfield correction
args['DARK_FILE'] = "dc.tdms"      # Dark reference file name for flatfield correction
args['OUTPUT_DIR'] = os.path.join(home, "research", "pyHS_dfs_output")  # Output directory for DFS results

# ============================================================================
# DFS (DARK FIELD SCATTERING) SPECIFIC SETTINGS
# ============================================================================
args['DFS_WL_RANGE'] = (500, 850)  # Wavelength range (nm) for creating DFS maximum intensity projection map
                                    # This range is used to identify the spectral region where particles scatter light most strongly

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================
args['CROP_RANGE_NM'] = (500, 850)  # Wavelength range (nm) to crop from the full hyperspectral cube
                                     # This reduces data size and focuses analysis on the region of interest

# ============================================================================
# BACKGROUND CORRECTION SETTINGS
# ============================================================================
args['BACKGROUND_MODE'] = 'global'  # Background correction method: 'global' (MATLAB style) or 'local'
                                    # Global: uses darkest pixels across entire image for background
                                    # Local: uses dark regions around each particle for individual background

# Global background parameters (MATLAB-compatible)
args['BACKGROUND_PERCENTILE'] = 0.1  # Fraction of darkest pixels to use for global background (10%)
                                      # Matches MATLAB's standard approach for background estimation

# Local background parameters (used when BACKGROUND_MODE = 'local')
args['BACKGROUND_LOCAL_SEARCH_RADIUS'] = 20  # Search radius (pixels) around each particle for local background
args['BACKGROUND_LOCAL_PERCENTILE'] = 1      # Percentile of darkest pixels within search area for background

# ============================================================================
# PARTICLE DETECTION SETTINGS
# ============================================================================
args['PARTICLE_DETECTION_STYLE'] = 'python'  # Detection algorithm: 'python' or 'matlab'
                                              # Python: threshold-based connected component analysis
                                              # MATLAB: multi-threshold single-pixel detection (partident.m compatible)

# Python-style detection parameters
args['DFS_INTENSITY_THRESHOLD'] = 0.01  # Normalized intensity threshold for particle detection
                                         # Lower values = more sensitive detection, higher values = more selective
args['MIN_PIXELS_CLUS'] = 1             # Minimum cluster size in pixels (applies to both Python & MATLAB styles)
                                         # Clusters smaller than this are rejected as noise

# MATLAB-style detection parameters (used when PARTICLE_DETECTION_STYLE = 'matlab')
args['PARTICLE_LOWER_BOUND'] = 0      # Lower threshold bound for multi-threshold detection
args['PARTICLE_UPPER_BOUND'] = 0.5    # Upper threshold bound for multi-threshold detection
args['NHOOD_SIZE'] = 1                # Neighborhood size for edge exclusion (must be odd number)
                                       # Larger values exclude more edge pixels from detection

# ============================================================================
# SPECTRAL ANALYSIS
# ============================================================================
# Representative selection parameters
args['PEAK_TOL_NM'] = 3.0     # Wavelength tolerance (nm) for grouping similar peaks
                               # Currently not actively used but available for advanced filtering

# Quality filtering parameters
args['MAX_WIDTH_NM'] = 59      # Maximum allowed FWHM (nm) for Lorentzian fits
                               # Particles with broader resonances are rejected as potentially damaged or aggregated
args['RSQ_MIN'] = 0.90         # Minimum R-squared value for accepting Lorentzian fits
                               # Ensures only high-quality spectral fits are included in analysis

# ============================================================================
# FITTING PARAMETERS - FANO MODEL
# ============================================================================
args['FIT_RANGE_NM'] = (500, 850)  # Wavelength range (nm) for fitting
args['FITTING_MODEL'] = 'fano'  # Using Fano model

# ============================================================================
# BRIGHT MODES (phase = 0 fixed)
# ============================================================================
args['NUM_BRIGHT_MODES'] = 1  # Number of bright modes (non-interacting background)

# ⚠️ CRITICAL: Initial guesses (MUST PROVIDE FOR EACH BRIGHT MODE)
args['BRIGHT_POSITION_INITIAL_GUESS'] = [650]  # Wavelengths in nm (REQUIRED!)
                                                # Example for 2 bright: [600, 750]
                                                # MUST match NUM_BRIGHT_MODES

args['BRIGHT_WIDTH_INITIAL_GUESS'] = [30]  # Linewidths γ in nm (REQUIRED!)
                                            # Example for 2 bright: [25, 35]
                                            # MUST match NUM_BRIGHT_MODES

args['BRIGHT_HEIGHT_INITIAL_GUESS'] = [1.0]  # Coupling strengths c (REQUIRED!)
                                              # Example for 2 bright: [1.0, 0.8]
                                              # MUST match NUM_BRIGHT_MODES

# Constraints
args['BRIGHT_POSITION_TOLERANCE'] = [50]  # ±nm constraint for each bright peak
                                           # Single value or list matching NUM_BRIGHT_MODES

args['BRIGHT_WIDTH_MAX'] = [60]  # Maximum width (gamma) for bright modes in nm
                                 # Single value or list matching NUM_BRIGHT_MODES

# ============================================================================
# DARK MODES (phase fitted)
# ============================================================================
args['NUM_DARK_MODES'] = 1  # Number of dark modes (interacting resonances)

# ⚠️ CRITICAL: Initial guesses (MUST PROVIDE FOR EACH DARK MODE)
args['DARK_POSITION_INITIAL_GUESS'] = [700]  # Wavelengths in nm (REQUIRED!)
                                              # Example for 2 dark: [620, 680]
                                              # MUST match NUM_DARK_MODES

args['DARK_WIDTH_INITIAL_GUESS'] = [40]  # Linewidths Γ in nm (REQUIRED!)
                                          # Example for 2 dark: [40, 50]
                                          # MUST match NUM_DARK_MODES

args['DARK_HEIGHT_INITIAL_GUESS'] = [0.5]  # Coupling strengths d (REQUIRED!)
                                            # Example for 2 dark: [0.5, 0.4]
                                            # MUST match NUM_DARK_MODES

args['DARK_PHASE_INITIAL_GUESS'] = [0.0]  # Phases θ in radians (OPTIONAL)
                                           # Default: [0.0] for all dark modes
                                           # Example for 2 dark: [0.0, 1.57]
                                           # MUST match NUM_DARK_MODES

# Constraints
args['DARK_POSITION_TOLERANCE'] = [50]  # ±nm constraint for each dark peak
                                         # Single value or list matching NUM_DARK_MODES

args['DARK_WIDTH_MAX'] = [80]  # Maximum width (Gamma) for dark modes in nm
                               # Single value or list matching NUM_DARK_MODES

# ============================================================================
# PYTORCH OPTIMIZER SETTINGS
# ============================================================================
args['USE_GPU'] = False  # Set to True if CUDA available

# Optimizer
args['OPTIMIZER'] = 'RAdam'  # 'RAdam', 'Adam', 'NAdam'

# Two-stage fitting
# Stage 1: Fit bright modes only
args['LEARNING_RATE_BRIGHT'] = 0.01
args['NUM_ITERATIONS_BRIGHT'] = 1000

# Stage 2: Add dark modes
args['LEARNING_RATE_DARK'] = 0.001
args['NUM_ITERATIONS_DARK'] = 1000

# Progress reporting
args['PRINT_EVERY'] = 100  # Print loss every N iterations

# ============================================================================
# REGULARIZATION WEIGHTS
# ============================================================================
args['REG_NEGATIVE_HEIGHT'] = 1.0       # Penalty for negative heights
args['REG_WIDTH_MAX'] = 0.1             # Penalty for exceeding width constraints
args['REG_POSITION_CONSTRAINT'] = 1.0   # Penalty for deviating from initial position

# ============================================================================
# OUTPUT AND VISUALIZATION SETTINGS
# ============================================================================
args['FIG_DPI'] = 300  # Resolution (dots per inch) for saved figures
args['OUTPUT_UNIT'] = 'eV'  # Unit for spectral output: 'nm' or 'eV'

# ============================================================================
# MANUAL COORDINATE OVERRIDE (ADVANCED USERS)
# ============================================================================
args['USE_MANUAL_COORDS'] = False  # Set to True to use manually specified coordinates
args['MANUAL_COORDS'] = [
    (30, 29), (42, 27), (42, 38), (33, 55), (33, 73),
    (15, 92), (18, 92), (40, 112), (13, 130), (14, 134),
    (11, 148), (15, 152), (19, 162), (26, 145), (29, 142),
    (33, 140), (30, 168), (39, 136), (42, 134), (21, 188),
    (23, 176), (31, 196),
]

# ============================================================================
# DEBUG AND DEVELOPMENT SETTINGS
# ============================================================================
args['DEBUG'] = True  # Enable debug mode for additional output