import os
import sys
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
# SPECTRAL ANALYSIS AND FITTING PARAMETERS
# ============================================================================
# Representative selection parameters
args['PEAK_TOL_NM'] = 3.0     # Wavelength tolerance (nm) for grouping similar peaks
                               # Currently not actively used but available for advanced filtering

# Quality filtering parameters
args['MAX_WIDTH_NM'] = 59      # Maximum allowed FWHM (nm) for Lorentzian fits
                               # Particles with broader resonances are rejected as potentially damaged or aggregated
args['RSQ_MIN'] = 0.90         # Minimum R-squared value for accepting Lorentzian fits
                               # Ensures only high-quality spectral fits are included in analysis

# Lorentzian fitting parameters
args['FIT_RANGE_NM'] = (500, 850)  # Wavelength range (nm) for Lorentzian curve fitting
                                    # Should encompass the full resonance peak for accurate parameter extraction

args['NUM_PEAKS'] = 1  # Number of Lorentzian peaks to fit per spectrum
                       # 1: Single peak (monomers, simple nanoparticles)
                       # 2: Two peaks (dimers, coupled nanoparticles)
                       # 3+: Multiple peaks (complex coupled systems)

args['PEAK_INITIAL_GUESS'] = 'auto'  # Initial guess for peak positions
                                      # 'auto': Automatic peak detection using scipy.signal.find_peaks
                                      # [650, 800]: Manual specification (wavelength in nm)
                                      # Must provide NUM_PEAKS values if manual
                                      # Example for 2 peaks: [650, 800]
                                      # Example for 3 peaks: [600, 700, 850]

# Peak Position Constraints
args['PEAK_POSITION_TOLERANCE'] = [20, 20, 30]  # Constrain peak positions during fitting
                                                 # None: No constraint (peaks can move freely within FIT_RANGE_NM)
                                                 # Single value (e.g., 50): All peaks constrained to ±50 nm from initial guess
                                                 # List (e.g., [20, 30, 40]): Different constraint per peak (must match NUM_PEAKS)
                                                 # 
                                                 # Example: PEAK_INITIAL_GUESS = [600, 700, 800], TOLERANCE = 30
                                                 #   → Peak 1 can only fit between 570-630 nm
                                                 #   → Peak 2 can only fit between 670-730 nm
                                                 #   → Peak 3 can only fit between 770-830 nm
                                                 #
                                                 # When to use:
                                                 # - Prevent peaks from shifting to wrong positions
                                                 # - Known approximate peak locations from theory/previous experiments
                                                 # - Avoid peak swapping in multi-peak fits

# Multi-Attempt Fitting with Intelligent Retry
args['FIT_MAX_ITERATIONS'] = 3  # Number of iterative refinement cycles
                                # Each iteration tries ALL 6 strategies and keeps the best result:
                                #   1. Current best parameters (from previous iteration)
                                #   2. Shift peaks left (-10 nm from current)
                                #   3. Shift peaks right (+10 nm from current)
                                #   4. Narrow FWHM (60% of current width)
                                #   5. Widen FWHM (150% of current width)
                                #   6. Random exploration (stochastic perturbation)
                                #
                                # The best result from iteration N becomes the starting point
                                # for iteration N+1, progressively improving R² (gradient-descent-like).
                                #
                                # Recommended values:
                                #   1: Single iteration, 6 strategy attempts (fast)
                                #   3: Three iterations, 18 total attempts (balanced)
                                #   5: Five iterations, 30 total attempts (thorough)
                                #
                                # Returns: Best parameters from final iteration

# ============================================================================
# OUTPUT AND VISUALIZATION SETTINGS
# ============================================================================
args['FIG_DPI'] = 300  # Resolution (dots per inch) for saved figures
                       # 300 DPI is publication quality, 150 DPI is suitable for presentations

args['OUTPUT_UNIT'] = 'eV'            # Unit for spectral output: 'nm' (wavelength) or 'eV' (energy)
                                      # 'nm': Traditional wavelength units (λ in nanometers)
                                      # 'eV': Energy units (E = hc/λ = 1239.842/λ_nm)

# ============================================================================
# MANUAL COORDINATE OVERRIDE (ADVANCED USERS)
# ============================================================================
args['USE_MANUAL_COORDS'] = False  # Set to True to use manually specified particle coordinates
                                   # When True, automatic detection is bypassed and only specified coordinates are analyzed

# Manual coordinate list (used only when USE_MANUAL_COORDS = True)
# Format: (Row, Col) - note this is (Y, X) in image coordinates, not (X, Y)
# Each coordinate will be expanded to a 3x3 pixel region for analysis
args['MANUAL_COORDS'] = [
    (30, 29),    # Particle 1
    (42, 27),    # Particle 2
    (42, 38),    # Particle 3
    (33, 55),    # Particle 4
    (33, 73),    # Particle 5
    (15, 92),    # Particle 6
    (18, 92),    # Particle 7
    (40, 112),   # Particle 8
    (13, 130),   # Particle 9
    (14, 134),   # Particle 10
    (11, 148),   # Particle 11
    (15, 152),   # Particle 12
    (19, 162),   # Particle 13
    (26, 145),   # Particle 14
    (29, 142),   # Particle 15
    (33, 140),   # Particle 16
    (30, 168),   # Particle 17
    (39, 136),   # Particle 18
    (42, 134),   # Particle 19
    (21, 188),   # Particle 20
    (23, 176),   # Particle 21
    (31, 196),   # Particle 22
]

# ============================================================================
# DEBUG AND DEVELOPMENT SETTINGS
# ============================================================================
args['DEBUG'] = True  # Enable debug mode for additional output and intermediate file saving
                       # When True, creates debug images and verbose console output
                       # Set to False for production runs to reduce output volume