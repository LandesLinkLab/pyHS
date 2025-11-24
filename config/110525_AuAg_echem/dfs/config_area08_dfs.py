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
args['SAMPLE_NAME'] = 'AuAg_area_8'  # Name of the main TDMS file (without .tdms extension)
args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/110525_AuAg_echem/hyperspectral')  # Directory containing TDMS files
args['WHITE_FILE'] = "wc.tdms"     # White reference file name for flatfield correction
args['DARK_FILE'] = "dc.tdms"      # Dark reference file name for flatfield correction
args['OUTPUT_DIR'] = os.path.join(home, "research/pyHS/110525_AuAg/hyperspectral/area_08")  

# ============================================================================
# DFS (DARK FIELD SCATTERING) SPECIFIC SETTINGS
# ============================================================================
args['DFS_WL_RANGE'] = (500, 1000)  # Wavelength range (nm) for creating DFS maximum intensity projection map
                                    # This range is used to identify the spectral region where particles scatter light most strongly

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================
args['CROP_RANGE_NM'] = (500, 1000)  # Wavelength range (nm) to crop from the full hyperspectral cube
                                     # This reduces data size and focuses analysis on the region of interest

# ============================================================================
# BACKGROUND CORRECTION SETTINGS
# ============================================================================
args['BACKGROUND_MODE'] = 'local'  # Background correction method: 'global' (MATLAB style) or 'local'
                                    # Global: uses darkest pixels across entire image for background
                                    # Local: uses dark regions around each particle for individual background

# Global background parameters (MATLAB-compatible)
args['BACKGROUND_PERCENTILE'] = 0.1  # Fraction of darkest pixels to use for global background (10%)
                                      # Matches MATLAB's standard approach for background estimation

# Local background parameters (used when BACKGROUND_MODE = 'local')
args['BACKGROUND_LOCAL_SEARCH_RADIUS'] = 5  # Search radius (pixels) around each particle for local background
args['BACKGROUND_LOCAL_PERCENTILE'] = 10      # Percentile of darkest pixels within search area for background

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
args['PEAK_TOL_NM'] = 1000.0     # Wavelength tolerance (nm) for grouping similar peaks
                               # Currently not actively used but available for advanced filtering

# Quality filtering parameters
args['MAX_WIDTH_NM'] = 1000      # Maximum allowed FWHM (nm) for Lorentzian fits
                               # Particles with broader resonances are rejected as potentially damaged or aggregated
args['RSQ_MIN'] = -100.0         # Minimum R-squared value for accepting Lorentzian fits
                               # Ensures only high-quality spectral fits are included in analysis


# ============================================================================
# FITTING PARAMETERS - SHARED
# ============================================================================
# Lorentzian fitting parameters
args['FIT_RANGE_NM'] = (500, 1000)  # Wavelength range (nm) for Lorentzian curve fitting
                                    # Should encompass the full resonance peak for accurate parameter extraction

args['FITTING_MODEL'] = 'lorentzian'  # 'lorentzian' or 'fano'
                                      # 'lorentzian': Traditional multi-peak Lorentzian fitting
                                      # 'fano': Physical Interference Model (bright + dark modes)

# ============================================================================
# FITTING PARAMETERS - (used when FITTING_MODEL = 'lorentzian')
# ============================================================================

args['NUM_PEAKS'] = 1  # Number of Lorentzian peaks to fit per spectrum
                       # 1: Single peak (monomers, simple nanoparticles)
                       # 2: Two peaks (dimers, coupled nanoparticles)
                       # 3+: Multiple peaks (complex coupled systems)

# ⚠️ CRITICAL: Initial guesses (MUST PROVIDE FOR EACH PEAK)
args['PEAK_POSITION_INITIAL_GUESS'] = [681.0]  # List of positions in nm (REQUIRED!)
                                      # Example for 2 peaks: [600, 700]
                                      # Example for 3 peaks: [600, 700, 850]
                                      # MUST be a list matching NUM_PEAKS

args['PEAK_WIDTH_INITIAL_GUESS'] = [40.0]  # List of FWHMs in nm (REQUIRED!)
                                          # Example: [30, 40] for two peaks with different widths
                                          # MUST be a list matching NUM_PEAKS

args['PEAK_HEIGHT_INITIAL_GUESS'] = [0.5]  # List of amplitudes (REQUIRED!)
                                          # Normalized (typically 0.1 - 1.0)
                                          # MUST be a list matching NUM_PEAKS

# Constraints
args['PEAK_WIDTH_MAX'] = [100.0]  # Max FWHM per peak (or single value for all)
                              # List: [50, 60] for different max per peak
                              # Single value: 60 applies to all peaks
                              
args['PEAK_POSITION_TOLERANCE'] = [200.0]  # Allowed deviation from initial guess (nm)
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

args['PEAK_MIN_DISTANCE'] = 30.0  # Minimum distance between peaks (nm) (multi-peak only)
                                   # Prevents peaks from collapsing onto each other

# ============================================================================
# PYTORCH OPTIMIZER SETTINGS
# ============================================================================

# GPU settings
args['USE_GPU'] = False  # Set to True if CUDA available

# Optimizer
args['OPTIMIZER'] = 'RAdam'  # 'RAdam', 'Adam', 'NAdam'
args['LEARNING_RATE'] = 0.01
args['NUM_ITERATIONS'] = 1
args['USE_LR_SCHEDULER'] = False  # Optional: adaptive learning rate

# Progress reporting
args['PRINT_EVERY'] = 100  # Print loss every N iterations

# ============================================================================
# REGULARIZATION WEIGHTS
# ============================================================================

# Penalty for negative heights (should be positive)
args['REG_NEGATIVE_HEIGHT'] = 1.0

# Penalty for exceeding width constraints
args['REG_WIDTH_MAX'] = 0.1

# Penalty for deviating from initial position guess
args['REG_POSITION_CONSTRAINT'] = 1.0

# Penalty for peaks being too close (multi-peak only)
args['REG_PEAK_DISTANCE'] = 0.01

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
args['USE_MANUAL_COORDS'] = True  # Set to True to use manually specified particle coordinates
                                   # When True, automatic detection is bypassed and only specified coordinates are analyzed

# Manual coordinate list (used only when USE_MANUAL_COORDS = True)
# Format: (Row, Col) - note this is (Y, X) in image coordinates, not (X, Y)
# Each coordinate will be expanded to a 3x3 pixel region for analysis
args['MANUAL_COORDS'] = [
    (9, 41),
    (96, 132),
    (85, 155),
    (53, 182),
    (71, 232),
    (96, 251)
]
# ============================================================================
# DEBUG AND DEVELOPMENT SETTINGS
# ============================================================================
args['DEBUG'] = True  # Enable debug mode for additional output and intermediate file saving
                       # When True, creates debug images and verbose console output
                       # Set to False for production runs to reduce output volume


# ============================================================================
# ⚠️ DEPRECATED PARAMETERS (No longer used with PyTorch fitting)
# ============================================================================
# These parameters are from scipy-based fitting and are IGNORED:
# - FIT_MAX_ITERATIONS
# - PEAK_POSITION_INITIAL_GUESS = 'auto'  (must now be explicit list)
#
# PyTorch requires explicit initial guesses for all parameters!


# ============================================================================
# EXAMPLE: Multi-peak Lorentzian (2 peaks)
# ============================================================================
"""
# For 2-peak fitting:
args['NUM_PEAKS'] = 2
args['PEAK_POSITION_INITIAL_GUESS'] = [600.0, 700.0]
args['PEAK_WIDTH_INITIAL_GUESS'] = [25.0, 30.0]
args['PEAK_HEIGHT_INITIAL_GUESS'] = [0.4, 0.6]
args['PEAK_WIDTH_MAX'] = [50.0, 60.0]  # Different max for each peak
args['PEAK_POSITION_TOLERANCE'] = [30.0, 40.0]  # Different tolerance
args['PEAK_MIN_DISTANCE'] = 50.0  # Peaks must be at least 50 nm apart
args['NUM_ITERATIONS'] = 2000  # More iterations for multi-peak
"""
