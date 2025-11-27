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
args['ANALYSIS_MODE'] = 'echem'

# ============================================================================
# BASIC FILE AND DIRECTORY SETTINGS
# ============================================================================
args['ECHEM_SAMPLE_NAME'] = 'AuAg_area05_p1_cv'  # Name of EChem spectral TDMS file (without .tdms extension)
                                            # This file contains time-series spectra during electrochemical experiment
args['ECHEM_CHI_FILE'] = 'chi_cv'     # CHI potentiostat data file name (without .txt extension)
                                            # Contains voltage, current, and charge data synchronized with spectra

args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/110525_AuAg_echem/echem')  # Directory containing TDMS and CHI files
args['WHITE_FILE'] = "wc.tdms"     # White reference file name for flatfield correction
args['DARK_FILE'] = "dc.tdms"      # Dark reference file name for flatfield correction
args['OUTPUT_DIR'] = os.path.join(home, "research/pyHS/110525_AuAg/echem/area05_p1")  # Output directory for EChem results
                                                                            # EChem saves to: OUTPUT_DIR/echem/

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================
args['CROP_RANGE_NM'] = (500, 1000)  # Wavelength range (nm) to crop from the full hyperspectral cube

# ============================================================================
# ECHEM EXPERIMENTAL PARAMETERS
# ============================================================================
args['ECHEM_TECHNIQUE'] = 'CV'  # Electrochemical technique: 'CV', 'CA', or 'CC'

# ============================================================================
# FITTING PARAMETERS - LORENTZIAN MODEL
# ============================================================================
args['FIT_RANGE_NM'] = (500, 1000)  # Wavelength range (nm) for fitting
args['FITTING_MODEL'] = 'lorentzian'  # Using Lorentzian model

# Number of peaks
args['NUM_PEAKS'] = 3  # 1: Single peak, 2: Two peaks, 3+: Multiple peaks

# ⚠️ CRITICAL: Initial guesses (MUST PROVIDE FOR EACH PEAK)
args['PEAK_POSITION_INITIAL_GUESS'] = [580, 640, 750]  # List of positions in nm (REQUIRED!)
                                                # Example for 2 peaks: [600, 800]
                                                # MUST match NUM_PEAKS

args['PEAK_WIDTH_INITIAL_GUESS'] = [70, 40, 70]  # List of FWHMs in nm (REQUIRED!)
                                            # Example for 2 peaks: [30, 50]
                                            # MUST match NUM_PEAKS

args['PEAK_HEIGHT_INITIAL_GUESS'] = [63, 10, 4]  # List of amplitudes (REQUIRED!)
                                            # Normalized (typically 0.1 - 1.0)
                                            # MUST match NUM_PEAKS

# Constraints
args['PEAK_WIDTH_MAX'] = [120, 70, 100]  # Max FWHM per peak
                                  # Single value: applies to all peaks
                                  # List: [80, 120] for different max per peak

args['PEAK_POSITION_TOLERANCE'] = [5, 5, 30]  # Allowed deviation from initial guess (nm)
                                            # None: No constraint
                                            # Single value: applies to all peaks
                                            # List: different constraint per peak

args['PEAK_MIN_DISTANCE'] = 50.0  # Minimum distance between peaks (nm)
                                   # Only used when NUM_PEAKS > 1

# ============================================================================
# PYTORCH OPTIMIZER SETTINGS
# ============================================================================
args['USE_GPU'] = False  # Set to True if CUDA available

# Optimizer
args['OPTIMIZER'] = 'RAdam'  # 'RAdam', 'Adam', 'NAdam'
args['LEARNING_RATE'] = 0.003
args['NUM_ITERATIONS'] = 30000
args['USE_LR_SCHEDULER'] = False  # Optional: adaptive learning rate

# Progress reporting
args['PRINT_EVERY'] = 100  # Print loss every N iterations

# ============================================================================
# REGULARIZATION WEIGHTS
# ============================================================================
args['REG_NEGATIVE_HEIGHT'] = 1.0       # Penalty for negative heights
args['REG_WIDTH_MAX'] = 0.1             # Penalty for exceeding width constraints
args['REG_POSITION_CONSTRAINT'] = 1.0   # Penalty for deviating from initial position
args['REG_PEAK_DISTANCE'] = 0.01        # Penalty for peaks being too close (multi-peak)

# ============================================================================
# ELECTROCHEMICAL REFERENCE PARAMETERS
# ============================================================================
args['ECHEM_OCP'] = 0.00  # Open circuit potential (V) used as baseline reference

# ============================================================================
# CYCLE/STEP ANALYSIS PARAMETERS
# ============================================================================
args['ECHEM_CYCLE_START'] = 1   # First cycle to include in averaging (1-indexed)
args['ECHEM_CYCLE_BACKCUT'] = 0  # Number of cycles to exclude from the end

# ============================================================================
# SPECTRAL PROCESSING PARAMETERS
# ============================================================================
args['ECHEM_LOWERCUT'] = 0  # Pixels to trim from blue (short wavelength) end of spectrum
args['ECHEM_UPPERCUT'] = 0  # Pixels to trim from red (long wavelength) end of spectrum

# ============================================================================
# QUALITY FILTERING PARAMETERS
# ============================================================================
args['ECHEM_MAX_WIDTH_EV'] = 10      # Maximum allowed FWHM in eV for fits
args['ECHEM_RSQ_MIN'] = 0.0001       # Minimum R-squared value for accepting fits

# ============================================================================
# OUTPUT AND VISUALIZATION SETTINGS
# ============================================================================
args['FIG_DPI'] = 300  # Resolution (dots per inch) for saved figures
args['ECHEM_CYCLE_PLOT_START'] = 1  # First cycle to display in detail plots
args['ECHEM_CYCLE_PLOT_END'] = 4    # Last cycle to display in detail plots
args['OUTPUT_UNIT'] = 'eV'  # Unit for spectral output: 'nm' or 'eV'

# ============================================================================
# DEBUG AND DEVELOPMENT SETTINGS
# ============================================================================
args['DEBUG'] = True  # Enable debug mode for additional output