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
args['ECHEM_SAMPLE_NAME'] = 'a15_echem_p2_cv1'  # Name of EChem spectral TDMS file (without .tdms extension)
args['ECHEM_CHI_FILE'] = 'chi_cv'     # CHI potentiostat data file name (without .txt extension)

args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/093025_AuAg_echem')  # Directory containing TDMS and CHI files
args['WHITE_FILE'] = "wc.tdms"     # White reference file name for flatfield correction
args['DARK_FILE'] = "dc.tdms"      # Dark reference file name for flatfield correction
args['OUTPUT_DIR'] = os.path.join(home, "research/pyHS", "093025_AuAg_echem/a15_echem_p2_cv1")  # Output directory

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================
args['CROP_RANGE_NM'] = (500, 1000)  # Wavelength range (nm) to crop from the full hyperspectral cube

# ============================================================================
# ECHEM EXPERIMENTAL PARAMETERS
# ============================================================================
args['ECHEM_TECHNIQUE'] = 'CV'  # Electrochemical technique: 'CV', 'CA', or 'CC'

# ============================================================================
# FITTING PARAMETERS - FANO MODEL
# ============================================================================
args['FIT_RANGE_NM'] = (500, 1000)  # Wavelength range (nm) for fitting
args['FITTING_MODEL'] = 'fano'  # Using Fano model

# ============================================================================
# BRIGHT MODES (phase = 0 fixed)
# ============================================================================
args['NUM_BRIGHT_MODES'] = 2  # Number of bright modes (non-interacting background)

# ⚠️ CRITICAL: Initial guesses (MUST PROVIDE FOR EACH BRIGHT MODE)
args['BRIGHT_POSITION_INITIAL_GUESS'] = [553, 730]  # Wavelengths in nm (REQUIRED!)
                                                     # Example for 2 bright: [550, 730]
                                                     # MUST match NUM_BRIGHT_MODES

args['BRIGHT_WIDTH_INITIAL_GUESS'] = [70, 60]  # Linewidths γ in nm (REQUIRED!)
                                                # Example for 2 bright: [70, 60]
                                                # MUST match NUM_BRIGHT_MODES

args['BRIGHT_HEIGHT_INITIAL_GUESS'] = [1.0, 0.8]  # Coupling strengths c (REQUIRED!)
                                                   # Example for 2 bright: [1.0, 0.8]
                                                   # MUST match NUM_BRIGHT_MODES

# Constraints
args['BRIGHT_POSITION_TOLERANCE'] = [10, 10]  # ±nm constraint for each bright peak
                                               # Single value or list matching NUM_BRIGHT_MODES

args['BRIGHT_WIDTH_MAX'] = [100, 70]  # Maximum width (gamma) for bright modes in nm
                                      # Single value or list matching NUM_BRIGHT_MODES

# ============================================================================
# DARK MODES (phase fitted)
# ============================================================================
args['NUM_DARK_MODES'] = 1  # Number of dark modes (interacting resonances)

# ⚠️ CRITICAL: Initial guesses (MUST PROVIDE FOR EACH DARK MODE)
args['DARK_POSITION_INITIAL_GUESS'] = [746]  # Wavelengths in nm (REQUIRED!)
                                              # Example for 2 dark: [620, 680]
                                              # MUST match NUM_DARK_MODES

args['DARK_WIDTH_INITIAL_GUESS'] = [20]  # Linewidths Γ in nm (REQUIRED!)
                                          # Example for 2 dark: [40, 50]
                                          # MUST match NUM_DARK_MODES

args['DARK_HEIGHT_INITIAL_GUESS'] = [0.5]  # Coupling strengths d (REQUIRED!)
                                            # Example for 2 dark: [0.5, 0.4]
                                            # MUST match NUM_DARK_MODES

args['DARK_PHASE_INITIAL_GUESS'] = [np.pi]  # Phases θ in radians (OPTIONAL)
                                           # Default: [0.0] for all dark modes
                                           # Example for 2 dark: [0.0, 1.57]
                                           # MUST match NUM_DARK_MODES

# Constraints
args['DARK_POSITION_TOLERANCE'] = [10]  # ±nm constraint for each dark peak
                                         # Single value or list matching NUM_DARK_MODES

args['DARK_WIDTH_MAX'] = [40]  # Maximum width (Gamma) for dark modes in nm
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
args['REG_PHASE_CONSTRAINT'] = 0.5

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
