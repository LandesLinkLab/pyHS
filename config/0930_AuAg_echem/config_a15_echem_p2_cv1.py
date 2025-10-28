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
                                            # This file contains time-series spectra during electrochemical experiment
args['ECHEM_CHI_FILE'] = 'chi_cv'     # CHI potentiostat data file name (without .txt extension)
                                            # Contains voltage, current, and charge data synchronized with spectra

args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/093025_AuAg_echem')  # Directory containing TDMS and CHI files
args['WHITE_FILE'] = "wc.tdms"     # White reference file name for flatfield correction
args['DARK_FILE'] = "dc.tdms"      # Dark reference file name for flatfield correction
args['OUTPUT_DIR'] = os.path.join(home, "research/pyHS", "093025_AuAg_echem/a15_echem_p2_cv1")  # Output directory for EChem results
                                                                            # EChem saves to: OUTPUT_DIR/echem/

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================
args['CROP_RANGE_NM'] = (500, 1000)  # Wavelength range (nm) to crop from the full hyperspectral cube
                                     # This reduces data size and focuses analysis on the region of interest
                                     # Applied after LOWERCUT/UPPERCUT pixel trimming

# ============================================================================
# ECHEM EXPERIMENTAL PARAMETERS
# ============================================================================
args['ECHEM_TECHNIQUE'] = 'CV'        # Electrochemical technique: 'CV', 'CA', or 'CC'
                                      # CV: Cyclic Voltammetry (voltage cycles)
                                      # CA: Chronoamperometry (potential steps)
                                      # CC: Chronocoulometry (potential steps with charge integration)

# ============================================================================
# FITTING PARAMETERS - SHARED
# ============================================================================
# Lorentzian fitting parameters
args['FIT_RANGE_NM'] = (500, 1000)  # Wavelength range (nm) for Lorentzian curve fitting
                                    # Should encompass the full resonance peak for accurate parameter extraction

args['FITTING_MODEL'] = 'fano'  # 'lorentzian' or 'fano'
                                      # 'lorentzian': Traditional multi-peak Lorentzian fitting
                                      # 'fano': Physical Interference Model (bright + dark modes)

# ============================================================================
# FITTING PARAMETERS - (used when FITTING_MODEL = 'fano')
# ============================================================================

# Bright modes (phase = 0 fixed)
args['NUM_BRIGHT_MODES'] = 2  # Number of bright modes (non-interacting background)
args['BRIGHT_POSITION_INITIAL_GUESS'] = [571, 720]  # Wavelengths in nm (REQUIRED, must be a list)
                                            # Example: [690, 565] for two bright peaks
args['BRIGHT_WIDTH_INITIAL_GUESS'] = [120, 130]  # REQUIRED: list of widths in nm
                                            # Example: [25, 35] for two bright modes
                                            # MUST be a list matching NUM_BRIGHT_MODES

args['BRIGHT_HEIGHT_INITIAL_GUESS'] = [0.5, 0.9]  # ⚠️ NEW REQUIRED: list of coupling strengths
                                                   # Initial guess for c_i parameters
                                                   # MUST be a list matching NUM_BRIGHT_MODES

args['BRIGHT_POSITION_TOLERANCE'] = [30, 30]  # ±nm constraint for each bright peak
                                               # Can be a single value or list matching NUM_BRIGHT_MODES
                                               # Example: 10 → all peaks ±10 nm
                                               # Example: [10, 20] → first ±10, second ±20
args['BRIGHT_WIDTH_MAX'] = [150, 100]  # Maximum width (gamma) for bright modes in nm
                                  # None: uses FANO_GAMMA_RANGE upper bound (100)
                                  # Number: constrains all bright modes (e.g., 80)
                                  # This prevents bright mode width from becoming too large                                               

# Dark modes (phase fitted)
args['NUM_DARK_MODES'] = 1  # Number of dark modes (interacting resonances)

args['DARK_POSITION_INITIAL_GUESS'] = [737]  # Wavelengths in nm (REQUIRED, must be a list)
                                     # Example: [620] for one dark mode at 620 nm
args['DARK_WIDTH_INITIAL_GUESS'] = [30]  # REQUIRED: list of widths in nm
                                          # Example: [40] for one dark mode
                                          # MUST be a list matching NUM_DARK_MODES

args['DARK_HEIGHT_INITIAL_GUESS'] = [0.3]  # ⚠️ NEW REQUIRED: list of coupling strengths d_j
                                            # MUST be a list matching NUM_DARK_MODES

args['DARK_PHASE_INITIAL_GUESS'] = [np.pi]  # ⚠️ NEW OPTIONAL: list of phases θ_j (radians)
                                           # Default: [0.0] for all dark modes
                                           # MUST be a list matching NUM_DARK_MODES

args['DARK_POSITION_TOLERANCE'] = [10]  # ±nm constraint for each dark peak
                                         # Can be a single value or list matching NUM_DARK_MODES
args['DARK_WIDTH_MAX'] = [60]  # Maximum width (Gamma) for dark modes in nm
                                # None: uses FANO_GAMMA_RANGE upper bound (100)
                                # Number: constrains all dark modes (e.g., 60)
                                # This prevents dark mode width from becoming too large

# ============================================================================
# PYTORCH OPTIMIZER SETTINGS
# ============================================================================

# GPU settings
args['USE_GPU'] = False  # Set to True if CUDA available

# Optimizer
args['OPTIMIZER'] = 'RAdam'  # 'RAdam', 'Adam', 'NAdam'

# Two-stage fitting (Fano specific)
# Stage 1: Fit bright modes only
args['LEARNING_RATE_BRIGHT'] = 0.001
args['NUM_ITERATIONS_BRIGHT'] = 1

# Stage 2: Add dark modes
args['LEARNING_RATE_DARK'] = 0.0008
args['NUM_ITERATIONS_DARK'] = 3000

# Progress reporting
args['PRINT_EVERY'] = 100  # Print loss every N iterations

# ============================================================================
# REGULARIZATION WEIGHTS
# ============================================================================

# Penalty for negative heights (should be positive)
args['REG_NEGATIVE_HEIGHT'] = 1.0
args['REG_WIDTH_MAX'] = 0.1
args['REG_POSITION_CONSTRAINT'] = 1.0
args['REG_PHASE_CONSTRAINT'] = 100.0

# ============================================================================
# ELECTROCHEMICAL REFERENCE PARAMETERS
# ============================================================================
args['ECHEM_OCP'] = 0.00              # Open circuit potential (V) used as baseline reference
                                      # Spectral changes (Δ parameters) are calculated relative to OCP
                                      # Measure this experimentally: equilibrium voltage before applying CV
                                      # 
                                      # How to measure:
                                      # 1. Prepare sample in electrochemical cell
                                      # 2. Wait for equilibration (no voltage applied)
                                      # 3. Measure stable potential → this is your OCP
                                      # 4. Set this value here before running analysis

# ============================================================================
# CYCLE/STEP ANALYSIS PARAMETERS
# ============================================================================
args['ECHEM_CYCLE_START'] = 1         # First cycle to include in averaging (1-indexed)
                                      # Use this to skip initial unstable cycles
                                      # Example: Set to 2 if first cycle shows unusual behavior

args['ECHEM_CYCLE_BACKCUT'] = 0       # Number of cycles to exclude from the end
                                      # Use this to exclude degraded final cycles
                                      # Example: Set to 1 if last cycle shows particle damage

# ============================================================================
# SPECTRAL PROCESSING PARAMETERS
# ============================================================================
args['ECHEM_LOWERCUT'] = 0          # Pixels to trim from blue (short wavelength) end of spectrum
                                      # Removes unreliable edge pixels affected by detector artifacts
                                      # Processing order: LOWERCUT/UPPERCUT (pixels) → CROP_RANGE_NM (wavelength)

args['ECHEM_UPPERCUT'] = 0          # Pixels to trim from red (long wavelength) end of spectrum
                                      # These values match MATLAB cv_analysis script parameters
                                      # Typical values: 140/260 but may need adjustment per instrument

# ============================================================================
# QUALITY FILTERING PARAMETERS
# ============================================================================
args['ECHEM_MAX_WIDTH_EV'] = 10     # Maximum allowed FWHM in eV for fits
                                      # More lenient than DFS (~0.059 eV) due to electrochemical broadening
                                      # Broader peaks during charging/discharging are normal
                                      # Peaks broader than this are rejected as severely degraded

args['ECHEM_RSQ_MIN'] = 0.0001          # Minimum R-squared value for accepting fits
                                      # More lenient than DFS (0.90) to accommodate noisier time-series data
                                      # Lower threshold accounts for rapid spectral changes during cycling

# ============================================================================
# OUTPUT AND VISUALIZATION SETTINGS
# ============================================================================
args['FIG_DPI'] = 300  # Resolution (dots per inch) for saved figures
                       # 300 DPI is publication quality, 150 DPI is suitable for presentations

# Display parameters for detailed cycle plots (reserved for future implementation)
args['ECHEM_CYCLE_PLOT_START'] = 1    # First cycle to display in detail plots
args['ECHEM_CYCLE_PLOT_END'] = 4      # Last cycle to display in detail plots
args['OUTPUT_UNIT'] = 'eV'            # Unit for spectral output: 'nm' (wavelength) or 'eV' (energy)
                                      # 'nm': Traditional wavelength units (λ in nanometers)
                                      # 'eV': Energy units (E = hc/λ = 1239.842/λ_nm)


# ============================================================================
# DEBUG AND DEVELOPMENT SETTINGS
# ============================================================================
args['DEBUG'] = True  # Enable debug mode for additional output and intermediate file saving
                       # When True, creates debug images and verbose console output
                       # Set to False for production runs to reduce output volume

