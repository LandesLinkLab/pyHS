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
args['ECHEM_SAMPLE_NAME'] = 'AuAg_area09_p2_cv'  # Name of EChem spectral TDMS file (without .tdms extension)
                                            # This file contains time-series spectra during electrochemical experiment
args['ECHEM_CHI_FILE'] = 'chi_cv'     # CHI potentiostat data file name (without .txt extension)
                                            # Contains voltage, current, and charge data synchronized with spectra

args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/110525_AuAg_echem/echem')  # Directory containing TDMS and CHI files
args['WHITE_FILE'] = "wc.tdms"     # White reference file name for flatfield correction
args['DARK_FILE'] = "dc.tdms"      # Dark reference file name for flatfield correction
args['OUTPUT_DIR'] = os.path.join(home, "research/pyHS/110525_AuAg/echem/area09_p2")  # Output directory for EChem results
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
# FITTING PARAMETERS - (used when FITTING_MODEL = 'lorentzian')
# ============================================================================
# Multi-Attempt Fitting 
args['FIT_MAX_ITERATIONS'] = 1  # Number of iterative refinement cycles

args['NUM_PEAKS'] = 3  # Number of Lorentzian peaks to fit per spectrum
                       # 1: Single peak (monomers, simple nanoparticles)
                       # 2: Two peaks (dimers, coupled nanoparticles)
                       # 3+: Multiple peaks (complex coupled systems)

args['PEAK_INITIAL_GUESS'] = [560, 670, 940]  # Initial guess for peak positions
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
args['ECHEM_MAX_WIDTH_EV'] = 10000     # Maximum allowed FWHM in eV for Lorentzian fits
                                      # More lenient than DFS (~0.059 eV) due to electrochemical broadening
                                      # Broader peaks during charging/discharging are normal
                                      # Peaks broader than this are rejected as severely degraded

args['ECHEM_RSQ_MIN'] = -100.0          # Minimum R-squared value for accepting fits
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
