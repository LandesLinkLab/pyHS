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
args['ECHEM_SAMPLE_NAME'] = 'AuNR_CV_001'  # Name of EChem spectral TDMS file (without .tdms extension)
                                            # This file contains time-series spectra during electrochemical experiment
args['ECHEM_CHI_FILE'] = 'AuNR_CV_001'     # CHI potentiostat data file name (without .txt extension)
                                            # Contains voltage, current, and charge data synchronized with spectra

args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/raw_jmkim')  # Directory containing TDMS and CHI files
args['WHITE_FILE'] = "wc.tdms"     # White reference file name for flatfield correction
args['DARK_FILE'] = "dc.tdms"      # Dark reference file name for flatfield correction
args['OUTPUT_DIR'] = os.path.join(home, "research", "pyHS_echem_output")  # Output directory for EChem results
                                                                            # EChem saves to: OUTPUT_DIR/echem/

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================
args['CROP_RANGE_NM'] = (500, 850)  # Wavelength range (nm) to crop from the full hyperspectral cube
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
args['FIT_RANGE_NM'] = (500, 850)  # Wavelength range (nm) for Lorentzian curve fitting
                                    # Should encompass the full resonance peak for accurate parameter extraction

args['FITTING_MODEL'] = 'lorentzian'  # 'lorentzian' or 'fano'
                                      # 'lorentzian': Traditional multi-peak Lorentzian fitting
                                      # 'fano': Physical Interference Model (bright + dark modes)                                 

# ============================================================================
# FITTING PARAMETERS - (used when FITTING_MODEL = 'lorentzian')
# ============================================================================
# Multi-Attempt Fitting 
args['FIT_MAX_ITERATIONS'] = 100  # Number of iterative refinement cycles                                      

args['NUM_PEAKS'] = 1  # Number of Lorentzian peaks to fit per spectrum
                       # 1: Single peak (monomers, simple nanoparticles)
                       # 2: Two peaks (dimers, coupled nanoparticles)
                       # 3+: Multiple peaks (complex coupled systems)

args['PEAK_POSITION_INITIAL_GUESS'] = 'auto'  # Initial guess for peak positions
                                      # 'auto': Automatic peak detection using scipy.signal.find_peaks
                                      # [650, 800]: Manual specification (wavelength in nm)
                                      # Must provide NUM_PEAKS values if manual
                                      # Example for 2 peaks: [650, 800]
                                      # Example for 3 peaks: [600, 700, 850]

args['PEAK_WIDTH_INITIAL_GUESS'] = None  # None (auto) or list of widths in nm
                                          # Example: [30, 40] for two peaks with different widths
                                          # If None, width is auto-estimated from data
                                          # MUST be a list matching NUM_PEAKS if provided

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

args['PEAK_WIDTH_MAX'] = 59  # Maximum allowed width (FWHM) in nm
                              # None for no limit, or a number (e.g., 59)
                              # This replaces the old filtering-only approach
                              # Now width is constrained DURING fitting 

# ============================================================================
# FITTING PARAMETERS - (used when FITTING_MODEL = 'fano')
# ============================================================================
args['FIT_BRIGHT_ITERATIONS'] = 50   # Step 1: Bright only iteration
args['FIT_DARK_ITERATIONS'] = 10    # Step 2: Dark only iteration

# Bright modes (phase = 0 fixed)
args['NUM_BRIGHT_MODES'] = 2  # Number of bright modes (non-interacting background)
args['BRIGHT_POSITION_INITIAL_GUESS'] = [690, 565]  # Wavelengths in nm (REQUIRED, must be a list)
                                            # Example: [690, 565] for two bright peaks

args['BRIGHT_WIDTH_INITIAL_GUESS'] = None  # None (auto=30 nm) or list of widths
                                            # Example: [25, 35] for two bright modes
                                            # MUST be a list matching NUM_BRIGHT_MODES if provided

args['BRIGHT_POSITION_TOLERANCE'] = [10, 10]  # ±nm constraint for each bright peak
                                               # Can be a single value or list matching NUM_BRIGHT_MODES
                                               # Example: 10 → all peaks ±10 nm
                                               # Example: [10, 20] → first ±10, second ±20

args['BRIGHT_WIDTH_MAX'] = None  # Maximum width (gamma) for bright modes in nm
                                  # None: uses FANO_GAMMA_RANGE upper bound (100)
                                  # Number: constrains all bright modes (e.g., 80)
                                  # This prevents bright mode width from becoming too large

# Dark modes (phase fitted)
args['NUM_DARK_MODES'] = 1  # Number of dark modes (interacting resonances)
args['DARK_POSITION_INITIAL_GUESS'] = [620]  # Wavelengths in nm (REQUIRED, must be a list)
                                     # Example: [620] for one dark mode at 620 nm

args['DARK_WIDTH_INITIAL_GUESS'] = None  # None (auto=30 nm) or list of widths
                                          # Example: [40] for one dark mode
                                          # MUST be a list matching NUM_DARK_MODES if provided

args['DARK_POSITION_TOLERANCE'] = [10]  # ±nm constraint for each dark peak
                                         # Can be a single value or list matching NUM_DARK_MODES

args['DARK_WIDTH_MAX'] = None  # Maximum width (Gamma) for dark modes in nm
                                # None: uses FANO_GAMMA_RANGE upper bound (100)
                                # Number: constrains all dark modes (e.g., 60)
                                # This prevents dark mode width from becoming too large

# Fano-specific fitting parameters
args['FANO_PHI_INIT'] = np.pi  # Initial phase for dark modes (radians)
                               # Default: π (180 degrees)
                               # Typical range: 0 to 2π

args['FANO_Q_RANGE'] = (-20, 20)  # Amplitude range for both bright (c_i) and dark (d_j) modes
                                   # Negative values allow for destructive interference
                                   # Example: (-20, 20) allows strong interference effects

args['FANO_PHI_RANGE'] = (0, 2*np.pi)  # Phase range for dark modes (radians)
                                        # Full range: (0, 2π) allows all possible phases

args['FANO_GAMMA_RANGE'] = (5, 100)  # Linewidth range in nm for both γ (bright) and Γ (dark)
                                      # (5, 100): typical range for plasmonic resonances
                                      # Adjust based on expected resonance widths

# Debug mode
args['FANO_DEBUG'] = True  # If True, prints detailed fitting information
                           # Shows Step 1 and Step 2 results with parameters
                           # Useful for troubleshooting and understanding fitting process

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
args['ECHEM_LOWERCUT'] = 140          # Pixels to trim from blue (short wavelength) end of spectrum
                                      # Removes unreliable edge pixels affected by detector artifacts
                                      # Processing order: LOWERCUT/UPPERCUT (pixels) → CROP_RANGE_NM (wavelength)

args['ECHEM_UPPERCUT'] = 260          # Pixels to trim from red (long wavelength) end of spectrum
                                      # These values match MATLAB cv_analysis script parameters
                                      # Typical values: 140/260 but may need adjustment per instrument

# ============================================================================
# QUALITY FILTERING PARAMETERS
# ============================================================================
args['ECHEM_MAX_WIDTH_EV'] = 0.15     # Maximum allowed FWHM in eV for Lorentzian fits
                                      # More lenient than DFS (~0.059 eV) due to electrochemical broadening
                                      # Broader peaks during charging/discharging are normal
                                      # Peaks broader than this are rejected as severely degraded

args['ECHEM_RSQ_MIN'] = 0.85          # Minimum R-squared value for accepting fits
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
