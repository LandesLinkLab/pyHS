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
args['ANALYSIS_MODE'] = 'echem'

# ============================================================================
# BASIC FILE AND DIRECTORY SETTINGS
# ============================================================================
args['ECHEM_SAMPLE_NAME'] = 'a15_echem_p4_cv4'  # Name of EChem spectral TDMS file (without .tdms extension)
                                            # This file contains time-series spectra during electrochemical experiment
args['ECHEM_CHI_FILE'] = 'chi_cv'     # CHI potentiostat data file name (without .txt extension)
                                            # Contains voltage, current, and charge data synchronized with spectra

args['DATA_DIR'] = os.path.join(home, 'dataset/pyHS/093025_AuAg_echem')  # Directory containing TDMS and CHI files
args['WHITE_FILE'] = "wc.tdms"     # White reference file name for flatfield correction
args['DARK_FILE'] = "dc.tdms"      # Dark reference file name for flatfield correction
args['OUTPUT_DIR'] = os.path.join(home, "research/pyHS", "093025_AuAg_echem")  # Output directory for EChem results
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
# SPECTRAL FITTING PARAMETERS
# ============================================================================
args['FIT_RANGE_NM'] = (500, 1000)    # Wavelength range (nm) for Lorentzian curve fitting
                                      # Should encompass the full resonance peak for accurate parameter extraction
                                      # Usually same as CROP_RANGE_NM         

args['NUM_PEAKS'] = 3  # Number of Lorentzian peaks to fit per spectrum
                       # 1: Single peak (monomers, simple nanoparticles)
                       # 2: Two peaks (dimers, coupled nanoparticles)
                       # 3+: Multiple peaks (complex coupled systems)

args['PEAK_INITIAL_GUESS'] = [580, 700, 760]  # Initial guess for peak positions
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
args['FIT_MAX_ITERATIONS'] = 100  # Number of iterative refinement cycles
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
args['ECHEM_MAX_WIDTH_EV'] = 10     # Maximum allowed FWHM in eV for Lorentzian fits
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

# ============================================================================
# CONFIGURATION NOTES
# ============================================================================
"""
EChem Configuration Guide:

Quick Start:
1. Set ECHEM_SAMPLE_NAME and ECHEM_CHI_FILE (same name usually)
2. Measure and set ECHEM_OCP before experiment
3. Run: python run_analysis.py --echem-config config/echem/config_echem.py

Data Processing Order:
1. Load TDMS file (full wavelength range)
2. Apply ECHEM_LOWERCUT/UPPERCUT (remove edge pixels)
3. Apply CROP_RANGE_NM (crop to wavelength range)
4. Flatfield correction with white/dark references
5. Lorentzian fitting for each time point
6. Cycle detection and averaging
7. Generate plots and save results

Typical Parameter Adjustment:
- ECHEM_LOWERCUT/UPPERCUT: Check debug plots, adjust if edge artifacts visible
- ECHEM_OCP: Must measure experimentally for each sample
- ECHEM_CYCLE_START: Set to 2 if first cycle is unstable
- ECHEM_MAX_WIDTH_EV/RSQ_MIN: Relax if too many fits rejected, tighten if quality poor

File Structure:
DATA_DIR/
├── AuNR_CV_001.tdms      # Spectral time-series data
├── AuNR_CV_001.txt       # CHI potentiostat data
├── wc.tdms               # White reference
└── dc.tdms               # Dark reference

Output Structure:
OUTPUT_DIR/
├── echem/
│   ├── debug/
│   │   ├── *_spectral_heatmap.png
│   │   ├── *_voltage_trace.png
│   │   └── *_sample_spectra.png
│   ├── spectra/
│   │   ├── *_spectrum_0001.txt
│   │   ├── *_spectrum_0002.txt
│   │   └── *_cycle_01.txt
│   ├── *_overview.png
│   ├── *_cycle_averaged.png
│   └── *_echem_results.pkl
└── log.txt
"""
