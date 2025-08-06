import os
import sys
import codecs
import argparse
import pickle as pkl
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable, TextIO

from data.dataset import Dataset
from spectrum.spectrum import SpectrumAnalyzer
from logger import setup_logger, close_logger


def main(opt: Dict[str, Any], args: Dict[str, Any]) -> None:
    """
    Main execution function for DFS hyperspectral analysis pipeline
    
    This function orchestrates the complete analysis workflow including:
    1. Setting up logging system for output capture
    2. Creating and running dataset preprocessing
    3. Performing spectral analysis and fitting
    4. Generating all output files and visualizations
    5. Proper cleanup of logging system
    
    The function is designed to be robust with proper error handling
    and cleanup even if analysis fails partway through.
    
    Parameters:
    -----------
    opt : Dict[str, Any]
        Command line options from argparse
        Contains: config - path to configuration file
    args : Dict[str, Any]
        Configuration parameters loaded from config file
        Contains all analysis parameters including:
        - SAMPLE_NAME: Name of sample to analyze
        - DATA_DIR: Directory containing TDMS files
        - OUTPUT_DIR: Directory for saving results
        - All detection and analysis parameters
    """
    
    # Set up dual logging (console + file) for complete output capture
    logger = setup_logger(args['OUTPUT_DIR'])
    
    try:
        # Print initial configuration information
        print(f"[info] Starting analysis for {args['SAMPLE_NAME']}")
        print(f"[info] Output directory: {args['OUTPUT_DIR']}")
        print(f"[info] Data directory: {args['DATA_DIR']}")
        print("")
        
        # Step 1: Create dataset object and run preprocessing pipeline
        # This includes: TDMS loading, wavelength cropping, flatfield correction,
        # max intensity map creation, particle detection, and background correction
        dataset = Dataset(args)
        dataset.run_dataset()

        # Step 2: Create spectrum analyzer and run spectral analysis pipeline
        # This includes: Lorentzian fitting, quality filtering, representative selection,
        # plot generation, data export, and summary statistics
        spectrum_analysis = SpectrumAnalyzer(args, dataset)
        spectrum_analysis.run_spectrum()

        print(f"\n[info] Completed  ▶  outputs saved in {args['OUTPUT_DIR']}")
        
    finally:
        # Always restore original stdout and close log file
        # This ensures proper cleanup even if analysis fails
        close_logger()


if __name__ == "__main__":
    """
    Command line interface for the DFS analysis pipeline
    
    This script expects a single command line argument:
    --config: Path to a Python configuration file containing analysis parameters
    
    The configuration file should define an 'args' dictionary with all necessary
    parameters for the analysis pipeline.
    
    Example usage:
    python run_analysis.py --config path/to/config.py
    """

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="DFS Hyperspectral Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_analysis.py --config configs/sample_config.py

The configuration file must be a valid Python file that defines
an 'args' dictionary containing all analysis parameters.
        """)
    
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help="Path to the Python configuration file")
    
    # Parse command line arguments
    opt = vars(parser.parse_args())
    
    # Load and execute configuration file
    # The config file should define an 'args' dictionary in its global scope
    try:
        with codecs.open(opt['config'], 'r', encoding='UTF-8') as fs:
            # Execute the configuration file to load 'args' variable
            exec(fs.read())
        
        # The 'args' variable should now be defined in the local scope
        # Call main analysis function with both command line options and config parameters
        main(opt, args)
        
    except FileNotFoundError:
        print(f"Error: Configuration file '{opt['config']}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {str(e)}")
        sys.exit(1)