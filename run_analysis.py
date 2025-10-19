import os
import sys
import codecs
import argparse
import pickle as pkl
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable, TextIO

from data.dataset import Dataset
from spectrum.spectrum import SpectrumAnalyzer
from echem.echem import EChemAnalyzer
from logger import setup_logger, close_logger


def main(opt: Dict[str, Any], args: Dict[str, Any]) -> None:
    """
    Main execution function for hyperspectral analysis pipeline
    
    This function orchestrates the complete analysis workflow including:
    - DFS (Dark Field Scattering) analysis for spatial mapping
    - EChem (Electrochemical) analysis for time-series spectroscopy
    
    The analysis mode is determined by args['ANALYSIS_MODE']:
    - 'dfs': Traditional DFS particle detection and analysis
    - 'echem': Electrochemical CV/CA/CC spectroscopy
    
    Parameters:
    -----------
    opt : Dict[str, Any]
        Command line options from argparse
        Contains: config - path to configuration file
    args : Dict[str, Any]
        Configuration parameters loaded from config file
        Contains all analysis parameters
    """
    
    # Set up dual logging (console + file) for complete output capture
    logger = setup_logger(args['OUTPUT_DIR'])
    
    try:
        # Determine analysis mode
        analysis_mode = args.get('ANALYSIS_MODE', 'dfs')
        
        print(f"[info] Analysis Mode: {analysis_mode.upper()}")
        print(f"[info] Output directory: {args['OUTPUT_DIR']}")
        print("")
        
        # Step 1: Load and preprocess data using unified Dataset
        print(f"[info] Loading dataset...")
        dataset = Dataset(args)  # mode is auto-detected from args['ANALYSIS_MODE']
        dataset.run_dataset()
        
        # Step 2: Run analysis based on mode
        if analysis_mode == 'dfs':
            print(f"\n[info] Running DFS analysis...")
            analyzer = SpectrumAnalyzer(args, dataset)
            analyzer.run_spectrum()
        
        elif analysis_mode == 'echem':
            print(f"\n[info] Running EChem analysis...")
            analyzer = EChemAnalyzer(args, dataset)
            analyzer.run_analysis()
        
        else:
            raise ValueError(f"Unknown ANALYSIS_MODE: {analysis_mode}. Use 'dfs' or 'echem'")

        print(f"\n[info] Completed ▶ outputs saved in {args['OUTPUT_DIR']}")
        
    finally:
        # Always restore original stdout and close log file
        close_logger()


if __name__ == "__main__":
    """
    Command line interface for the hyperspectral analysis pipeline
    
    This script expects a single command line argument:
    --config: Path to a Python configuration file containing analysis parameters
    
    The configuration file should define an 'args' dictionary with all necessary
    parameters for the analysis pipeline.
    
    Example usage:
    python run_analysis.py --config config/dfs/config_dfs.py
    python run_analysis.py --config config/echem/config_echem.py
    """

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Hyperspectral Analysis Pipeline (DFS + EChem)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # DFS analysis
  python run_analysis.py --config config/dfs/config_dfs.py
  
  # EChem analysis
  python run_analysis.py --config config/echem/config_echem.py

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
        import traceback
        traceback.print_exc()
        sys.exit(1)