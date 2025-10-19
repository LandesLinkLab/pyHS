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
    Main execution function for hyperspectral analysis pipeline
    
    This function orchestrates the complete analysis workflow including:
    - DFS (Dark Field Scattering) analysis for spatial mapping
    - EChem (Electrochemical) analysis for time-series spectroscopy
    
    The analysis mode is determined by args['ANALYSIS_MODE']:
    - 'dfs': Traditional DFS particle detection and analysis
    - 'echem': Electrochemical CV/CA/CC spectroscopy
    - 'both': Sequential execution of both pipelines
    
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
        
        # Execute analysis based on mode
        if analysis_mode == 'dfs':
            run_dfs_analysis(args)
        elif analysis_mode == 'echem':
            run_echem_analysis(args)
        elif analysis_mode == 'both':
            print("[info] Running DFS analysis first...")
            run_dfs_analysis(args)
            print("\n[info] Running EChem analysis...")
            run_echem_analysis(args)
        else:
            raise ValueError(f"Unknown ANALYSIS_MODE: {analysis_mode}. Use 'dfs', 'echem', or 'both'")

        print(f"\n[info] Completed  ▶  outputs saved in {args['OUTPUT_DIR']}")
        
    finally:
        # Always restore original stdout and close log file
        close_logger()


def run_dfs_analysis(args: Dict[str, Any]) -> None:
    """
    Execute DFS (Dark Field Scattering) analysis pipeline
    
    This includes:
    - Loading hyperspectral image cube (H × W × λ)
    - Particle detection and clustering
    - Lorentzian fitting for each particle
    - Statistical analysis and visualization
    
    Parameters:
    -----------
    args : Dict[str, Any]
        Configuration parameters for DFS analysis
    """
    print("\n" + "="*60)
    print("DFS ANALYSIS PIPELINE")
    print("="*60)
    print(f"[info] Sample: {args['SAMPLE_NAME']}")
    print(f"[info] Data directory: {args['DATA_DIR']}")
    
    # Step 1: Create dataset object and run preprocessing pipeline
    dataset = Dataset(args)
    dataset.run_dataset()

    # Step 2: Create spectrum analyzer and run spectral analysis pipeline
    spectrum_analysis = SpectrumAnalyzer(args, dataset)
    spectrum_analysis.run_spectrum()
    
    print("\n[info] DFS analysis complete")


def run_echem_analysis(args: Dict[str, Any]) -> None:
    """
    Execute EChem (Electrochemical) analysis pipeline
    
    This includes:
    - Loading time-series spectral data (Time × λ)
    - Loading potentiostat data (voltage, current, charge)
    - Lorentzian fitting for each time point
    - CV/CA/CC cycle analysis
    - Correlation with electrochemical parameters
    
    Results are saved to: OUTPUT_DIR/echem/
    
    Parameters:
    -----------
    args : Dict[str, Any]
        Configuration parameters for EChem analysis
    """
    print("\n" + "="*60)
    print("ECHEM ANALYSIS PIPELINE")
    print("="*60)
    print(f"[info] Sample: {args['ECHEM_SAMPLE_NAME']}")
    print(f"[info] Data directory: {args['DATA_DIR']}")
    print(f"[info] EChem output: {args['OUTPUT_DIR']}/echem/")
    
    # Import echem modules (lazy import to avoid dependency if not needed)
    from echem.echem_dataset import EChemDataset
    from echem.echem_analysis import EChemAnalyzer
    
    # Step 1: Load TDMS spectral data and potentiostat data
    dataset = EChemDataset(args)
    dataset.run_dataset()
    
    # Step 2: Perform electrochemical spectroscopy analysis
    analyzer = EChemAnalyzer(args, dataset)
    analyzer.run_analysis()
    
    print("\n[info] EChem analysis complete")


if __name__ == "__main__":
    """
    Command line interface for the hyperspectral analysis pipeline
    
    This script expects a single command line argument:
    --config: Path to a Python configuration file containing analysis parameters
    
    The configuration file should define an 'args' dictionary with all necessary
    parameters for the analysis pipeline.
    
    Example usage:
    python run_analysis.py --config config/config_dfs.py
    python run_analysis.py --config config/echem/config_echem.py
    """

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Hyperspectral Analysis Pipeline (DFS + EChem)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # DFS analysis only
  python run_analysis.py --config config/config_dfs.py
  
  # EChem analysis only
  python run_analysis.py --config config/echem/config_echem.py
  
  # Both analyses sequentially
  python run_analysis.py --config config/config_both.py

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