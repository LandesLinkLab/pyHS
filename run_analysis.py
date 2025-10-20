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
    
    Parameters:
    -----------
    opt : Dict[str, Any]
        Command line options from argparse
    args : Dict[str, Any]
        Configuration parameters loaded from config file
    """
    
    # Set up dual logging (console + file)
    logger = setup_logger(args['OUTPUT_DIR'])
    
    try:
        # Determine analysis mode
        analysis_mode = args.get('ANALYSIS_MODE', 'dfs')
        
        print(f"[info] Analysis Mode: {analysis_mode.upper()}")
        print(f"[info] Output directory: {args['OUTPUT_DIR']}")
        print("")
        
        # Load and preprocess data
        print(f"[info] Loading dataset...")
        dataset = Dataset(args)
        dataset.run_dataset()
        
        # Run analysis based on mode
        if analysis_mode == 'dfs':
            print(f"\n[info] Running DFS analysis...")
            analyzer = SpectrumAnalyzer(args, dataset)
            analyzer.run_spectrum()
        
        elif analysis_mode == 'echem':
            print(f"\n[info] Running EChem analysis...")
            analyzer = EChemAnalyzer(args, dataset)
            analyzer.run_analysis()
        
        else:
            raise ValueError(f"Unknown ANALYSIS_MODE: {analysis_mode}")

        print(f"\n[info] Completed ▶ outputs saved in {args['OUTPUT_DIR']}")
        
    finally:
        close_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    opt = vars(parser.parse_args())
    
    # Load config and run
    with codecs.open(opt['config'], 'r', encoding='UTF-8') as fs:
        exec(fs.read())
    
    main(opt, args)