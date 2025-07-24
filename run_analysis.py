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
    
    logger = setup_logger(args['OUTPUT_DIR'])
    
    try:

        print(f"[info] Starting analysis for {args['SAMPLE_NAME']}")
        print(f"[info] Output directory: {args['OUTPUT_DIR']}")
        print(f"[info] Data directory: {args['DATA_DIR']}")
        print("")
        
        dataset = Dataset(args)
        dataset.run_dataset()

        spectrum_analysis = SpectrumAnalyzer(args, dataset)
        spectrum_analysis.run_spectrum()

        print(f"\n[info] Completed  ▶  outputs saved in {args['OUTPUT_DIR']}")
        
    finally:

        close_logger()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type = str,
                        required = True,
                        help = "Path to the configuration file")
    
    opt = vars(parser.parse_args())
    
    with codecs.open(opt['config'], 'r', encoding='UTF-8') as fs:

        exec(fs.read())

    main(opt, args)