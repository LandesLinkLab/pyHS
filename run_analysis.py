import os
import sys
import codecs
import argparse
import pickle as pkl
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable, TextIO

# from config import config as config
from data.dataset import Dataset
from spectrum.spectrum import SpectrumAnalyzer


def main(opt: Dict[str, Any], args: Dict[str, Any]) -> None:

    dataset = Dataset(args)
    dataset.run_dataset()

    spectrum_analysis = SpectrumAnalyzer(args, dataset)
    spectrum_analysis.run_spectrum()


    print(f"[info] Completed  ▶  outputs saved in {args['OUTPUT_DIR']}")


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
