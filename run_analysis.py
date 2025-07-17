import os
import sys
import pickle
import codecs
import argparse
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable, TextIO

# from config import config as config
from data.dataset import Dataset
from data.spectrum import SpectrumAnalyzer


def main(opt: Dict[str, Any], args: Dict[str, Any]) -> None:

    os.path

    dataset = Dataset(args,
                    sample_name=opt['sample'],
                    image_shape=tuple(opt['image_shape']) if opt['image_shape'] else None)
    
    dataset.run_dataset()

    spectrum_analysis = SpectrumAnalyzer(dataset, config)


    print(f"[info] Completed  ▶  outputs saved in {config.OUTPUT_DIR}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type = str,
                        required = True,
                        help = "Path to the configuration file")
    parser.add_argument("--image_shape", 
                        nargs=2, 
                        type=int,
                        metavar=("ROWS", "COLS"),
                        help="override automatic image-size detection")
    
    opt = vars(parser.parse_args())
    
    with codecs.open(opt['config'], 'r', encoding='UTF-8') as fs:

        exec(fs.read())

    main(opt, args)
