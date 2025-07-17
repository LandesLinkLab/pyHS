import os
import sys
import pickle
import codecs
import argparse
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable, TextIO

# from config import config as cfg
from data.dataset import Dataset
from data.spectrum import SpectrumAnalyzer


def main(opt: Dict[str, Any], args: Dict[str, Any]) -> None:

    ds = Dataset(sample_name=args.sample,
                 cfg=cfg,
                 image_shape=tuple(args.image_shape) if args.image_shape else None)
    ds.load_cube()
    ds.flatfield()
    ds.preprocess()
    ds.detect_particles()

    sa = SpectrumAnalyzer(ds, cfg)
    sa.select_representatives()
    sa.fit_and_plot()
    sa.dump_pickle()

    print(f"[✓] Completed  ▶  outputs saved in {cfg.OUTPUT_DIR}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type = str,
                        required = True,
                        help = "Path to the configuration file")
    parser.add_argument("--sample", 
                        required=True,
                        help="basename without extension (e.g. AuNR_PMMA_1)")
    parser.add_argument("--image_shape", 
                        nargs=2, 
                        type=int,
                        metavar=("ROWS", "COLS"),
                        help="override automatic image-size detection")
    
    opt = vars(parser.parse_args())
    
    with codecs.open(opt['config'], 'r', encoding='UTF-8') as fs:

        exec(fs.read())

    main(opt, args)
