
#!/usr/bin/env python
"""
Entry-point:
    python run_analysis.py --sample AuNR_PMMA_1
"""
import argparse
from config import config as cfg
from data.dataset import Dataset
from data.spectrum import SpectrumAnalyzer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True,
                    help="basename without extension (e.g. AuNR_PMMA_1)")
    ap.add_argument("--image_shape", nargs=2, type=int,
                    metavar=("ROWS", "COLS"),
                    help="override automatic image-size detection")
    args = ap.parse_args()

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
    main()
