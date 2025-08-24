import argparse, os, pandas as pd
from src.data.utils_io import load_train_csv, list_series_dirs
from src.utils.logging import get_logger

logger = get_logger("preprocess")

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = load_train_csv(args.train_csv)
    logger.info(f"Loaded train.csv: {df.shape}")
    series = list_series_dirs(args.series_root)
    logger.info(f"Found series dirs: {len(series)}")
    # Placeholder: real pipeline would cache resampled volumes/patches
    df.to_csv(os.path.join(args.out_dir, "train_meta.csv"), index=False)
    logger.info("Wrote train_meta.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--series_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args)
