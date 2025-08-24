import os
import sys
import pandas as pd

# Add the project root to Python path so we can import from src/
# Adjust this path to match your actual project location
project_root = r'C:\Users\priya\My_files\My_Projects\RSNA_Kaggle'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import the modules
try:
    from src.data.utils_io import load_train_csv, list_series_dirs
    from src.utils.logging import get_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating simple replacements for missing modules...")
    
    # Simple replacement functions if imports fail
    def load_train_csv(path):
        return pd.read_csv(path)
    
    def list_series_dirs(root):
        if not os.path.exists(root):
            return []
        return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    
    def get_logger(name):
        import logging
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        return logging.getLogger(name)

logger = get_logger("preprocess")

def main():
    # Set parameters directly - adjust these paths as needed
    train_csv = os.path.join(project_root, "data", "raw", "train.csv")
    series_root = os.path.join(project_root, "data", "raw", "series")
    out_dir = os.path.join(project_root, "data", "interim")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Load and process data
    if os.path.exists(train_csv):
        df = load_train_csv(train_csv)
        logger.info(f"Loaded train.csv: {df.shape}")
    else:
        logger.error(f"train.csv not found at: {train_csv}")
        return
    
    if os.path.exists(series_root):
        series = list_series_dirs(series_root)
        logger.info(f"Found series dirs: {len(series)}")
    else:
        logger.warning(f"Series root not found at: {series_root}, creating empty list")
        series = []
    
    # Save processed metadata
    output_file = os.path.join(out_dir, "train_meta.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"Wrote train_meta.csv to: {output_file}")

if __name__ == "__main__":
    main()
