import os, glob
import pandas as pd

def load_train_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def list_series_dirs(root: str) -> list[str]:
    # Expected form per Data tab: series/<SeriesInstanceUID>/DICOMs...
    # Adjust as needed by actual structure on the platform[8].
    if not os.path.exists(root):
        return []
    return [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
