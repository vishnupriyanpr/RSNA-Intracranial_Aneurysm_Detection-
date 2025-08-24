import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def make_folds(df: pd.DataFrame, n_splits=5, seed=1337, target_col="Aneurysm Present"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df = df.copy()
    df["fold"] = -1
    y = df[target_col].astype(int).values
    for fold, (_, val_idx) in enumerate(skf.split(df, y)):
        df.loc[val_idx, "fold"] = fold
    return df
