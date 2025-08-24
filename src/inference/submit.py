import pandas as pd
import numpy as np

def build_submission(ids, probs, columns, out_path):
    df = pd.DataFrame(probs, columns=columns)
    df.insert(0, "SeriesInstanceUID", ids)
    df.to_csv(out_path, index=False)
    return out_path

