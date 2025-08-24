import numpy as np
import random

def random_flip_3d(x: np.ndarray):
    # x: (C,Z,Y,X) or (Z,Y,X)
    if x.ndim == 4:
        axes = [1,2,3]
    else:
        axes = [0,1,2]
    for ax in axes:
        if random.random() < 0.5:
            x = np.flip(x, axis=ax).copy()
    return x

def intensity_gamma(x: np.ndarray, low=0.9, high=1.1):
    g = random.uniform(low, high)
    x = np.clip(x, 1e-6, 1.0) ** g
    return x

def train_augments(x: np.ndarray):
    x = random_flip_3d(x)
    x = intensity_gamma(x)
    return x
