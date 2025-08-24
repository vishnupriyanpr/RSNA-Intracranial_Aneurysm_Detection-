import numpy as np
from src.training.metrics import competition_weighted_auc

def test_competition_weighted_auc():
    y_true = np.array([[1,0,0],[0,1,0],[1,1,1]])
    y_prob = np.array([[0.9,0.1,0.2],[0.2,0.8,0.3],[0.8,0.7,0.6]])
    names = ["Aneurysm Present","LocA","LocB"]
    weights = {"Aneurysm Present":13.0,"default":1.0}
    final, aucs = competition_weighted_auc(y_true, y_prob, names, weights)
    assert 0.0 <= final <= 1.0
    assert len(aucs) == 3
