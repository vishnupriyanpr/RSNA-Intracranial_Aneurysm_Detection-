import numpy as np
from sklearn.metrics import roc_auc_score

def columnwise_auc(y_true, y_prob):
    aucs = []
    for i in range(y_true.shape[1]):
        try:
            aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
        except ValueError:
            aucs.append(0.5)
    return np.array(aucs)

def competition_weighted_auc(y_true, y_prob, col_names, weights_map):
    aucs = columnwise_auc(y_true, y_prob)
    weights = np.array([weights_map.get(name, weights_map.get("default", 1.0)) for name in col_names])
    # Per Evaluation: final = average of presence AUC and average of other 13 AUCs[1][2]
    presence_idx = 0
    presence_auc = aucs[presence_idx]
    other_mask = np.ones_like(aucs, dtype=bool); other_mask[presence_idx] = False
    other_mean = aucs[other_mask].mean() if other_mask.any() else 0.0
    final = (presence_auc + other_mean) / 2.0
    return final, aucs
