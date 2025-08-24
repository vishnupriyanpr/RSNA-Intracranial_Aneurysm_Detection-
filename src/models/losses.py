import torch
import torch.nn.functional as F

def weighted_bce_with_logits(logits, targets, col_weights, presence_idx=0, presence_extra_weight=1.0):
    # logits: (B,C), targets: (B,C)
    # col_weights: list/tuple length C, with presence weight 13.0 and others 1.0 per competition[1][2]
    # presence_extra_weight: optional training-time emphasis
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    w = torch.tensor(col_weights, device=logits.device).view(1, -1)
    w = w.clone()
    w[:, presence_idx] = w[:, presence_idx] * presence_extra_weight
    loss = (bce * w).mean()
    return loss
