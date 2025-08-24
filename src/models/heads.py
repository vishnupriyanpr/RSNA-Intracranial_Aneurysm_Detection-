import torch
import torch.nn as nn

class GlobalMultiLabelHeads(nn.Module):
    def __init__(self, in_ch: int, n_labels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_ch, n_labels)

    def forward(self, feats):
        g = self.pool(feats).flatten(1)
        logits = self.fc(g)
        return logits  # return logits; apply sigmoid in loss or inference as needed
