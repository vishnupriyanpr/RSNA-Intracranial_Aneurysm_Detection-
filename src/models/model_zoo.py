import torch.nn as nn
from .backbones_3d import Simple3DConvNet
from .heads import GlobalMultiLabelHeads

class Aneurysm3DGlobal(nn.Module):
    def __init__(self, in_ch=1, n_labels=14):
        super().__init__()
        self.backbone = Simple3DConvNet(in_ch=in_ch, base=32)
        self.heads = GlobalMultiLabelHeads(self.backbone.out_ch, n_labels)

    def forward(self, x):
        f = self.backbone(x)
        logits = self.heads(f)
        return logits
