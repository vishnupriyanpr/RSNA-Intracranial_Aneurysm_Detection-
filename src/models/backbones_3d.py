import torch
import torch.nn as nn

class Simple3DConvNet(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
        )
        self.enc1 = nn.Sequential(
            nn.Conv3d(base, base, 3, stride=2, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base, base*2, 3, padding=1),
            nn.GroupNorm(16, base*2),
            nn.SiLU(),
            nn.Conv3d(base*2, base*2, 3, stride=2, padding=1),
            nn.GroupNorm(16, base*2),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(base*2, base*4, 3, padding=1),
            nn.GroupNorm(32, base*4),
            nn.SiLU(),
            nn.Conv3d(base*4, base*4, 3, stride=2, padding=1),
            nn.GroupNorm(32, base*4),
            nn.SiLU(),
        )
        self.out_ch = base*4

    def forward(self, x):
        x = self.stem(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        return x
