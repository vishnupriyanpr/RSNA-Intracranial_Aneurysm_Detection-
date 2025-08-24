import torch
import math

def sliding_window_infer(model, volume, patch_size=(64,128,128), overlap=0.25):
    # volume: (B=1,C,Z,Y,X)
    _, _, Z, Y, X = volume.shape
    sz, sy, sx = patch_size
    oz, oy, ox = int(sz*(1-overlap)), int(sy*(1-overlap)), int(sx*(1-overlap))
    zs = list(range(0, max(1, Z-sz+1), max(1, oz))) or [0]
    ys = list(range(0, max(1, Y-sy+1), max(1, oy))) or [0]
    xs = list(range(0, max(1, X-sx+1), max(1, ox))) or [0]
    pred_sum = None; count = 0
    for z in zs:
        for y in ys:
            for x in xs:
                patch = volume[:, :, z:z+sz, y:y+sy, x:x+sx]
                if patch.shape[2] < sz or patch.shape[3] < sy or patch.shape[4] < sx:
                    pad = (0, max(0, sx - patch.shape[4]),
                           0, max(0, sy - patch.shape[3]),
                           0, max(0, sz - patch.shape[2]))
                    patch = torch.nn.functional.pad(patch, pad)
                logits = model(patch)  # (B,C)
                if pred_sum is None:
                    pred_sum = logits
                else:
                    pred_sum += logits
                count += 1
    return pred_sum / max(1, count)
