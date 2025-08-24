import torch

def simple_tta(model, x, tta=3):
    # tta flips along depth and width; average probabilities
    outs = []
    logits = model(x)
    outs.append(logits)
    if tta >= 2:
        outs.append(model(torch.flip(x, dims=[2])))  # flip Z
    if tta >= 3:
        outs.append(model(torch.flip(x, dims=[3])))  # flip Y
    logits = torch.stack(outs, dim=0).mean(0)
    return logits
