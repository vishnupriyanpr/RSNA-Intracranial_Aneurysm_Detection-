import torch

def build_scheduler(optimizer, total_steps, warmup_steps=0, kind="cosine"):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if kind == "cosine":
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
