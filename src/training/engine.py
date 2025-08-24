import torch, math
from torch.cuda.amp import autocast, GradScaler
from ..utils.logging import get_logger

logger = get_logger("train")

class Trainer:
    def __init__(self, model, optimizer, criterion, device="cuda", use_amp=True, ema=False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)
        self.ema = ema
        self.ema_decay = 0.999
        self.ema_shadow = {k: v.clone().detach() for k, v in self.model.state_dict().items()} if ema else None

    def _update_ema(self):
        if not self.ema: return
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                self.ema_shadow[k].mul_(self.ema_decay).add_(v, alpha=1.0 - self.ema_decay)

    def _swap_to_ema(self):
        if not self.ema: return None
        current = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema_shadow, strict=True)
        return current

    def _restore_from(self, state):
        if state is not None:
            self.model.load_state_dict(state, strict=True)

    def train_one_epoch(self, loader, grad_accum_steps=1, presence_extra_weight=1.0):
        self.model.train()
        running = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        for step, (x, y, _) in enumerate(loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y, presence_extra_weight=presence_extra_weight)  # criterion wraps col_weights

            self.scaler.scale(loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self._update_ema()

            running += loss.item()

        return running / max(1, len(loader))

    @torch.no_grad()
    def validate(self, loader, sigmoid=True):
        self.model.eval()
        all_logits, all_targets = [], []
        for x, y, _ in loader:
            x = x.to(self.device, non_blocking=True)
            logits = self.model(x)
            all_logits.append(logits.cpu())
            all_targets.append(y)
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        if sigmoid:
            probs = all_logits.sigmoid().numpy()
        else:
            probs = all_logits.numpy()
        return probs, all_targets.numpy()
