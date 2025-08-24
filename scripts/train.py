import argparse, os, yaml, torch
import pandas as pd
from torch.utils.data import DataLoader
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.dataset import RSNAAneurysmDataset
from src.models.model_zoo import Aneurysm3DGlobal
from src.models.losses import weighted_bce_with_logits
from src.training.engine import Trainer
from src.training.metrics import competition_weighted_auc
from src.training.cv import make_folds

logger = get_logger("train")

def main(args):
    set_seed(1337)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    df = pd.read_csv(args.train_csv)
    targets = cfg["targets"]
    weights_map = cfg["weights"]
    spacing_cfg = cfg["spacing"]
    window_cfg = cfg["windowing"]

    df = make_folds(df, n_splits=cfg["num_folds"], seed=cfg["seed"], target_col="Aneurysm Present")
    fold = args.fold
    trn_df = df[df.fold != fold].reset_index(drop=True)
    val_df = df[df.fold == fold].reset_index(drop=True)

    train_ds = RSNAAneurysmDataset(trn_df, args.data_root, targets, spacing_cfg, window_cfg, mode="train")
    val_ds   = RSNAAneurysmDataset(val_df, args.data_root, targets, spacing_cfg, window_cfg, mode="valid")
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = Aneurysm3DGlobal(in_ch=1, n_labels=len(targets))
    col_weights = [weights_map.get(t, weights_map.get("default", 1.0)) for t in targets]
    def criterion(logits, y, presence_extra_weight=cfg["train"]["presence_loss_weight"]):
        return weighted_bce_with_logits(logits, y, col_weights, presence_idx=0, presence_extra_weight=presence_extra_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    trainer = Trainer(model, optimizer, criterion, device="cuda", use_amp=cfg["train"]["mixed_precision"], ema=cfg["train"]["ema"])

    best = -1
    for epoch in range(cfg["train"]["epochs"]):
        trn_loss = trainer.train_one_epoch(train_loader, grad_accum_steps=cfg["train"]["grad_accum_steps"])
        probs, yval = trainer.validate(val_loader, sigmoid=True)
        from src.training.metrics import columnwise_auc
        import numpy as np
        final, aucs = competition_weighted_auc(yval, probs, targets, weights_map)
        logger.info(f"Epoch {epoch+1}: train_loss={trn_loss:.4f} final_auc={final:.4f} presence_auc={aucs[0]:.4f}")
        if final > best:
            best = final
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_fold{fold}.pt"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    main(args)
