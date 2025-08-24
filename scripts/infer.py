import argparse, os, yaml, torch
import pandas as pd
from torch.utils.data import DataLoader
from src.data.dataset import RSNAAneurysmDataset
from src.models.model_zoo import Aneurysm3DGlobal
from src.inference.submit import build_submission

def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    targets = cfg["targets"]
    spacing_cfg = cfg["spacing"]
    window_cfg = cfg["windowing"]

    # Expect a test_meta.csv listing SeriesInstanceUID and Modality for test
    df = pd.read_csv(args.test_meta)
    ds = RSNAAneurysmDataset(df, args.data_root, targets, spacing_cfg, window_cfg, mode="test")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = Aneurysm3DGlobal(in_ch=1, n_labels=len(targets))
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.cuda().eval()

    ids, probs_all = [], []
    with torch.no_grad():
        for x, _, uid in loader:
            x = x.cuda(non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs[0])
            ids.append(uid)
    ids = [i if isinstance(i, str) else i[0] for i in ids]
    out = os.path.join(args.out_dir, "submission.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    build_submission(ids, probs_all, targets, out)
    print(f"Wrote {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--test_meta", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args)
