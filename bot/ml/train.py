"""Train the CNN on 2016-2021, early-stop on 2022 validation (rank IC of the 24h head).

    python -m bot.ml.train --epochs 20 --hidden 64 --tag base
Writes models/<tag>.pt plus models/<tag>.json with val metrics per epoch.

Deployment refit (frozen recipe, no early stopping): extend the training window and run
a fixed number of epochs, saving the final epoch. Monitoring metrics are then computed on
the holdout split (2026 YTD) because 2022 is inside the training window.

    python -m bot.ml.train --tag deploy --train-end 2026-01-01 --epochs 3 --select-head 24
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import bot.ml.dataset as dataset
from bot.ml.dataset import WindowDataset, get_panel, sample_index
from bot.ml.features import HORIZONS, targets
from bot.ml.model import CNN

MODELS = Path(__file__).resolve().parents[2] / "models"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def rank_ic(pred: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    """Mean per-hour Spearman correlation between predictions and realised targets."""
    df = pd.DataFrame({"t": idx[:, 0], "p": pred, "y": y})
    ics = df.groupby("t")[["p", "y"]].apply(lambda g: g["p"].corr(g["y"], method="spearman") if len(g) >= 5 else np.nan)
    return float(np.nanmean(ics))


@torch.no_grad()
def predict(model: nn.Module, ds: WindowDataset, bs: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for batch in ds.batches(bs):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        out.append(model(x.to(DEVICE)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--subsample", type=float, default=1.0, help="fraction of train samples per epoch")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-head", type=int, default=336, help="horizon whose val rank-IC drives early stopping")
    ap.add_argument("--train-end", default=None, help="refit mode: train on 2016-01-01..TRAIN_END (minus embargo), "
                    "no early stopping, save the final epoch; metrics reported on the holdout split")
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience (epochs)")
    args = ap.parse_args()
    refit = args.train_end is not None
    if refit:
        dataset.SPLITS["train"] = ("2016-01-01", args.train_end)   # process-local override; embargo still applied
    monitor_split = "holdout" if refit else "val"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feat, vol, lc, valid, times, coins = get_panel()
    y = targets(lc, vol)
    tr_idx = sample_index(valid, times, "train", y)
    va_idx = sample_index(valid, times, monitor_split, y)
    print(f"train {len(tr_idx):,} ({dataset.SPLITS['train'][0]}..{dataset.SPLITS['train'][1]} minus embargo) "
          f"{monitor_split} {len(va_idx):,} samples, device {DEVICE}"
          f"{' [refit: fixed epochs, no early stop]' if refit else ''}", flush=True)
    tr_ds, va_ds = WindowDataset(feat, y, tr_idx), WindowDataset(feat, y, va_idx)
    va_y = y[va_idx[:, 0], va_idx[:, 1]]

    model = CNN(hidden=args.hidden, drop=args.drop).to(DEVICE)
    print(f"params {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = int(len(tr_ds) * args.subsample) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    loss_fn = nn.HuberLoss(delta=1.0)
    hw = torch.tensor([0.3, 0.3, 0.4], device=DEVICE)  # slight emphasis on the 14-day head

    MODELS.mkdir(exist_ok=True)
    best, hist, bad = -1e9, [], 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for i, (x, yb) in enumerate(tr_ds.batches(args.bs, shuffle=True, n_samples=steps_per_epoch * args.bs)):
            x, yb = x.to(DEVICE), yb.to(DEVICE)
            pred = model(x)
            loss = (loss_fn(pred, yb) * 1.0) if hw is None else \
                (nn.functional.huber_loss(pred, yb, reduction="none", delta=1.0).mean(0) * hw).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        va_pred = predict(model, va_ds)
        ics = {f"ic_{h}": rank_ic(va_pred[:, k], va_y[:, k], va_idx) for k, h in enumerate(HORIZONS)}
        va_loss = float(np.mean((va_pred - va_y) ** 2))
        rec = {"epoch": ep, "train_loss": tot / max(i + 1, 1), "val_mse": va_loss, **ics, "sec": time.time() - t0}
        hist.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        score = ics[f"ic_{args.select_head}"]
        ck = {"state": model.state_dict(), "hidden": args.hidden, "drop": args.drop, "coins": coins, "epoch": ep,
              "val": rec, "train_split": list(dataset.SPLITS["train"]), "monitor_split": monitor_split,
              "args": vars(args)}
        if refit:
            best = score
            torch.save(ck, MODELS / f"{args.tag}.pt")       # always keep the latest epoch
        elif score > best:
            best, bad = score, 0
            torch.save(ck, MODELS / f"{args.tag}.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True)
                break
    (MODELS / f"{args.tag}.json").write_text(json.dumps(hist, indent=1))
    print(f"{'final' if refit else 'best'} {monitor_split} ic_{args.select_head} {best:.4f}")


if __name__ == "__main__":
    main()
