"""Обучение Deep Sets: GroupKFold по scenario_id, затем финальная модель на всех данных."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import config
from .data_io import load_component_properties, load_train
from .features import FeaturePipeline
from .model import DeepSetsMT


def _run_fold(
    model: DeepSetsMT,
    Xt,
    mt,
    ct,
    yt,
    Xv,
    mv,
    cv,
    yv,
    fp: FeaturePipeline,
    device,
    epochs: int,
    batch_size: int,
    lr: float,
    w: torch.Tensor,
    y_mean: torch.Tensor,
    y_std_t: torch.Tensor,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    ds = TensorDataset(
        torch.tensor(Xt, dtype=torch.float32),
        torch.tensor(mt, dtype=torch.float32),
        torch.tensor(ct, dtype=torch.float32),
        torch.tensor(yt, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for _ in range(epochs):
        model.train()
        for xb, mb, cb, yb in dl:
            xb, mb, cb, yb = xb.to(device), mb.to(device), cb.to(device), yb.to(device)
            opt.zero_grad()
            pv, pe = model(xb, mb, cb)
            ybs = (yb - y_mean) / y_std_t
            loss = w[0] * nn.functional.mse_loss(pv, ybs[:, 0]) + w[
                1
            ] * nn.functional.mse_loss(pe, ybs[:, 1])
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pv, pe = model(
            torch.tensor(Xv, dtype=torch.float32, device=device),
            torch.tensor(mv, dtype=torch.float32, device=device),
            torch.tensor(cv, dtype=torch.float32, device=device),
        )
        pred = torch.stack([pv, pe], dim=1) * y_std_t + y_mean
        mae = torch.mean(torch.abs(pred.cpu() - torch.tensor(yv))).item()
    return mae


def _train_full(model, X, mask, cond, y, fp, device, epochs, batch_size, lr, w):
    y_mean = torch.tensor(fp.y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.tensor(fp.y_std, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
        torch.tensor(cond, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    for _ in tqdm(range(epochs), desc="final fit"):
        model.train()
        for xb, mb, cb, yb in dl:
            xb, mb, cb, yb = xb.to(device), mb.to(device), cb.to(device), yb.to(device)
            opt.zero_grad()
            pv, pe = model(xb, mb, cb)
            ybs = (yb - y_mean) / y_std_t
            loss = w[0] * nn.functional.mse_loss(pv, ybs[:, 0]) + w[
                1
            ] * nn.functional.mse_loss(pe, ybs[:, 1])
            loss.backward()
            opt.step()
        sched.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--epochs-cv", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR
    out_dir = Path(args.out_dir) if args.out_dir else config.ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    mix = load_train(data_dir / config.TRAIN_CSV)
    props = load_component_properties(data_dir / config.PROPS_CSV)

    fp = FeaturePipeline().fit(mix, props)
    pack = fp.transform_scenarios(mix, props)
    X, mask, cond, y = pack["X"], pack["mask"], pack["cond"], pack["y"]
    groups = np.array(pack["scenario_ids"])

    elem_dim = X.shape[-1]
    cond_dim = cond.shape[-1]
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    y_mean = torch.tensor(fp.y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.tensor(fp.y_std, dtype=torch.float32, device=device)
    w = torch.tensor([1.0, 0.32], dtype=torch.float32, device=device)

    gkf = GroupKFold(n_splits=args.folds)
    fold_maes = []
    for fold, (tr, va) in enumerate(gkf.split(X, groups=groups)):
        model = DeepSetsMT(elem_dim, cond_dim).to(device)
        mae = _run_fold(
            model,
            X[tr],
            mask[tr],
            cond[tr],
            y[tr],
            X[va],
            mask[va],
            cond[va],
            y[va],
            fp,
            device,
            args.epochs_cv,
            args.batch_size,
            args.lr,
            w,
            y_mean,
            y_std_t,
        )
        fold_maes.append(mae)
        print(f"fold {fold + 1} MAE: {mae:.4f}")

    final = DeepSetsMT(elem_dim, cond_dim).to(device)
    _train_full(
        final,
        X,
        mask,
        cond,
        y,
        fp,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        w,
    )

    fp.save(out_dir / "feature_pipeline.json")
    torch.save(
        {
            "state_dict": final.state_dict(),
            "elem_dim": elem_dim,
            "cond_dim": cond_dim,
        },
        out_dir / "model.pth",
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "groupkfold_mae_mean": float(np.mean(fold_maes)),
                "groupkfold_maes": fold_maes,
                "folds": args.folds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Artifacts ->", out_dir)


if __name__ == "__main__":
    main()
