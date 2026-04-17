"""Пермутационная важность глобальных условий + пример SHAP для табличного пула (опционально)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import config
from .data_io import load_component_properties, load_train
from .features import FeaturePipeline
from .model import DeepSetsMT


def _predict_batch(model, X, mask, cond, y_mean, y_std, device):
    with torch.no_grad():
        pv, pe = model(
            torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(mask, dtype=torch.float32, device=device),
            torch.tensor(cond, dtype=torch.float32, device=device),
        )
    pred = torch.stack([pv, pe], dim=1).cpu().numpy()
    return pred * y_std + y_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    art_dir = Path(args.artifacts_dir) if args.artifacts_dir else config.ARTIFACTS_DIR
    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    fp = FeaturePipeline.load(art_dir / "feature_pipeline.json")
    try:
        ckpt = torch.load(art_dir / "model.pth", map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(art_dir / "model.pth", map_location="cpu")
    model = DeepSetsMT(ckpt["elem_dim"], ckpt["cond_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mix = load_train(data_dir / config.TRAIN_CSV)
    props = load_component_properties(data_dir / config.PROPS_CSV)
    pack = fp.transform_scenarios(mix, props)
    X, mask, cond, y = pack["X"], pack["mask"], pack["cond"], pack["y"]

    base = _predict_batch(model, X, mask, cond, fp.y_mean, fp.y_std, device)
    base_mae = np.mean(np.abs(base - y))

    names = ["temp", "time", "bio"] + [f"cat_{i}" for i in range(cond.shape[1] - 3)]
    drops = []
    for j in range(cond.shape[1]):
        c2 = cond.copy()
        c2[:, j] = np.random.permutation(c2[:, j])
        p2 = _predict_batch(model, X, mask, c2, fp.y_mean, fp.y_std, device)
        drops.append(float(np.mean(np.abs(p2 - y)) - base_mae))

    report = {
        "baseline_mae": base_mae,
        "permutation_delta_mae_cond_columns": dict(zip(names, drops)),
        "note": "Положительное значение — столбец важен для качества на train.",
    }

    try:
        import shap
        from sklearn.linear_model import Ridge

        w = mask[:, :, None]
        pooled = (X * w).sum(axis=1) / (mask.sum(axis=1, keepdims=True).clip(min=1.0))
        tab = np.concatenate([pooled, cond], axis=1)
        colnames = [f"pool_{j}" for j in range(tab.shape[1] - cond.shape[1])] + names
        lin = Ridge(alpha=1.0).fit(tab, y[:, 0])
        expl = shap.LinearExplainer(lin, tab)
        sv = expl.shap_values(tab[: min(200, len(tab))])
        mean_abs = np.abs(sv).mean(axis=0)
        top = sorted(
            zip(colnames, mean_abs.tolist()), key=lambda x: -x[1]
        )[:20]
        report["shap_linear_surrogate_top20_delta_kv"] = {k: float(v) for k, v in top}
    except Exception as e:
        report["shap_note"] = f"skipped: {e}"

    out = art_dir / "interpretation_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
