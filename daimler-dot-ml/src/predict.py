"""Инференс на daimler_mixtures_test.csv -> predictions.csv (3 колонки, без дублей)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from . import config
from .data_io import load_component_properties, load_test
from .features import FeaturePipeline
from .model import DeepSetsMT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR
    art_dir = Path(args.artifacts_dir) if args.artifacts_dir else config.ARTIFACTS_DIR
    out_path = Path(args.output) if args.output else (config.PROJECT_ROOT / "predictions.csv")

    fp = FeaturePipeline.load(art_dir / "feature_pipeline.json")
    p = art_dir / "model.pth"
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(p, map_location="cpu")
    elem_dim = ckpt["elem_dim"]
    cond_dim = ckpt["cond_dim"]
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = DeepSetsMT(elem_dim, cond_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mix = load_test(data_dir / config.TEST_CSV)
    props = load_component_properties(data_dir / config.PROPS_CSV)
    pack = fp.transform_scenarios(mix, props)
    X, mask, cond = pack["X"], pack["mask"], pack["cond"]
    ids = pack["scenario_ids"]

    with torch.no_grad():
        pv, pe = model(
            torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(mask, dtype=torch.float32, device=device),
            torch.tensor(cond, dtype=torch.float32, device=device),
        )
    pred = torch.stack([pv, pe], dim=1).cpu().numpy()
    y_mean = fp.y_mean
    y_std = fp.y_std
    pred = pred * y_std + y_mean

    df = pd.DataFrame(
        {
            config.PRED_COL_ID: ids,
            config.PRED_COL_DELTA: pred[:, 0],
            config.PRED_COL_EOT: pred[:, 1],
        }
    )
    df = df.drop_duplicates(subset=[config.PRED_COL_ID], keep="first")
    df = df.dropna()
    df.to_csv(out_path, index=False, encoding="utf-8")
    print("Wrote", out_path, "rows", len(df))


if __name__ == "__main__":
    main()
