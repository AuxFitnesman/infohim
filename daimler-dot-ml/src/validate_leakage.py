"""Проверка пересечений scenario_id и (компонент, партия) между train и test."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config
from .data_io import load_test, load_train
from .features import _merge_dedupe_mixtures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR

    tr = load_train(data_dir / config.TRAIN_CSV)
    te = load_test(data_dir / config.TEST_CSV)

    s_tr = set(tr[config.COL_SCENARIO].unique())
    s_te = set(te[config.COL_SCENARIO].unique())
    print("Пересечение scenario_id train∩test:", len(s_tr & s_te))

    tr_m = _merge_dedupe_mixtures(tr)
    te_m = _merge_dedupe_mixtures(te)
    tr_m["_pair"] = list(zip(tr_m[config.COL_COMPONENT], tr_m["_batch_key"]))
    te_m["_pair"] = list(zip(te_m[config.COL_COMPONENT], te_m["_batch_key"]))
    p_tr = set(tr_m["_pair"])
    p_te = set(te_m["_pair"])
    inter = p_tr & p_te
    print("Уникальных пар (компонент, партия) в test, которые есть в train:", len(inter))
    if inter:
        sample = list(inter)[:15]
        print("Примеры:", sample)


if __name__ == "__main__":
    main()
