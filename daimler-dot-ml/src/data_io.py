"""Загрузка CSV смесей и свойств компонентов."""
from __future__ import annotations

import pandas as pd

from . import config


def load_train(path=None) -> pd.DataFrame:
    p = path or (config.DATA_DIR / config.TRAIN_CSV)
    return pd.read_csv(p, encoding="utf-8")


def load_test(path=None) -> pd.DataFrame:
    p = path or (config.DATA_DIR / config.TEST_CSV)
    return pd.read_csv(p, encoding="utf-8")


def load_component_properties(path=None) -> pd.DataFrame:
    p = path or (config.DATA_DIR / config.PROPS_CSV)
    return pd.read_csv(p, encoding="utf-8")
