"""Подстановка свойств (партия > typical), свёртка рецепта, подготовка тензоров для Deep Sets."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config

_PROP_COL = "Наименование показателя"
_VAL_COL = "Значение показателя"
_UNIT_COL = "Единица измерения_по_партиям"


def _norm_batch(s: Any) -> str:
    if pd.isna(s):
        return ""
    t = str(s).strip().lower()
    if t in ("", "nan", "б/н", "без номера", "—", "-", "none", "null"):
        return ""
    return t


def _to_float(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("\xa0", " ").replace(" ", "").replace(",", ".")
    if s in ("", "нет", "nan", "none"):
        return np.nan
    m = re.match(r"^([<>]=?)\s*([0-9.eE+-]+)$", s)
    if m:
        return float(m.group(2))
    try:
        return float(s)
    except ValueError:
        return np.nan


def _pivot_numeric_properties(props: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Wide-таблица: индекс (Компонент, партия_норм), столбцы — показатели (только числовые)."""
    df = props.copy()
    df["_batch_n"] = df[config.COL_BATCH].apply(
        lambda x: config.TYPICAL_TOKEN
        if str(x).strip().lower() == config.TYPICAL_TOKEN
        else _norm_batch(x)
    )
    df["_v"] = df[_VAL_COL].map(_to_float)
    df = df[df["_v"].notna() & np.isfinite(df["_v"])]
    pivot = df.pivot_table(
        index=[config.COL_COMPONENT, "_batch_n"],
        columns=_PROP_COL,
        values="_v",
        aggfunc="mean",
    )
    pivot = pivot.reset_index().rename(columns={"_batch_n": "_batch_key"})
    ind_cols = [config.COL_COMPONENT, "_batch_key"]
    feat_cols = [c for c in pivot.columns if c not in ind_cols]
    return pivot.set_index(ind_cols), feat_cols


def _merge_dedupe_mixtures(mix: pd.DataFrame) -> pd.DataFrame:
    """Одинаковые (сценарий, компонент, партия) → суммируем массовую долю."""
    m = mix.copy()
    m["_bk"] = m[config.COL_BATCH].map(_norm_batch)
    gcols = [config.COL_SCENARIO, config.COL_COMPONENT, "_bk"]
    agg = {
        config.COL_MASS: "sum",
        config.COL_TEMP: "first",
        config.COL_TIME: "first",
        config.COL_BIO: "first",
        config.COL_CAT: "first",
    }
    if config.COL_DELTA in m.columns:
        agg[config.COL_DELTA] = "first"
    if config.COL_EOT in m.columns:
        agg[config.COL_EOT] = "first"
    return m.groupby(gcols, as_index=False).agg(agg).rename(columns={"_bk": "_batch_key"})


def _lookup_row(
    comp: str,
    batch_key: str,
    wide: pd.DataFrame,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Вектор признаков компонента и маска «значение измерено» (1) / fallback (0)."""
    vec = np.zeros(len(feat_cols), dtype=np.float64)
    known = np.zeros(len(feat_cols), dtype=np.float64)
    key_batch = (comp, batch_key) if batch_key else None
    if key_batch and key_batch in wide.index:
        row = wide.loc[key_batch, feat_cols].values.astype(np.float64)
        vec[:] = np.nan_to_num(row, nan=0.0)
        known[:] = np.isfinite(row).astype(np.float64)
        return vec, known
    key_typ = (comp, config.TYPICAL_TOKEN)
    if key_typ in wide.index:
        row = wide.loc[key_typ, feat_cols].values.astype(np.float64)
        vec[:] = np.nan_to_num(row, nan=0.0)
        known[:] = 0.5
        return vec, known
    return vec, known


class FeaturePipeline:
    """Обучается на train: медианы импутации, список признаков, max set size, скейлеры."""

    def __init__(self, max_props: int = 48, pair_dim: int = 8):
        self.max_props = max_props
        self.pair_dim = pair_dim
        self.feat_cols: list[str] = []
        self.max_n: int = 0
        self.medians: np.ndarray | None = None
        self.scaler_cond: StandardScaler | None = None
        self.scaler_mass: StandardScaler | None = None
        self.y_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.cat_levels: list[Any] = []

    def fit(self, mix_train: pd.DataFrame, props: pd.DataFrame) -> FeaturePipeline:
        wide, feat_cols = _pivot_numeric_properties(props)
        non_typ = wide.reset_index()["_batch_key"] != config.TYPICAL_TOKEN
        var = wide.reset_index()[non_typ][feat_cols].var(axis=0, skipna=True)
        var = var.fillna(0).sort_values(ascending=False)
        self.feat_cols = list(var.index[: self.max_props])
        for c in self.feat_cols:
            if c not in wide.columns:
                wide[c] = np.nan
        self.medians = wide[self.feat_cols].median(axis=0).values.astype(np.float64)

        m = _merge_dedupe_mixtures(mix_train)
        self.max_n = int(
            m.groupby(config.COL_SCENARIO).size().max()
        )

        cats = sorted(m[config.COL_CAT].dropna().unique().tolist())
        self.cat_levels = cats

        cond = m[[config.COL_TEMP, config.COL_TIME, config.COL_BIO]].values.astype(
            np.float64
        )
        self.scaler_cond = StandardScaler().fit(cond)
        masses = m[[config.COL_MASS]].values.astype(np.float64)
        self.scaler_mass = StandardScaler().fit(masses)

        y = m[[config.COL_DELTA, config.COL_EOT]].values.astype(np.float64)
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0) + 1e-6
        return self

    def transform_scenarios(
        self,
        mix: pd.DataFrame,
        props: pd.DataFrame,
        *,
        fit_y: bool = False,
    ) -> dict[str, Any]:
        wide, _ = _pivot_numeric_properties(props)
        for j, c in enumerate(self.feat_cols):
            if c not in wide.columns:
                wide[c] = np.nan
        wide = wide[self.feat_cols]
        med = self.medians
        m = _merge_dedupe_mixtures(mix)

        scenarios = m[config.COL_SCENARIO].unique()
        B = len(scenarios)
        D = 1 + len(self.feat_cols) + len(self.feat_cols) + self.pair_dim
        X = np.zeros((B, self.max_n, D), dtype=np.float64)
        mask = np.zeros((B, self.max_n), dtype=np.float64)
        cond = np.zeros((B, 3 + len(self.cat_levels)), dtype=np.float64)
        y = np.zeros((B, 2), dtype=np.float64)
        has_y = config.COL_DELTA in m.columns
        scen_to_idx = {s: i for i, s in enumerate(scenarios)}

        for scen in scenarios:
            sub = m[m[config.COL_SCENARIO] == scen].reset_index(drop=True)
            bi = scen_to_idx[scen]
            n = len(sub)
            mask[bi, :n] = 1.0

            cvec = sub[[config.COL_TEMP, config.COL_TIME, config.COL_BIO]].values.astype(
                np.float64
            )
            cond[bi, :3] = self.scaler_cond.transform(cvec[0:1])[0]
            catv = sub[config.COL_CAT].iloc[0]
            if catv in self.cat_levels:
                cond[bi, 3 + self.cat_levels.index(catv)] = 1.0

            masses = sub[[config.COL_MASS]].values.astype(np.float64)
            ms = self.scaler_mass.transform(masses).ravel()
            props_m = np.zeros((n, len(self.feat_cols)))
            known_m = np.zeros((n, len(self.feat_cols)))
            for k in range(n):
                comp = sub.iloc[k][config.COL_COMPONENT]
                bk = sub.iloc[k]["_batch_key"]
                v, kn = _lookup_row(comp, bk, wide, self.feat_cols)
                props_m[k] = np.where(np.isfinite(v), v, med)
                known_m[k] = kn

            for k in range(n):
                X[bi, k, 0] = ms[k]
                X[bi, k, 1 : 1 + len(self.feat_cols)] = props_m[k]
                X[bi, k, 1 + len(self.feat_cols) : 1 + 2 * len(self.feat_cols)] = (
                    props_m[k] * ms[k]
                )
                off = 1 + 2 * len(self.feat_cols)
                if self.pair_dim > 0 and n > 1:
                    acc = []
                    for t in range(n):
                        if t == k:
                            continue
                        acc.append(ms[k] * ms[t])
                    if acc:
                        acc = sorted(acc, reverse=True)[: self.pair_dim]
                        X[bi, k, off : off + len(acc)] = acc

            if has_y:
                y[bi] = sub[[config.COL_DELTA, config.COL_EOT]].iloc[0].values.astype(
                    np.float64
                )

        return {
            "X": X,
            "mask": mask,
            "cond": cond,
            "y": y if has_y else None,
            "scenario_ids": list(scenarios),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "feat_cols": self.feat_cols,
            "max_n": self.max_n,
            "medians": self.medians.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_std": self.y_std.tolist(),
            "cat_levels": self.cat_levels,
            "pair_dim": self.pair_dim,
            "scaler_cond_mean": self.scaler_cond.mean_.tolist(),
            "scaler_cond_scale": self.scaler_cond.scale_.tolist(),
            "scaler_mass_mean": self.scaler_mass.mean_.tolist(),
            "scaler_mass_scale": self.scaler_mass.scale_.tolist(),
        }
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FeaturePipeline:
        meta = json.loads(path.read_text(encoding="utf-8"))
        fp = cls(max_props=len(meta["feat_cols"]), pair_dim=meta.get("pair_dim", 8))
        fp.feat_cols = meta["feat_cols"]
        fp.max_n = meta["max_n"]
        fp.medians = np.array(meta["medians"], dtype=np.float64)
        fp.y_mean = np.array(meta["y_mean"], dtype=np.float64)
        fp.y_std = np.array(meta["y_std"], dtype=np.float64)
        fp.cat_levels = meta["cat_levels"]
        fp.scaler_cond = StandardScaler()
        fp.scaler_cond.mean_ = np.array(meta["scaler_cond_mean"])
        fp.scaler_cond.scale_ = np.array(meta["scaler_cond_scale"])
        fp.scaler_cond.var_ = fp.scaler_cond.scale_**2
        fp.scaler_cond.n_features_in_ = 3
        fp.scaler_mass = StandardScaler()
        fp.scaler_mass.mean_ = np.array(meta["scaler_mass_mean"])
        fp.scaler_mass.scale_ = np.array(meta["scaler_mass_scale"])
        fp.scaler_mass.var_ = fp.scaler_mass.scale_**2
        fp.scaler_mass.n_features_in_ = 1
        return fp
