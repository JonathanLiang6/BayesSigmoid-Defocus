from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import NUMERIC_COLS, REQUIRED_COLS


@dataclass
class CleanConfig:
    strict_drop_missing: bool = True


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_err: Exception | None = None
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=",", encoding=enc)
            last_err = None
            break
        except Exception as e:
            last_err = e
    if df is None:
        raise RuntimeError(f"Failed to read CSV {path} with encodings {encodings}") from last_err
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df = df.dropna(axis=1, how="all")
    return df


def clean_data(df: pd.DataFrame, cfg: CleanConfig | None = None) -> pd.DataFrame:
    cfg = cfg or CleanConfig()
    out = df.copy()

    out = out.replace("-", pd.NA)
    out = out.replace("—", pd.NA)

    if "性别" in out.columns:
        out["性别"] = out["性别"].map({"男": 0, "女": 1}).astype("float")

    for c in NUMERIC_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    required_cols = [c for c in REQUIRED_COLS if c in out.columns]
    if cfg.strict_drop_missing and required_cols:
        out = out.dropna(subset=required_cols).reset_index(drop=True)

    return out


def save_cleaned(df: pd.DataFrame, results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "cleaned_data.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path