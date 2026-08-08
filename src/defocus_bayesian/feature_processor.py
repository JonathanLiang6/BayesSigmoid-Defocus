from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureProcessor:
    feature_columns: list[str]
    scaler: StandardScaler | None = None

    def fit(self, df: pd.DataFrame) -> "FeatureProcessor":
        self._validate(df)
        x = df[self.feature_columns].to_numpy(dtype=float)
        self.scaler = StandardScaler()
        self.scaler.fit(x)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("FeatureProcessor must be fit() before transform().")
        self._validate(df)
        x = df[self.feature_columns].to_numpy(dtype=float)
        return self.scaler.transform(x)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        if df[self.feature_columns].isna().any().any():
            bad = df[self.feature_columns].isna().sum().sort_values(ascending=False)
            bad = bad[bad > 0]
            raise ValueError(f"NaNs in feature columns:\n{bad}")

