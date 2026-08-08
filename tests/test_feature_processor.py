from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.defocus_bayesian.feature_processor import FeatureProcessor


class TestFeatureProcessor:
    def test_fit_transform(self):
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        fp = FeatureProcessor(["feature1", "feature2"])
        result = fp.fit_transform(df)
        
        assert result.shape == (5, 2)
        assert fp.scaler is not None
        assert np.allclose(result.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_transform_after_fit(self):
        df_train = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        df_test = pd.DataFrame({
            "feature1": [3.0],
            "feature2": [30.0],
        })
        fp = FeatureProcessor(["feature1", "feature2"])
        fp.fit(df_train)
        result = fp.transform(df_test)
        
        assert result.shape == (1, 2)
        assert np.allclose(result, [[0.0, 0.0]], atol=1e-10)

    def test_transform_without_fit(self):
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
        })
        fp = FeatureProcessor(["feature1"])
        
        with pytest.raises(RuntimeError):
            fp.transform(df)

    def test_missing_columns(self):
        df = pd.DataFrame({
            "feature1": [1.0, 2.0],
        })
        fp = FeatureProcessor(["feature1", "feature2"])
        
        with pytest.raises(KeyError):
            fp.fit(df)