from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import NUMERIC_COLS, REQUIRED_COLS
from src.utils.data_processor import CleanConfig, clean_data


class TestCleanData:
    def test_basic_cleaning(self):
        df = pd.DataFrame({
            "被测者编号": ["S1", "S2"],
            "检测批次": ["第一次", "第二次"],
            "眼别": ["左眼", "右眼"],
            "脉络膜血管指数 (CVI)": ["0.45", "0.50"],
            "黄斑中心凹下脉络膜厚度 (μm)": ["100", "120"],
            "离焦剂量 (D)": ["3.5", "4.0"],
            "年龄 (岁)": ["22", "25"],
            "性别": ["男", "女"],
            "眼轴长度 (mm)": ["24.5", "25.0"],
            "RDV15(D)": ["0.5", "0.6"],
            "验光 - 离焦量 (近视度数，D)": ["-3.0", "-4.0"],
        })
        cfg = CleanConfig()
        result = clean_data(df, cfg)
        
        assert len(result) == 2
        assert result["性别"].dtype == float
        assert result["性别"].tolist() == [0.0, 1.0]
        for col in NUMERIC_COLS:
            if col in df.columns:
                assert np.issubdtype(result[col].dtype, np.number)

    def test_drop_missing(self):
        df = pd.DataFrame({
            "被测者编号": ["S1", "S2", "S3"],
            "检测批次": ["第一次", "第二次", "第一次"],
            "眼别": ["左眼", "右眼", "左眼"],
            "脉络膜血管指数 (CVI)": [0.45, None, 0.50],
            "黄斑中心凹下脉络膜厚度 (μm)": [100, 120, 110],
            "离焦剂量 (D)": [3.5, 4.0, 4.5],
            "年龄 (岁)": [22, 25, 23],
            "性别": ["男", "女", "男"],
            "眼轴长度 (mm)": [24.5, 25.0, 24.8],
            "RDV15(D)": [0.5, 0.6, 0.55],
            "验光 - 离焦量 (近视度数，D)": [-3.0, -4.0, -3.5],
        })
        cfg = CleanConfig(strict_drop_missing=True)
        result = clean_data(df, cfg)
        
        assert len(result) == 2

    def test_gender_mapping(self):
        df = pd.DataFrame({
            "性别": ["男", "女", "男", "女"],
        })
        cfg = CleanConfig(strict_drop_missing=False)
        result = clean_data(df, cfg)
        
        assert result["性别"].tolist() == [0.0, 1.0, 0.0, 1.0]