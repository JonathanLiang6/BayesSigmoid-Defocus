from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw_data.csv"
RESULTS_DIR = DATA_DIR / "results"
OUTPUT_DIR = DATA_DIR / "figures"

REC_DOSE_MIN = 3.5
REC_DOSE_MAX = 5.0

FEATURE_LABELS_EN = {
    "脉络膜血管指数 (CVI)": "CVI",
    "黄斑中心凹下脉络膜厚度 (μm)": "Choroid thickness (um)",
    "年龄 (岁)": "Age (years)",
    "性别": "Sex (0=Male,1=Female)",
    "眼轴长度 (mm)": "Axial length (mm)",
    "验光 - 离焦量 (近视度数，D)": "Refractive error (D)",
    "离焦剂量 (D)": "Defocus dose (D)",
    "RDV15(D)": "RDV15(D)",
}

FEATURE_PRIORITY = {
    "core": {"黄斑中心凹下脉络膜厚度 (μm)"},
    "important": {"脉络膜血管指数 (CVI)", "眼轴长度 (mm)"},
    "auxiliary": {
        "验光 - 离焦量 (近视度数，D)",
        "年龄 (岁)",
        "性别",
    },
}

NUMERIC_COLS = [
    "脉络膜血管指数 (CVI)",
    "黄斑中心凹下脉络膜厚度 (μm)",
    "离焦剂量 (D)",
    "年龄 (岁)",
    "眼轴长度 (mm)",
    "RDV15(D)",
    "验光 - 离焦量 (近视度数，D)",
]

REQUIRED_COLS = [
    "被测者编号",
    "检测批次",
    "眼别",
    "脉络膜血管指数 (CVI)",
    "黄斑中心凹下脉络膜厚度 (μm)",
    "离焦剂量 (D)",
    "年龄 (岁)",
    "性别",
    "眼轴长度 (mm)",
    "RDV15(D)",
    "验光 - 离焦量 (近视度数，D)",
]

FEATURE_CANDIDATES = [
    "脉络膜血管指数 (CVI)",
    "黄斑中心凹下脉络膜厚度 (μm)",
    "年龄 (岁)",
    "性别",
    "眼轴长度 (mm)",
    "验光 - 离焦量 (近视度数，D)",
]