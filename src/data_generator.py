from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR

logger = logging.getLogger(__name__)


def generate_test_data(n_samples: int = 50, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    subjects = np.repeat(np.arange(1, n_samples // 2 + 1), 2)
    eyes = np.tile(["OD", "OS"], n_samples // 2)
    batches = rng.choice(["第一次", "第二次"], size=n_samples)
    doses = rng.choice([3.5, 5.0], size=n_samples)
    
    ages = rng.integers(18, 35, size=n_samples)
    genders = rng.choice(["男", "女"], size=n_samples)
    
    axial_lengths = 24.0 + rng.normal(0, 1.2, size=n_samples)
    axial_lengths = np.clip(axial_lengths, 22.0, 28.0)
    
    cvi_values = 0.45 + rng.normal(0, 0.08, size=n_samples)
    cvi_values = np.clip(cvi_values, 0.30, 0.70)
    
    choroid_thickness = 10.0 + rng.normal(0, 15.0, size=n_samples)
    choroid_thickness = np.clip(choroid_thickness, 0.0, 60.0)
    
    refractive_errors = -2.0 - rng.exponential(2.0, size=n_samples)
    refractive_errors = np.clip(refractive_errors, -8.0, -1.0)
    
    baseline = -0.05 + 0.02 * (axial_lengths - 24) + 0.1 * (cvi_values - 0.45)
    max_response = 0.08 + 0.05 * (choroid_thickness / 20)
    slope = 1.5 + 0.5 * (cvi_values - 0.45)
    threshold = 4.2 + 0.3 * (axial_lengths - 24) / 2
    
    rdv15 = baseline + (max_response - baseline) / (1 + np.exp(-slope * (doses - threshold)))
    rdv15 += rng.normal(0, 0.02, size=n_samples)
    rdv15 = np.clip(rdv15, -0.2, 0.3)

    df = pd.DataFrame({
        "被测者编号": subjects,
        "检测批次": batches,
        "眼别": eyes,
        "脉络膜血管指数 (CVI)": np.round(cvi_values, 4),
        "黄斑中心凹下脉络膜厚度 (μm)": np.round(choroid_thickness, 4),
        "离焦剂量 (D)": doses,
        "年龄 (岁)": ages,
        "性别": genders,
        "眼轴长度 (mm)": np.round(axial_lengths, 2),
        "RDV15(D)": np.round(rdv15, 4),
        "验光 - 离焦量 (近视度数，D)": np.round(refractive_errors, 2),
    })

    return df


def save_test_data(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = DATA_DIR / "raw_data.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Generated test data saved to: {output_path}")
    return output_path


def main():
    logging.basicConfig(level=logging.INFO)
    df = generate_test_data(n_samples=50, random_seed=42)
    save_test_data(df)
    print(f"Generated {len(df)} samples")
    print("Data summary:")
    print(df.describe())


if __name__ == "__main__":
    main()