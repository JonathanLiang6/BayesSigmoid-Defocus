from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

from src.config import FEATURE_LABELS_EN, FEATURE_PRIORITY, REC_DOSE_MIN, REC_DOSE_MAX
from src.defocus_bayesian import SigmoidModel


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    font_names = {f.name for f in font_manager.fontManager.ttflist}
    zh_candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    font_family = next((f for f in zh_candidates if f in font_names), "DejaVu Sans")
    plt.rcParams["font.family"] = font_family
    plt.rcParams["axes.unicode_minus"] = False


def bi_label(en: str, zh: str) -> str:
    return f"{en} / {zh}"


def translate_columns_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={k: v for k, v in FEATURE_LABELS_EN.items() if k in out.columns})
    if "检测批次" in out.columns:
        out["Visit Batch"] = out["检测批次"].astype(str).replace({"第一次": "Visit-1", "第二次": "Visit-2"})
    if "眼别" in out.columns:
        out["Eye"] = out["眼别"].astype(str)
    return out


def plot_dose_response(
    df_clean: pd.DataFrame,
    model: SigmoidModel,
    feature_cols: list[str],
    features_z: np.ndarray | None,
    *,
    output_dir: Path,
    dose_min: float = 0.0,
    dose_max: float = 10.0,
) -> None:
    doses = df_clean["离焦剂量 (D)"].to_numpy(dtype=float)
    y = df_clean["RDV15(D)"].to_numpy(dtype=float)

    grid = np.linspace(dose_min, dose_max, 250)
    grid_feat = None
    if features_z is not None:
        mean_feat = features_z.mean(axis=0, keepdims=True)
        grid_feat = np.repeat(mean_feat, repeats=len(grid), axis=0)

    pred = model.predict(grid, grid_feat)

    plt.figure(figsize=(10, 6))
    plt.scatter(doses, y, s=24, alpha=0.7, label=bi_label("Observed", "观测值"))
    plt.plot(grid, pred["mean"], color="C1", lw=2, label=bi_label("Posterior mean", "后验均值"))
    plt.fill_between(grid, pred["hdi_low"], pred["hdi_high"], color="C1", alpha=0.25, label="94% HDI")
    plt.xlabel(bi_label("Defocus dose (D)", "离焦量 (D)"))
    plt.ylabel(bi_label("RDV15 response", "RDV15反应"))
    plt.title(bi_label("Dose-response model fit", "剂量-反应模型拟合"))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "dose_response_curve.png", dpi=200)
    plt.close()


def plot_posterior_check(model: SigmoidModel, *, output_dir: Path) -> None:
    az.plot_ppc(model.idata_, num_pp_samples=80)
    plt.tight_layout()
    plt.savefig(output_dir / "posterior_predictive_check.png", dpi=200)
    plt.close()


def plot_feature_importance(
    model: SigmoidModel,
    feature_cols: list[str],
    features_z: np.ndarray | None,
    *,
    output_dir: Path,
) -> None:
    if not model.feature_importance or features_z is None or not len(feature_cols):
        return

    post = model.idata_.posterior
    weights = {}
    for name in ["w_baseline", "w_max_response", "w_slope", "w_threshold"]:
        if name in post:
            w = post[name].stack(sample=("chain", "draw")).to_numpy()
            weights[name] = np.mean(np.abs(w), axis=-1)

    if not weights:
        return

    imp = np.zeros(len(feature_cols), dtype=float)
    for v in weights.values():
        imp += v
    imp = imp / len(weights)
    order = np.argsort(imp)[::-1]

    plt.figure(figsize=(10, 5))
    feat_labels = np.array([FEATURE_LABELS_EN.get(c, c) for c in feature_cols])
    sns.barplot(x=imp[order], y=feat_labels[order], orient="h", color="C0")
    plt.xlabel(bi_label("Mean |weight| (posterior)", "后验平均绝对权重"))
    plt.ylabel(bi_label("Feature", "特征"))
    plt.title(bi_label("Clinical feature importance", "临床特征重要性"))
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=200)
    plt.close()


def plot_training_fit(
    df_clean: pd.DataFrame,
    model: SigmoidModel,
    feature_cols: list[str],
    features_z: np.ndarray | None,
    *,
    output_dir: Path,
) -> None:
    doses = df_clean["离焦剂量 (D)"].to_numpy(dtype=float)
    y = df_clean["RDV15(D)"].to_numpy(dtype=float)
    pred_train = model.predict(doses, features_z)

    plt.figure(figsize=(6.5, 6.0))
    plt.scatter(y, pred_train["mean"], alpha=0.75, s=32)
    vmin = min(float(np.min(y)), float(np.min(pred_train["mean"])))
    vmax = max(float(np.max(y)), float(np.max(pred_train["mean"])))
    plt.plot([vmin, vmax], [vmin, vmax], "r--", lw=1)
    plt.xlabel(bi_label("Observed RDV15", "实际RDV15"))
    plt.ylabel(bi_label("Predicted RDV15", "预测RDV15"))
    plt.title(bi_label("Training fit: observed vs predicted", "训练拟合：观测vs预测"))
    plt.tight_layout()
    plt.savefig(output_dir / "training_observed_vs_predicted.png", dpi=220)
    plt.close()


def plot_residual_diagnostics(
    df_clean: pd.DataFrame,
    model: SigmoidModel,
    feature_cols: list[str],
    features_z: np.ndarray | None,
    *,
    output_dir: Path,
) -> None:
    doses = df_clean["离焦剂量 (D)"].to_numpy(dtype=float)
    y = df_clean["RDV15(D)"].to_numpy(dtype=float)
    pred_train = model.predict(doses, features_z)
    resid = y - pred_train["mean"]

    plt.figure(figsize=(10, 4.5))
    plt.scatter(pred_train["mean"], resid, alpha=0.7)
    plt.axhline(0.0, color="black", lw=1, ls="--")
    plt.xlabel(bi_label("Predicted RDV15", "预测RDV15"))
    plt.ylabel(bi_label("Residual", "残差"))
    plt.title(bi_label("Residual diagnostics", "残差诊断"))
    plt.tight_layout()
    plt.savefig(output_dir / "residual_diagnostics.png", dpi=200)
    plt.close()


def plot_recommended_dose_distribution(rec_df: pd.DataFrame, *, output_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    sns.histplot(rec_df["recommended_best_dose_D"], bins=12, kde=True, color="#2E86DE")
    plt.xlabel(bi_label("Recommended best dose (D)", "推荐最佳离焦量 (D)"))
    plt.ylabel(bi_label("Count", "人数"))
    plt.xlim(REC_DOSE_MIN, REC_DOSE_MAX)
    plt.title(bi_label("Recommended doses (clinical comfort range 3.5–5.0D)", "推荐剂量分布（临床舒适区间3.5–5.0D）"))
    plt.tight_layout()
    plt.savefig(output_dir / "recommended_dose_distribution.png", dpi=220)
    plt.close()