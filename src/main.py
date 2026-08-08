from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    FEATURE_CANDIDATES,
    FEATURE_LABELS_EN,
    OUTPUT_DIR,
    RAW_PATH,
    REC_DOSE_MAX,
    REC_DOSE_MIN,
    RESULTS_DIR,
)
from src.defocus_bayesian import FeatureProcessor, SigmoidActiveLearner, SigmoidModel
from src.utils.data_processor import clean_data, load_data, save_cleaned
from src.visualization.plots import (
    plot_dose_response,
    plot_feature_importance,
    plot_posterior_check,
    plot_recommended_dose_distribution,
    plot_residual_diagnostics,
    plot_training_fit,
    set_plot_style,
    translate_columns_for_plot,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _find_best_feature_set(df: pd.DataFrame) -> list[str]:
    candidates = [c for c in FEATURE_CANDIDATES if c in df.columns]
    if not candidates:
        return []
    best = candidates[:]
    return best


def _compute_ed85(grid: np.ndarray, mean_pred: np.ndarray, threshold: float = 0.85) -> float:
    y_max = float(np.max(mean_pred))
    y_min = float(np.min(mean_pred))
    target = y_min + threshold * (y_max - y_min)
    idx = np.argmin(np.abs(mean_pred - target))
    return float(grid[idx])


def _apply_high_dose_penalty(dose: float, density_threshold: float = 4.5) -> float:
    if dose > density_threshold:
        excess = dose - density_threshold
        penalty = excess * 0.5
        return max(density_threshold, dose - penalty)
    return dose


def generate_posterior_summary(model: SigmoidModel) -> pd.DataFrame:
    if model.idata_ is None:
        raise RuntimeError("Model not fit.")
    summ = az.summary(model.idata_, kind="all")
    return summ


def generate_convergence_diagnostics(model: SigmoidModel) -> dict:
    conv = model.check_convergence()
    logger.info("Convergence: %s", conv)
    return conv


def compute_fit_metrics(model: SigmoidModel, df_clean: pd.DataFrame, features_z: np.ndarray | None) -> dict:
    doses = df_clean["离焦剂量 (D)"].to_numpy(dtype=float)
    y_obs = df_clean["RDV15(D)"].to_numpy(dtype=float)
    pred = model.predict(doses, features_z)

    residuals = y_obs - pred["mean"]
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs))**2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "r_squared": r_squared,
        "n_observations": len(y_obs),
        "sigma_mean": float(model.idata_.posterior["sigma"].mean().values) if model.idata_ is not None else float("nan"),
    }


def precompute_learning_curves(
    model: SigmoidModel,
    grid: np.ndarray,
    feature_cols: list[str],
    features_z: np.ndarray | None,
    *,
    n_simulations: int = 100,
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    curves = []

    for sim_id in range(n_simulations):
        if features_z is not None and len(feature_cols):
            idx = rng.integers(0, len(features_z))
            feat = features_z[idx : idx + 1].repeat(len(grid), axis=0)
        else:
            feat = None

        pred = model.predict(grid, feat)
        noise = rng.normal(0, pred["std"])
        y_sim = pred["mean"] + noise
        curves.append(y_sim)

    curves = np.array(curves)

    oscillation = np.diff(curves, axis=1)
    sign_changes = np.sum(np.diff(np.sign(oscillation), axis=0) != 0, axis=0)
    convergence_metric = 1.0 / (1.0 + sign_changes.astype(float))

    return pd.DataFrame({
        "dose": grid,
        "mean_response": np.mean(curves, axis=0),
        "std_response": np.std(curves, axis=0),
        "convergence": convergence_metric,
    })


def main() -> int:
    logger.info("=" * 60)
    logger.info("Bayesian Sigmoid Defocus Dose Analysis Pipeline")
    logger.info("=" * 60)

    set_plot_style()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("[1/7] Loading data ...")
    raw_path = RAW_PATH if RAW_PATH.exists() else DATA_DIR / "raw_data.csv"
    if not raw_path.exists():
        logger.error("Raw data not found at %s", raw_path)
        logger.error("Run src/data_generator.py first to generate test data.")
        return 1
    df_raw = load_data(raw_path)
    logger.info("  Raw: %d rows, %d columns", len(df_raw), len(df_raw.columns))

    logger.info("[2/7] Cleaning data ...")
    df_clean = clean_data(df_raw)
    cleaned_path = save_cleaned(df_clean, DATA_DIR)
    logger.info("  Cleaned: %d rows saved to %s", len(df_clean), cleaned_path)

    logger.info("[3/7] Building features ...")
    feature_cols = _find_best_feature_set(df_clean)
    logger.info("  Features: %s", feature_cols)

    features_z = None
    if feature_cols:
        fp = FeatureProcessor(feature_columns=feature_cols)
        features_z = fp.fit_transform(df_clean)
        logger.info("  Feature matrix shape: %s", features_z.shape)

    logger.info("[4/7] Fitting Bayesian Sigmoid Model ...")
    model = SigmoidModel(
        n_chains=2,
        n_cores=1,
        tune=2000,
        draws=1500,
        target_accept=0.90,
        max_treedepth=10,
        use_robust=True,
        feature_importance=bool(feature_cols),
    )

    doses = df_clean["离焦剂量 (D)"].to_numpy(dtype=float)
    responses = df_clean["RDV15(D)"].to_numpy(dtype=float)

    model.fit(doses, responses, features_z, random_seed=42)
    logger.info("  Model fit complete.")

    logger.info("[5/7] Generating diagnostics ...")
    conv = generate_convergence_diagnostics(model)
    fit_metrics = compute_fit_metrics(model, df_clean, features_z)
    logger.info("  Fit metrics: %s", json.dumps(fit_metrics, indent=2))

    posterior_path = RESULTS_DIR / "posterior_summary.csv"
    az.summary(model.idata_).to_csv(posterior_path, encoding="utf-8-sig")
    logger.info("  Posterior summary saved to %s", posterior_path)

    logger.info("[6/7] Computing personalized dose recommendations ...")
    grid = np.linspace(REC_DOSE_MIN - 0.5, REC_DOSE_MAX + 1.0, 500)

    if features_z is not None and len(feature_cols):
        pred_all = model.predict(grid, features_z[:1].repeat(len(grid), axis=0))
        mean_curve = pred_all["mean"]
    else:
        pred_all = model.predict(grid, None)
        mean_curve = pred_all["mean"]

    ed85 = _compute_ed85(grid, mean_curve, threshold=0.85)
    logger.info("  ED85 (population): %.3f D", ed85)

    learning_curves_df = precompute_learning_curves(
        model, grid, feature_cols, features_z, n_simulations=100, random_seed=42,
    )
    learning_curves_df.to_csv(RESULTS_DIR / "learning_curves.csv", index=False, encoding="utf-8-sig")

    rec_doses = []
    unique_patients = df_clean["被测者编号"].unique()
    for pid in unique_patients:
        patient_data = df_clean[df_clean["被测者编号"] == pid]
        eye = patient_data["眼别"].iloc[0] if "眼别" in patient_data.columns else ""

        if features_z is not None and len(feature_cols):
            patient_idx = patient_data.index[0]
            patient_feat = features_z[patient_idx : patient_idx + 1].repeat(len(grid), axis=0)
            pred_patient = model.predict(grid, patient_feat)
        else:
            pred_patient = model.predict(grid, None)

        best_dose = _compute_ed85(grid, pred_patient["mean"], threshold=0.85)
        safe_dose = _apply_high_dose_penalty(best_dose, density_threshold=4.5)

        rec_doses.append({
            "patient_id": pid,
            "eye": eye,
            "recommended_best_dose_D": round(best_dose, 3),
            "recommended_safe_dose_D": round(safe_dose, 3),
            "predicted_response_at_best": round(float(np.max(pred_patient["mean"])), 4),
        })

    rec_df = pd.DataFrame(rec_doses)
    rec_path = RESULTS_DIR / "personalized_recommendations.csv"
    rec_df.to_csv(rec_path, index=False, encoding="utf-8-sig")
    logger.info("  Recommendations saved to %s (%d patients)", rec_path, len(rec_df))

    logger.info("[7/7] Generating figures ...")
    fig_dir = OUTPUT_DIR
    plot_dose_response(df_clean, model, feature_cols, features_z, output_dir=fig_dir)
    plot_posterior_check(model, output_dir=fig_dir)
    plot_feature_importance(model, feature_cols, features_z, output_dir=fig_dir)
    plot_training_fit(df_clean, model, feature_cols, features_z, output_dir=fig_dir)
    plot_residual_diagnostics(df_clean, model, feature_cols, features_z, output_dir=fig_dir)
    plot_recommended_dose_distribution(rec_df, output_dir=fig_dir)
    logger.info("  Figures saved to %s", fig_dir)

    report = {
        "convergence": conv,
        "fit_metrics": fit_metrics,
        "ed85_population": ed85,
        "n_patients": len(rec_df),
        "recommendation_summary": {
            "mean_best_dose": round(float(rec_df["recommended_best_dose_D"].mean()), 3),
            "std_best_dose": round(float(rec_df["recommended_best_dose_D"].std()), 3),
            "range_best_dose": [
                round(float(rec_df["recommended_best_dose_D"].min()), 3),
                round(float(rec_df["recommended_best_dose_D"].max()), 3),
            ],
        },
    }
    report_path = RESULTS_DIR / "analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("  Analysis report saved to %s", report_path)

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())