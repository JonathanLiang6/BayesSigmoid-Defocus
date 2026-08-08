from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    FEATURE_CANDIDATES,
    OUTPUT_DIR,
    REC_DOSE_MAX,
    REC_DOSE_MIN,
    RESULTS_DIR,
)
from src.defocus_bayesian import FeatureProcessor, SigmoidActiveLearner, SigmoidModel
from src.utils.data_processor import clean_data, load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_learning_session(
    patient_data: pd.DataFrame,
    model: SigmoidModel,
    feature_cols: list[str],
    features_z: np.ndarray | None,
    *,
    max_rounds: int = 12,
    grid_step: float = 0.05,
    random_seed: int = 42,
) -> dict:
    doses = patient_data["离焦剂量 (D)"].to_numpy(dtype=float)
    responses = patient_data["RDV15(D)"].to_numpy(dtype=float)

    learner = SigmoidActiveLearner(
        model=SigmoidModel(
            n_chains=2,
            n_cores=1,
            tune=2000,
            draws=1500,
            target_accept=0.90,
            max_treedepth=10,
            use_robust=True,
            feature_importance=bool(feature_cols),
        ),
        dose_min=REC_DOSE_MIN - 0.5,
        dose_max=REC_DOSE_MAX + 1.0,
        grid_step=grid_step,
        stop_uncertainty=0.05,
        stop_delta_dose=0.1,
        max_rounds=max_rounds,
    )

    patient_feat = None
    if features_z is not None and len(feature_cols):
        idx = patient_data.index[0]
        patient_feat = features_z[idx : idx + 1]

    result = learner.run(
        doses,
        responses,
        patient_feat,
        simulate=True,
        simulate_sigma=0.03,
        random_seed=random_seed,
    )

    history = result["history"]
    final_doses = result["final_doses"]

    recommended_dose = history[-1]["recommended_dose"] if history else float("nan")
    predicted_response = history[-1]["predicted_response"] if history else float("nan")
    uncertainty = history[-1]["recommended_std"] if history else float("nan")

    return {
        "n_rounds": len(history),
        "recommended_dose": recommended_dose,
        "predicted_response": predicted_response,
        "uncertainty": uncertainty,
        "converged": uncertainty <= 0.05 if not np.isnan(uncertainty) else False,
        "history": history,
        "total_doses_used": len(final_doses),
    }


def batch_generate_learning_sessions(
    df_clean: pd.DataFrame,
    model: SigmoidModel,
    feature_cols: list[str],
    features_z: np.ndarray | None,
    *,
    output_dir: Path,
    max_rounds: int = 12,
    grid_step: float = 0.05,
) -> pd.DataFrame:
    unique_patients = df_clean["被测者编号"].unique()
    logger.info("Processing %d patients ...", len(unique_patients))

    sessions = []
    total = len(unique_patients)

    for i, pid in enumerate(unique_patients, 1):
        patient_data = df_clean[df_clean["被测者编号"] == pid]
        eye = patient_data["眼别"].iloc[0] if "眼别" in patient_data.columns else ""

        try:
            session = generate_learning_session(
                patient_data,
                model,
                feature_cols,
                features_z,
                max_rounds=max_rounds,
                grid_step=grid_step,
                random_seed=42 + i * 7,
            )
        except Exception as e:
            logger.warning("Patient %s failed: %s", pid, e)
            session = {
                "n_rounds": 0,
                "recommended_dose": float("nan"),
                "predicted_response": float("nan"),
                "uncertainty": float("nan"),
                "converged": False,
                "history": [],
                "total_doses_used": 0,
            }

        sessions.append({
            "patient_id": pid,
            "eye": eye,
            "n_rounds": session["n_rounds"],
            "recommended_dose": session["recommended_dose"],
            "predicted_response": session["predicted_response"],
            "uncertainty": session["uncertainty"],
            "converged": session["converged"],
            "total_doses_used": session["total_doses_used"],
        })

        if i % 10 == 0 or i == total:
            logger.info("  [%d/%d] Done", i, total)

    sessions_df = pd.DataFrame(sessions)
    sessions_path = output_dir / "learning_sessions.csv"
    sessions_df.to_csv(sessions_path, index=False, encoding="utf-8-sig")
    logger.info("Learning sessions saved to %s", sessions_path)

    sessions_detail_path = output_dir / "learning_sessions_detail.json"
    with open(sessions_detail_path, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Learning sessions detail saved to %s", sessions_detail_path)

    return sessions_df


def main() -> int:
    logger.info("=" * 60)
    logger.info("Batch Learning Session Generator")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / "raw_data.csv"
    if not raw_path.exists():
        logger.error("Raw data not found at %s", raw_path)
        logger.error("Run src/data_generator.py first.")
        return 1

    df_raw = load_data(raw_path)
    df_clean = clean_data(df_raw)
    logger.info("Loaded %d records after cleaning.", len(df_clean))

    feature_cols = [c for c in FEATURE_CANDIDATES if c in df_clean.columns]
    logger.info("Features: %s", feature_cols)

    features_z = None
    if feature_cols:
        fp = FeatureProcessor(feature_columns=feature_cols)
        features_z = fp.fit_transform(df_clean)

    logger.info("Fitting base model on full dataset ...")
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
    logger.info("Base model fit complete.")

    sessions_df = batch_generate_learning_sessions(
        df_clean,
        model,
        feature_cols,
        features_z,
        output_dir=RESULTS_DIR,
        max_rounds=12,
        grid_step=0.05,
    )

    converged = sessions_df["converged"].sum()
    total = len(sessions_df)
    logger.info("Convergence rate: %.1f%% (%d/%d)", 100.0 * converged / total, converged, total)
    logger.info("Mean rounds: %.2f", sessions_df["n_rounds"].mean())

    logger.info("=" * 60)
    logger.info("Learning session generation complete.")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
