from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import SigmoidModel


@dataclass
class SigmoidActiveLearner:
    model: SigmoidModel
    dose_min: float = 0.0
    dose_max: float = 10.0
    grid_step: float = 0.05
    stop_uncertainty: float = 0.05
    stop_delta_dose: float = 0.1
    max_rounds: int = 12

    def run(
        self,
        doses: np.ndarray,
        responses: np.ndarray,
        features: np.ndarray | None = None,
        *,
        simulate: bool = True,
        simulate_sigma: float = 0.03,
        random_seed: int = 123,
    ) -> dict[str, object]:
        """
        If simulate=True, we "measure" the recommended dose by drawing from the
        model's predictive mean + Gaussian noise. Replace this with real
        measurement in clinical use.
        """
        rng = np.random.default_rng(random_seed)
        doses = np.asarray(doses, dtype=float).reshape(-1)
        responses = np.asarray(responses, dtype=float).reshape(-1)
        if features is not None:
            features = np.asarray(features, dtype=float)

        history: list[dict[str, float]] = []
        used = set(float(x) for x in doses.tolist())
        prev_reco: float | None = None

        for _round in range(1, self.max_rounds + 1):
            self.model.fit(doses, responses, features, random_seed=random_seed + _round)

            grid = np.arange(self.dose_min, self.dose_max + 1e-9, self.grid_step)
            if len(used):
                mask = np.array([float(g) not in used for g in grid], dtype=bool)
                candidate = grid[mask] if mask.any() else grid
            else:
                candidate = grid

            pred = self.model.predict(candidate, features if features is None else features[:1].repeat(len(candidate), axis=0))
            best_idx = int(np.argmax(pred["mean"]))
            reco = float(candidate[best_idx])
            reco_std = float(pred["std"][best_idx])

            history.append(
                {
                    "round": float(_round),
                    "recommended_dose": reco,
                    "recommended_std": reco_std,
                    "predicted_response": float(pred["mean"][best_idx]),
                }
            )

            if reco_std <= self.stop_uncertainty:
                break
            if prev_reco is not None and abs(prev_reco - reco) < self.stop_delta_dose:
                break

            if not simulate:
                break

            # Simulated "measurement"
            y_new = float(pred["mean"][best_idx] + rng.normal(0.0, simulate_sigma))
            doses = np.concatenate([doses, [reco]])
            responses = np.concatenate([responses, [y_new]])
            if features is not None:
                features = np.vstack([features, features[0]])

            used.add(reco)
            prev_reco = reco

        return {"history": history, "final_doses": doses, "final_responses": responses}

