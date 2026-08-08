from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytensor
pytensor.config.compiledir = str(Path(__file__).resolve().parent.parent.parent / ".pytensor_cache")
os.makedirs(pytensor.config.compiledir, exist_ok=True)

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def sigmoid_4p(x: pt.TensorVariable, baseline, max_response, slope, threshold):
    return baseline + (max_response - baseline) / (1.0 + pt.exp(-slope * (x - threshold)))


def _softplus(x: pt.TensorVariable) -> pt.TensorVariable:
    # Numerically stable softplus, avoids version-specific helper APIs.
    return pt.log1p(pt.exp(-pt.abs(x))) + pt.maximum(x, 0)


@dataclass
class SigmoidModel:
    n_chains: int = 2
    n_cores: int = 1
    tune: int = 2000
    draws: int = 1500
    target_accept: float = 0.90
    max_treedepth: int = 10
    use_robust: bool = True
    feature_importance: bool = True

    model_: pm.Model | None = None
    idata_: az.InferenceData | None = None

    def fit(
        self,
        doses: np.ndarray,
        responses: np.ndarray,
        features: np.ndarray | None = None,
        feature_weight_scale: np.ndarray | None = None,
        *,
        random_seed: int = 42,
    ) -> az.InferenceData:
        doses = np.asarray(doses, dtype=float).reshape(-1)
        responses = np.asarray(responses, dtype=float).reshape(-1)
        if doses.shape[0] != responses.shape[0]:
            raise ValueError("doses and responses must have same length.")

        if features is not None:
            features = np.asarray(features, dtype=float)
            if features.shape[0] != doses.shape[0]:
                raise ValueError("features must have same row count as doses.")
            n_features = int(features.shape[1])
            if feature_weight_scale is None:
                feature_weight_scale = np.full(n_features, 0.25, dtype=float)
            else:
                feature_weight_scale = np.asarray(feature_weight_scale, dtype=float).reshape(-1)
                if feature_weight_scale.shape[0] != n_features:
                    raise ValueError("feature_weight_scale length must match number of features.")
        else:
            n_features = 0

        with pm.Model() as model:
            x = pm.Data("dose", doses)
            y = pm.Data("y", responses)

            y_q10, y_q90 = np.quantile(responses, [0.10, 0.90])
            y_std = float(np.std(responses))
            x_mean = float(np.mean(doses))
            x_std = float(np.std(doses))
            x_min = float(np.min(doses))
            x_max = float(np.max(doses))

            # Adjust priors for more realistic human physiology
            # Baseline should be near zero or slightly negative
            baseline_0 = pm.Normal("baseline_0", mu=float(np.mean(responses)), sigma=max(0.05, 0.3 * y_std))
            # Max response should be small positive values based on real data range
            max_resp_0 = pm.Normal("max_response_0", mu=float(np.max(responses)), sigma=max(0.05, 0.3 * y_std))
            # Slope should be smaller for smoother curves (more realistic physiological response)
            slope_0 = pm.Normal("slope_0", mu=max(0.5 / max(x_std, 0.3), 0.3), sigma=0.5)
            # Threshold should be in the middle of the dose range
            thr_0 = pm.TruncatedNormal(
                "threshold_0",
                mu=x_mean,
                sigma=max(0.5 * x_std, 0.3),
                lower=x_min - 0.5,
                upper=x_max + 0.5,
            )

            if features is not None and self.feature_importance:
                f = pm.Data("features", features)
                w_baseline = pm.Normal("w_baseline", mu=0.0, sigma=feature_weight_scale, shape=n_features)
                w_max = pm.Normal("w_max_response", mu=0.0, sigma=feature_weight_scale, shape=n_features)
                w_slope = pm.Normal("w_slope", mu=0.0, sigma=0.7 * feature_weight_scale, shape=n_features)
                w_thr = pm.Normal("w_threshold", mu=0.0, sigma=0.8 * feature_weight_scale, shape=n_features)

                baseline = pm.Deterministic("baseline", baseline_0 + pt.dot(f, w_baseline))
                max_response_raw = pm.Deterministic("max_response_raw", max_resp_0 + pt.dot(f, w_max))
                slope_raw = pm.Deterministic("slope_raw", slope_0 + pt.dot(f, w_slope))
                threshold = pm.Deterministic("threshold", thr_0 + pt.dot(f, w_thr))
            else:
                baseline = baseline_0
                max_response_raw = max_resp_0
                slope_raw = slope_0
                threshold = thr_0

            max_response = pm.Deterministic(
                "max_response",
                baseline + _softplus(max_response_raw - baseline),
            )
            slope = pm.Deterministic("slope", _softplus(slope_raw))

            mu = pm.Deterministic("mu", sigmoid_4p(x, baseline, max_response, slope, threshold))

            sigma = pm.HalfNormal("sigma", sigma=0.05)
            if self.use_robust:
                nu = pm.Exponential("nu", lam=1 / 10) + 1
                pm.StudentT("likelihood", nu=nu, mu=mu, sigma=sigma, observed=y)
            else:
                pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)

            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.n_chains,
                cores=self.n_cores,
                target_accept=self.target_accept,
                random_seed=random_seed,
                nuts={"max_treedepth": self.max_treedepth},
                init="adapt_diag",
                progressbar=True,
            )
            idata.extend(pm.sample_posterior_predictive(idata, progressbar=True))

        self.model_ = model
        self.idata_ = idata
        return idata

    def predict(
        self,
        doses: np.ndarray,
        features: np.ndarray | None = None,
        *,
        hdi_prob: float = 0.94,
    ) -> dict[str, np.ndarray]:
        if self.model_ is None or self.idata_ is None:
            raise RuntimeError("Model is not fit; call fit() first.")

        doses = np.asarray(doses, dtype=float).reshape(-1)
        with self.model_:
            pm.set_data({"dose": doses})
            if "features" in self.model_.named_vars:
                if features is None:
                    raise ValueError("This model was fit with features; features is required for predict().")
                pm.set_data({"features": np.asarray(features, dtype=float)})
            ppc = pm.sample_posterior_predictive(self.idata_, var_names=["mu"], progressbar=False)

        mu = ppc.posterior_predictive["mu"].stack(sample=("chain", "draw")).to_numpy()
        mean = mu.mean(axis=-1)
        std = mu.std(axis=-1)
        alpha = 1.0 - float(hdi_prob)
        hdi_low = np.quantile(mu, alpha / 2.0, axis=-1)
        hdi_high = np.quantile(mu, 1.0 - alpha / 2.0, axis=-1)
        return {"mean": mean, "std": std, "hdi_low": hdi_low, "hdi_high": hdi_high}

    def get_posterior_stats(self) -> dict[str, dict[str, float]]:
        if self.idata_ is None:
            raise RuntimeError("Model is not fit; call fit() first.")
        vars_ = ["baseline_0", "max_response_0", "slope_0", "threshold_0", "sigma"]
        out: dict[str, dict[str, float]] = {}
        summ = az.summary(self.idata_, var_names=[v for v in vars_ if v in self.idata_.posterior], kind="stats")
        for v in summ.index:
            out[str(v)] = {"mean": float(summ.loc[v, "mean"]), "std": float(summ.loc[v, "sd"])}
        return out

    def check_convergence(self) -> dict[str, object]:
        if self.idata_ is None:
            raise RuntimeError("Model is not fit; call fit() first.")
        summ = az.summary(self.idata_, kind="diagnostics")
        rhat = summ["r_hat"].dropna()
        ess = summ["ess_bulk"].dropna()
        return {
            "converged": bool((rhat < 1.01).all()),
            "rhat_max": float(rhat.max()) if len(rhat) else float("nan"),
            "ess_bulk_min": float(ess.min()) if len(ess) else float("nan"),
        }

    def save_posterior(self, path: str | Path) -> None:
        if self.idata_ is None:
            raise RuntimeError("Model is not fit; call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.idata_.to_netcdf(path)

    def load_posterior(self, path: str | Path) -> az.InferenceData:
        path = Path(path)
        self.idata_ = az.from_netcdf(path)
        return self.idata_

