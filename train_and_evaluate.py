"""Train the four requested regressors and export reproducible reports."""
from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data_pipeline import DATA_PATH, OLS_FEATURES, REGIMES, SPEED_FEATURES, generate_dataset, prepare_data, speed_state

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
ARTIFACT_PATH = MODEL_DIR / "artifacts.joblib"
PALETTE = {"navy": "#0D1433", "panel": "#1B2B41", "blue": "#6C90C3", "white": "#F1F5F9", "grey": "#CBD5E1", "green": "#569578", "amber": "#D4A24C", "red": "#B82938"}


def transition_matrices(train: pd.DataFrame) -> dict[str, np.ndarray]:
    matrices = {}
    for regime in REGIMES:
        subset = train.loc[train.regime.eq(regime)]
        state = subset.speed_kmh.map(speed_state).to_numpy(dtype=int)
        counts = np.zeros((3, 3), dtype=float)
        # Count only consecutive five-minute observations in this same regime.
        positions = subset.index.to_numpy()
        for i in range(len(state) - 1):
            if positions[i + 1] == positions[i] + 1:
                counts[state[i], state[i + 1]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        matrix = np.divide(counts, row_sums, out=np.zeros((3, 3)), where=row_sums > 0)
        for state in range(3):
            if row_sums[state, 0] == 0:
                matrix[state, state] = 1.0
        matrices[regime] = matrix
    return matrices


def add_markov_features(frame: pd.DataFrame, matrices: dict[str, np.ndarray]) -> pd.DataFrame:
    out = frame.copy()
    states = out.speed_kmh.map(speed_state).to_numpy(dtype=int)
    probs = np.vstack([matrices[r][s] for r, s in zip(out.regime, states)])
    out[["P_Free", "P_Slow", "P_Cong"]] = probs
    return out


def steady_state(matrix: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(matrix.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    if v.sum() < 0:
        v = -v
    v = np.maximum(v, 0)
    return v / v.sum() if v.sum() else np.full(3, 1 / 3)


def _features(frame: pd.DataFrame, hybrid: bool = False) -> pd.DataFrame:
    cols = SPEED_FEATURES + (["P_Free", "P_Slow", "P_Cong"] if hybrid else [])
    return frame[cols]


def _ols_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[OLS_FEATURES]


def _make_models(seed: int = 42) -> dict[str, object]:
    return {
        "OLS Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=180, max_depth=14, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=240, max_depth=5, learning_rate=.045, subsample=.9, colsample_bytree=.9, reg_lambda=2, objective="reg:squarederror", n_jobs=-1, random_state=seed),
        "Hybrid Markov + XGBoost": XGBRegressor(n_estimators=280, max_depth=5, learning_rate=.04, subsample=.9, colsample_bytree=.9, reg_lambda=2, objective="reg:squarederror", n_jobs=-1, random_state=seed),
    }


def _metric_row(name: str, model: object, x: pd.DataFrame, y: pd.Series, regime: pd.Series) -> dict[str, float | str]:
    start = time.perf_counter()
    pred = model.predict(x)
    latency = (time.perf_counter() - start) * 1000 / max(len(x), 1)
    peak = regime.isin(["Morning Peak", "Evening Peak"])
    return {"Model": name, "RMSE (km/h)": root_mean_squared_error(y, pred), "MAE (km/h)": mean_absolute_error(y, pred), "MAPE (%)": mean_absolute_percentage_error(y, pred) * 100, "R2": r2_score(y, pred), "Peak-Hour RMSE (km/h)": root_mean_squared_error(y[peak], pred[peak]), "Inference Latency (ms/sample)": latency}


def _save_charts(matrices: dict[str, np.ndarray], metrics: pd.DataFrame, test: pd.DataFrame, preds: dict[str, np.ndarray], hybrid_model: object) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": PALETTE["navy"], "axes.facecolor": PALETTE["panel"], "text.color": PALETTE["white"], "axes.labelcolor": PALETTE["grey"], "xtick.color": PALETTE["grey"], "ytick.color": PALETTE["grey"]})
    research_cmap = LinearSegmentedColormap.from_list("research_navy", [PALETTE["navy"], "#274272", PALETTE["blue"]])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, regime in zip(axes, REGIMES):
        ax.imshow(matrices[regime], cmap=research_cmap, vmin=0, vmax=1)
        ax.set_title(regime, color=PALETTE["white"])
        ax.set_xticks(range(3), ["Free", "Slow", "Cong"]); ax.set_yticks(range(3), ["Free", "Slow", "Cong"])
        for (i, j), v in np.ndenumerate(matrices[regime]): ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "markov_transition_heatmap.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(metrics)); w = .36
    ax.bar(x-w/2, metrics["RMSE (km/h)"], w, label="Overall RMSE", color=PALETTE["blue"])
    ax.bar(x+w/2, metrics["Peak-Hour RMSE (km/h)"], w, label="Peak-hour RMSE", color=PALETTE["amber"])
    ax.set_xticks(x, metrics["Model"], rotation=18, ha="right"); ax.set_ylabel("Speed error (km/h)"); ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_rmse_comparison.png", dpi=300); plt.close(fig)
    peak = test.regime.isin(["Morning Peak", "Evening Peak"])
    ix = np.flatnonzero(peak)[:240]
    fig, ax = plt.subplots(figsize=(12, 5)); ax.plot(test.iloc[ix].timestamp, test.iloc[ix].target_speed_5min, label="Actual", color=PALETTE["white"], lw=1.7)
    ax.plot(test.iloc[ix].timestamp, preds["Hybrid Markov + XGBoost"][ix], label="Hybrid", color=PALETTE["green"], lw=1.4)
    ax.plot(test.iloc[ix].timestamp, preds["XGBoost"][ix], label="XGBoost", color=PALETTE["amber"], lw=1.2); ax.set_ylabel("Speed (km/h)"); ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "peak_hour_forecast_timeseries.png", dpi=300); plt.close(fig)
    importance = hybrid_model.feature_importances_
    names = SPEED_FEATURES + ["P_Free", "P_Slow", "P_Cong"]
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(np.array(names)[order], importance[order], color=PALETTE["blue"]); ax.set_xlabel("Gain importance"); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300); plt.close(fig)


def train_and_evaluate() -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not DATA_PATH.exists(): generate_dataset()
    prepared = prepare_data()
    matrices = transition_matrices(prepared.train)
    train, val, test = (add_markov_features(x, matrices) for x in (prepared.train, prepared.validation, prepared.test))
    models = _make_models()
    Xtr = {"OLS Linear Regression": _ols_features(train), "Random Forest": _features(train), "XGBoost": _features(train), "Hybrid Markov + XGBoost": _features(train, True)}
    ytr = train.target_speed_5min
    # Fit only on train. The chronological validation split is retained for transparent reporting.
    for name, model in models.items(): model.fit(Xtr[name], ytr)
    y = test.target_speed_5min
    xte = {"OLS Linear Regression": _ols_features(test), "Random Forest": _features(test), "XGBoost": _features(test), "Hybrid Markov + XGBoost": _features(test, True)}
    metrics = pd.DataFrame([_metric_row(name, model, xte[name], y, test.regime) for name, model in models.items()])
    predictions = {name: model.predict(xte[name]) for name, model in models.items()}
    # Training-only 5-fold chronological CV score for the OLS baseline.
    cv = cross_val_score(models["OLS Linear Regression"], Xtr["OLS Linear Regression"], ytr, cv=TimeSeriesSplit(n_splits=5), scoring="neg_root_mean_squared_error")
    metrics.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    _save_charts(matrices, metrics, test, predictions, models["Hybrid Markov + XGBoost"])
    ols = models["OLS Linear Regression"][-1]
    artifacts = {"models": models, "markov_matrices": matrices, "steady_states": {k: steady_state(v) for k, v in matrices.items()}, "scaler": prepared.scaler, "metrics": metrics, "cv_rmse": -cv, "ols_intercept": float(ols.intercept_), "ols_coefficients": dict(zip([f"z_{name}" for name in OLS_FEATURES], map(float, ols.coef_))), "clean": prepared.clean, "raw": prepared.raw, "imputation_counts": prepared.imputation_counts, "split_sizes": {"train": len(prepared.train), "validation": len(prepared.validation), "test": len(prepared.test)}, "test": test, "test_predictions": predictions}
    joblib.dump(artifacts, ARTIFACT_PATH)
    return artifacts


if __name__ == "__main__":
    result = train_and_evaluate()
    print(result["metrics"].to_string(index=False))
    print(f"Artifacts saved to {ARTIFACT_PATH}")
