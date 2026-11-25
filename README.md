# AI-Enhanced Markov Chain Model for Short-Term Traffic Congestion Prediction

A self-contained, software-only undergraduate Applied Statistics & Stochastic Processes project. It synthesizes 60 days of five-minute traffic observations, estimates regime-specific discrete-time Markov chains, trains four five-minute speed regressors, and presents the results in a Streamlit dashboard.

## Requirements

- Python 3.10 or newer
- Packages listed in `requirements.txt`

The pipeline runs offline after package installation. It does not download datasets or model weights.

## Quick start

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python data_pipeline.py
python train_and_evaluate.py
streamlit run app.py
```

The dashboard also creates the CSV and model artifacts automatically on first launch. Generated files are placed beside the source files in `data/`, `models/`, and `outputs/`. Use **Retrain All Models** in the export tab to regenerate the dataset and refit the full pipeline.

The interface uses the project palette in `.streamlit/config.toml`, animated
status and header details, scenario presets, and scroll-triggered Plotly chart
reveals with restrained, matte navy panels and clear system-style typography.
Forecast bars and trajectory lines animate as
their charts enter the viewport; no play button is required. The dashboard
respects reduced-motion preferences and reports results measured on the test
split; the synthetic pipeline does not promise a fixed improvement threshold.

## Project structure

```text
app.py                  Streamlit dashboard and simulator
data_pipeline.py        Deterministic synthetic data and preprocessing
train_and_evaluate.py   Markov matrices, training, metrics, and chart exports
data/urban_traffic_5min.csv
models/artifacts.joblib
outputs/metrics_summary.csv
outputs/*.png
```

## Methods

The synthetic generator uses a fixed seed (`42`), 5-minute timestamps over 60 days (17,280 rows), Greenshields speed-density-flow relationships, commute demand cycles, weather effects, incident cascades, and about 1.5% missing sensor readings. Speed states are Free Flow (≥50 km/h), Slowing (30–<50 km/h), and Congested (<30 km/h). Regime-specific transition matrices are counted only from adjacent training observations and their rows are normalized to one.

Rows are kept chronological: 70% training, 15% validation, and 15% test. Linear interpolation and physical clipping precede splitting; feature scaling for OLS is fit inside its training-only pipeline. Five-fold time-series cross-validation is reported for the OLS baseline. Four models are benchmarked on the untouched final test segment using RMSE, MAE, MAPE, R², peak-hour RMSE, and inference latency: OLS, Random Forest, XGBoost, and Markov-feature XGBoost.

The hybrid receives the empirical next-state probability vector for the current speed state and regime. The generated metrics are measured from the run; no performance percentage is hard-coded. The charts are high-resolution dark-themed PNG files suitable for slides.

## Outputs

- `models/artifacts.joblib`: fitted estimators, transition matrices, steady-state vectors, preprocessing information, split data, metrics, and test predictions.
- `outputs/metrics_summary.csv`: four-model regression comparison.
- `outputs/markov_transition_heatmap.png`
- `outputs/model_rmse_comparison.png`
- `outputs/peak_hour_forecast_timeseries.png`
- `outputs/feature_importance.png`

The dashboard's final tab offers a CSV download and an in-memory ZIP containing the four charts and metrics CSV.
