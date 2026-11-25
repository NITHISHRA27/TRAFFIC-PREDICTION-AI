"""Deterministic synthetic urban arterial traffic data and preprocessing."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "urban_traffic_5min.csv"
STATES = ["Free Flow", "Slowing", "Congested"]
REGIMES = ["Morning Peak", "Off-Peak", "Evening Peak"]
SPEED_FEATURES = ["speed_kmh", "flow_veh_min", "occupancy_pct", "hour", "day_of_week", "weather_code", "incident_flag"]
OLS_FEATURES = ["speed_kmh", "flow_veh_min", "occupancy_pct", "weather_code", "incident_flag"]
WEATHER_CODE = {"Clear": 0, "Rain": 1, "Fog": 2}


def speed_state(speed: float) -> int:
    return 0 if speed >= 50 else (1 if speed >= 30 else 2)


def _regime(hour: float) -> str:
    if 7 <= hour < 10:
        return "Morning Peak"
    if 17 <= hour < 20:
        return "Evening Peak"
    return "Off-Peak"


def generate_dataset(path: str | Path = DATA_PATH, seed: int = 42) -> pd.DataFrame:
    """Create 60 days of 5-minute data with a Greenshields-derived flow signal."""
    rng = np.random.default_rng(seed)
    n = 60 * 24 * 12
    timestamps = pd.date_range("2025-01-06", periods=n, freq="5min")
    hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60
    dow = timestamps.dayofweek.to_numpy()
    regime = np.array([_regime(h) for h in hour])
    morning = (regime == "Morning Peak").astype(float)
    evening = (regime == "Evening Peak").astype(float)
    weekday = (dow < 5).astype(float)

    # Demand follows daily commute cycles; latent congestion propagates over time.
    demand = 0.20 + 0.42 * morning + 0.48 * evening + 0.12 * np.sin(2 * np.pi * (hour - 6) / 24)
    demand *= np.where(weekday > 0, 1.0, 0.70)
    weather = rng.choice(["Clear", "Rain", "Fog"], size=n, p=[0.78, 0.16, 0.06])
    rain = (weather == "Rain").astype(float)
    fog = (weather == "Fog").astype(float)
    incidents = (rng.random(n) < 0.012).astype(int)
    # Incident episodes last 15–45 minutes, creating credible bottleneck cascades.
    for i in np.flatnonzero(incidents):
        incidents[i:min(n, i + int(rng.integers(3, 10)))] = 1
    congestion = np.zeros(n)
    eps = rng.normal(0, 0.035, n)
    for i in range(n):
        forced = max(0.0, demand[i] - 0.45) * 0.33 + incidents[i] * 0.36 + rain[i] * 0.08 + fog[i] * 0.11
        congestion[i] = np.clip((congestion[i - 1] if i else 0.05) * 0.82 + forced + eps[i], 0, 0.92)
    # Greenshields: v = vf(1-k/kj), q = kv; scale q to vehicles/minute.
    free_speed = 83 - rain * 9 - fog * 15
    density_ratio = np.clip(0.08 + demand * 0.46 + congestion * 0.47, 0.04, 0.96)
    speed = free_speed * (1 - density_ratio) + rng.normal(0, 3.0, n) - incidents * 7
    speed = np.clip(speed, 5, 95)
    flow = np.clip(0.52 * density_ratio * free_speed * (1 - density_ratio) + rng.normal(0, 1.5, n), 5, 60)
    occupancy = np.clip(100 * density_ratio + rng.normal(0, 2.5, n), 0, 100)
    frame = pd.DataFrame({
        "timestamp": timestamps, "speed_kmh": speed, "flow_veh_min": flow,
        "occupancy_pct": occupancy, "hour": hour, "day_of_week": dow,
        "regime": regime, "weather": weather, "incident_flag": incidents,
        "target_speed_5min": np.r_[speed[1:], speed[-1]],
    })
    # Sensor gaps in the observed sensor fields; maintain a valid future target.
    for col in ["speed_kmh", "flow_veh_min", "occupancy_pct"]:
        frame.loc[rng.random(n) < 0.015, col] = np.nan
    destination = Path(path)
    os.makedirs(destination.parent, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


@dataclass
class PreparedData:
    raw: pd.DataFrame
    clean: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler
    imputation_counts: dict[str, int]


def prepare_data(path: str | Path = DATA_PATH) -> PreparedData:
    raw = pd.read_csv(path, parse_dates=["timestamp"])
    n = len(raw)
    a, b = int(n * .70), int(n * .85)
    # Split before interpolation so values on one side of a boundary cannot
    # be used to impute another split.
    parts = [raw.iloc[:a].copy(), raw.iloc[a:b].copy(), raw.iloc[b:].copy()]
    numeric = ["speed_kmh", "flow_veh_min", "occupancy_pct", "target_speed_5min"]
    counts: dict[str, int] = {}
    for part in parts:
        for col in numeric:
            counts[col] = counts.get(col, 0) + int(part[col].isna().sum())
            part[col] = part[col].interpolate(method="linear", limit_direction="both")
        part["speed_kmh"] = part["speed_kmh"].clip(5, 95)
        part["flow_veh_min"] = part["flow_veh_min"].clip(0, 60)
        part["occupancy_pct"] = part["occupancy_pct"].clip(0, 100)
        part["target_speed_5min"] = part["target_speed_5min"].clip(5, 95)
    clean = pd.concat(parts, ignore_index=True)
    clean["weather_code"] = clean["weather"].map(WEATHER_CODE).astype(int)
    train, validation, test = parts
    train["weather_code"] = train["weather"].map(WEATHER_CODE).astype(int)
    validation["weather_code"] = validation["weather"].map(WEATHER_CODE).astype(int)
    test["weather_code"] = test["weather"].map(WEATHER_CODE).astype(int)
    scaler = StandardScaler().fit(train[OLS_FEATURES])
    return PreparedData(raw, clean, train, validation, test, scaler, {k: int(v) for k, v in counts.items() if v})


if __name__ == "__main__":
    data = generate_dataset()
    print(f"Wrote {len(data):,} rows to {DATA_PATH}")
