"""Streamlit research dashboard for Markov + ML traffic forecasting."""
from __future__ import annotations

import io
import zipfile

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_pipeline import (
    DATA_PATH,
    OLS_FEATURES,
    REGIMES,
    STATES,
    WEATHER_CODE,
    generate_dataset,
    speed_state,
)
from train_and_evaluate import ARTIFACT_PATH, OUTPUT_DIR, train_and_evaluate

st.set_page_config(
    page_title="Traffic Markov Lab",
    page_icon=":material/traffic:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

NAVY = "#0D1433"
PANEL = "#1B2B41"
INDIGO = "#171F55"
BORDER = "#274272"
BLUE = "#6C90C3"
WHITE = "#F1F5F9"
GREY = "#CBD5E1"
GREEN = "#569578"
AMBER = "#D4A24C"
RED = "#B82938"
STATE_STYLE = [
    ("#022B22", "#275D46", GREEN),
    ("#2B2110", "#8C6218", AMBER),
    ("#19171B", "#51080D", RED),
]
CHART_NAMES = (
    "markov_transition_heatmap.png",
    "model_rmse_comparison.png",
    "peak_hour_forecast_timeseries.png",
    "feature_importance.png",
)
METRICS_PATH = OUTPUT_DIR / "metrics_summary.csv"
MODEL_NAMES = [
    "OLS Linear Regression",
    "Random Forest",
    "XGBoost",
    "Hybrid Markov + XGBoost",
]

st.html(
    """
    <style>
    @keyframes rise-in {
      from { opacity: 0; transform: translateY(14px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes signal-pulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(86, 149, 120, .50); }
      55% { box-shadow: 0 0 0 7px rgba(86, 149, 120, 0); }
    }
    @keyframes scroll-reveal {
      from { opacity: 0; transform: translateY(18px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes graph-draw {
      to { stroke-dashoffset: 0; }
    }
    html, body {
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
      -webkit-font-smoothing: antialiased;
    }
    .lab-hero {
      position: relative; overflow: hidden;
      padding: clamp(26px, 3.2vw, 42px); margin: 8px 0 22px;
      border: 1px solid rgba(108,144,195,.24); border-radius: 24px;
      background: linear-gradient(135deg, #171F55 0%, #1B2B41 58%, #111a38 100%);
      box-shadow: 0 18px 48px rgba(0,0,0,.18);
      animation: rise-in .55s cubic-bezier(.2,.7,.2,1) both;
    }
    .lab-hero:after {
      content: ""; position: absolute; inset: auto 0 0;
      height: 1px; opacity: .8;
      background: linear-gradient(90deg, transparent, rgba(108,144,195,.55), transparent);
    }
    .lab-kicker {
      color: #9BB6D8; font: 650 11px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: .10em; text-transform: uppercase; margin-bottom: 12px;
    }
    .lab-title {
      color: #F1F5F9; font-size: clamp(30px, 4vw, 48px); font-weight: 650;
      line-height: 1.14; letter-spacing: -.028em; max-width: 1040px; margin: 0;
      overflow-wrap: normal; word-break: normal; text-wrap: balance;
    }
    .lab-subtitle {
      color: #CBD5E1; font-size: 15px; line-height: 1.55;
      margin: 14px 0 0; max-width: 760px;
    }
    .lab-live {
      display: inline-flex; align-items: center; gap: 9px; color: #CBD5E1;
      font: 600 12px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: .02em; margin-top: 20px;
    }
    .lab-live-dot {
      width: 8px; height: 8px; border-radius: 50%; background: #569578;
      animation: signal-pulse 1.8s ease-out infinite;
    }
    .state-chip {
      display: inline-flex; align-items: center; gap: 10px; padding: 11px 15px;
      border: 1px solid; border-radius: 14px; font-size: 14px; font-weight: 650;
      line-height: 1.5;
      animation: rise-in .42s ease-out both;
    }
    .state-chip small { display:block; font-size: 11px; line-height: 1.5; font-weight: 600; opacity: .86; letter-spacing: .04em; text-transform: uppercase; }
    .traffic-alert {
      padding: 16px 18px; border: 1px solid; border-radius: 16px;
      margin: 12px 0 4px; animation: rise-in .42s ease-out both;
    }
    .traffic-alert strong { display:block; font-size: 14px; line-height: 1.45; letter-spacing: .02em; }
    .traffic-alert p { margin: 6px 0 0; color: #CBD5E1; font-size: 14px; line-height: 1.55; }
    .section-eyebrow {
      color: #8EACD2; font: 650 11px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: .08em; text-transform: uppercase; margin: 0 0 7px;
    }
    .st-key-speed_probabilities,
    .st-key-forecast_comparison,
    .st-key-transition_heatmap,
    .st-key-steady_distribution,
    .st-key-markov_evolution,
    .st-key-model_benchmark,
    .st-key-peak_trajectory {
      min-width: 0;
      border: 1px solid rgba(108,144,195,.22);
      border-radius: 18px;
      background: #141d3a;
      box-shadow: 0 8px 24px rgba(0,0,0,.14);
      touch-action: pan-y;
      transition: border-color .22s ease, background-color .22s ease;
      animation: scroll-reveal .8s cubic-bezier(.2,.7,.2,1) both;
    }
    .st-key-speed_probabilities:hover,
    .st-key-forecast_comparison:hover,
    .st-key-transition_heatmap:hover,
    .st-key-steady_distribution:hover,
    .st-key-markov_evolution:hover,
    .st-key-model_benchmark:hover,
    .st-key-peak_trajectory:hover {
      border-color: rgba(108,144,195,.48);
      background: #182440;
    }
    .st-key-markov_evolution .js-line,
    .st-key-peak_trajectory .js-line {
      stroke-dasharray: 4500;
      stroke-dashoffset: 4500;
      animation: graph-draw 1.6s ease-out both;
    }
    @supports (animation-timeline: view()) {
      .st-key-speed_probabilities,
      .st-key-forecast_comparison,
      .st-key-transition_heatmap,
      .st-key-steady_distribution,
      .st-key-markov_evolution,
      .st-key-model_benchmark,
      .st-key-peak_trajectory {
        animation: scroll-reveal auto linear both;
        animation-timeline: view();
        animation-range: entry 0% cover 34%;
      }
      .st-key-markov_evolution .js-line,
      .st-key-peak_trajectory .js-line {
        animation: graph-draw auto linear both;
        animation-timeline: view();
        animation-range: entry 5% cover 40%;
      }
    }
    @media (max-width: 640px) {
      .lab-hero { padding: 24px 20px; border-radius: 20px; }
      .lab-title { font-size: clamp(28px, 8vw, 38px); line-height: 1.16; }
      .lab-subtitle { font-size: 14px; }
      .lab-kicker { font-size: 10px; letter-spacing: .07em; }
      .st-key-speed_probabilities,
      .st-key-forecast_comparison,
      .st-key-transition_heatmap,
      .st-key-steady_distribution,
      .st-key-markov_evolution,
      .st-key-model_benchmark,
      .st-key-peak_trajectory { border-radius: 16px; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *:before, *:after { animation: none !important; scroll-behavior: auto !important; transition: none !important; }
    }
    </style>
    """
)


@st.cache_resource(show_spinner=False)
def load_artifacts() -> dict:
    required_files = [
        ARTIFACT_PATH,
        DATA_PATH,
        METRICS_PATH,
        *(OUTPUT_DIR / name for name in CHART_NAMES),
    ]
    if not all(path.exists() for path in required_files):
        with st.spinner("Preparing the synthetic dataset, models, and research charts…"):
            if not DATA_PATH.exists():
                generate_dataset()
            return train_and_evaluate()
    return joblib.load(ARTIFACT_PATH)


def set_preset(speed, flow, occupancy, regime, weather, incident) -> None:
    values = (speed, flow, occupancy, regime, weather, incident)
    keys = ("speed", "flow", "occupancy", "regime", "weather", "incident")
    for key, value in zip(keys, values):
        st.session_state[key] = value


def style_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=NAVY,
        plot_bgcolor=PANEL,
        font=dict(
            color=WHITE,
            family='-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif',
        ),
        margin=dict(l=20, r=20, t=55, b=24),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12),
        hoverlabel=dict(bgcolor=INDIGO, bordercolor=BORDER, font_color=WHITE),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def predict(art, speed, flow, occupancy, regime, weather, incident):
    state = speed_state(speed)
    probabilities = art["markov_matrices"][regime][state]
    hour = {"Morning Peak": 8.5, "Off-Peak": 13.0, "Evening Peak": 18.0}[regime]
    base = pd.DataFrame(
        [
            {
                "speed_kmh": speed,
                "flow_veh_min": flow,
                "occupancy_pct": occupancy,
                "hour": hour,
                "day_of_week": 2,
                "weather_code": WEATHER_CODE[weather],
                "incident_flag": int(incident),
            }
        ]
    )
    hybrid = base.copy()
    hybrid[["P_Free", "P_Slow", "P_Cong"]] = probabilities
    inputs = {
        "OLS Linear Regression": base[OLS_FEATURES],
        "XGBoost": base,
        "Hybrid Markov + XGBoost": hybrid,
    }
    forecasts = [
        float(art["models"][name].predict(inputs[name])[0])
        for name in ("OLS Linear Regression", "XGBoost", "Hybrid Markov + XGBoost")
    ]
    probability_map = dict(
        zip(("P_Free", "P_Slow", "P_Cong"), map(float, probabilities))
    )
    return forecasts, probability_map


for key, value in {
    "speed": 52.0,
    "flow": 28.0,
    "occupancy": 38.0,
    "regime": "Off-Peak",
    "weather": "Clear",
    "incident": False,
}.items():
    st.session_state.setdefault(key, value)

art = load_artifacts()
metrics = art["metrics"]
hybrid_row = metrics.loc[metrics.Model.eq("Hybrid Markov + XGBoost")].iloc[0]
ols_rmse = float(
    metrics.loc[metrics.Model.eq("OLS Linear Regression"), "RMSE (km/h)"].iloc[0]
)
improvement = (
    ols_rmse - float(hybrid_row["RMSE (km/h)"])
) / max(ols_rmse, 1e-8) * 100

st.html(
    """
    <header class="lab-hero">
      <div class="lab-kicker">Urban mobility / stochastic systems / 5-minute horizon</div>
      <h1 class="lab-title">AI-Enhanced Markov Chain Model for Short-Term Traffic Congestion Prediction</h1>
      <p class="lab-subtitle">Applied Statistics &amp; Stochastic Processes &nbsp;·&nbsp; Module V + Module IV</p>
      <div class="lab-live"><span class="lab-live-dot"></span> Forecast engine ready</div>
    </header>
    """
)

kpis = st.columns(4)
for column, (label, value) in zip(
    kpis,
    (
        ("Hybrid RMSE", f"{hybrid_row['RMSE (km/h)']:.2f} km/h"),
        ("RMSE vs. OLS", f"{improvement:.1f}%"),
        ("Peak-hour RMSE", f"{hybrid_row['Peak-Hour RMSE (km/h)']:.2f} km/h"),
        (
            "Inference latency (ms/sample)",
            f"{hybrid_row['Inference Latency (ms/sample)']:.4f}",
        ),
    ),
):
    with column.container(border=True):
        st.metric(label, value)
if improvement < 15:
    st.caption(
        "Measured on the held-out test split. The requested 15% reduction is not "
        "achieved by this run; metrics are reported as measured, not hard-coded."
    )

tabs = st.tabs(
    [
        "Live simulator",
        "Markov engine",
        "Model benchmarks",
        "Data & exports",
    ]
)

with tabs[0]:
    st.markdown('<p class="section-eyebrow">01 / interactive forecast console</p>', unsafe_allow_html=True)
    st.subheader("Live 5-minute congestion simulator")
    presets = [
        ("Morning Rush Cascade", (24.0, 48.0, 78.0, "Morning Peak", "Clear", False)),
        ("Sudden Rainstorm", (38.0, 33.0, 62.0, "Off-Peak", "Rain", False)),
        ("Highway Lane Incident", (21.0, 42.0, 83.0, "Evening Peak", "Clear", True)),
        ("Late Night Free Flow", (74.0, 16.0, 18.0, "Off-Peak", "Clear", False)),
    ]
    preset_columns = st.columns(4)
    for column, (label, values) in zip(preset_columns, presets):
        column.button(
            label,
            key=f"preset_{label}",
            on_click=set_preset,
            args=values,
            width="stretch",
            icon=":material/bolt:",
        )

    speed_columns = st.columns(3)
    speed = speed_columns[0].slider(
        "Current speed (km/h)", 5.0, 90.0, key="speed"
    )
    flow = speed_columns[1].slider(
        "Traffic flow (veh/min)", 5.0, 60.0, key="flow"
    )
    occupancy = speed_columns[2].slider(
        "Occupancy (%)", 5.0, 95.0, key="occupancy"
    )
    context_columns = st.columns(3)
    regime = context_columns[0].selectbox("Time regime", REGIMES, key="regime")
    weather = context_columns[1].selectbox(
        "Weather", ["Clear", "Rain", "Fog"], key="weather"
    )
    incident = context_columns[2].selectbox(
        "Incident status",
        [False, True],
        key="incident",
        format_func=lambda active: "Active incident" if active else "No incident",
    )

    forecasts, probabilities = predict(
        art, speed, flow, occupancy, regime, weather, incident
    )
    state = speed_state(speed)
    fill, border, accent = STATE_STYLE[state]
    st.html(
        f"""
        <div class="state-chip" style="background:{fill};border-color:{border};color:{accent}">
          <span aria-hidden="true">●</span>
          <span><small>Current traffic state</small>{STATES[state]} &nbsp;·&nbsp; {speed:.1f} km/h</span>
        </div>
        """
    )

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Markov next-state probabilities")
            prob_names = ["P_Free", "P_Slow", "P_Cong"]
            probability_figure = go.Figure(
                go.Bar(
                    x=[probabilities[key] for key in prob_names],
                    y=STATES,
                    orientation="h",
                    marker_color=[GREEN, AMBER, RED],
                    text=[f"{probabilities[key]:.1%}" for key in prob_names],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{y}: %{x:.1%}<extra></extra>",
                )
            )
            probability_figure.update_layout(
                height=230,
                xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
                yaxis=dict(autorange="reversed"),
                showlegend=False,
            )
            st.plotly_chart(
                style_chart(probability_figure),
                width="stretch",
                config={"displayModeBar": False, "scrollZoom": False},
                key="speed_probabilities",
                alt="Markov probabilities for free flow, slowing, and congested traffic",
            )
            st.caption("Conditional on the current speed state and selected time regime.")
    with right:
        with st.container(border=True):
            st.markdown("#### Predicted speed in 5 minutes")
            model_labels = [
                "OLS baseline",
                "Standalone XGBoost",
                "Hybrid Markov + XGBoost",
            ]
            forecast_figure = go.Figure(
                go.Bar(
                    x=forecasts,
                    y=model_labels,
                    orientation="h",
                    marker_color=[GREY, BLUE, BORDER],
                    text=[f"{value:.1f} km/h" for value in forecasts],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{y}: %{x:.2f} km/h<extra></extra>",
                )
            )
            forecast_figure.update_layout(
                height=280,
                xaxis_title="Forecast speed (km/h)",
                xaxis_range=[0, max(90, max(forecasts) * 1.2)],
                showlegend=False,
            )
            st.plotly_chart(
                style_chart(forecast_figure),
                width="stretch",
                config={"displayModeBar": False, "scrollZoom": False},
                key="forecast_comparison",
                alt="Five-minute speed forecasts from OLS, standalone XGBoost, and the hybrid model",
            )

    if probabilities["P_Cong"] >= 0.45 or forecasts[2] < 35:
        st.html(
            """
            <div class="traffic-alert" style="background:#2B0307;border-color:#75020F;color:#F1F5F9">
              <strong>Early congestion warning</strong>
              <p>Congestion probability is elevated or the hybrid forecast is below 35 km/h.</p>
            </div>
            """
        )
    else:
        st.html(
            """
            <div class="traffic-alert" style="background:#022B22;border-color:#275D46;color:#569578">
              <strong>Traffic outlook · clear</strong>
              <p>No early congestion trigger for the selected conditions.</p>
            </div>
            """
        )

with tabs[1]:
    st.markdown('<p class="section-eyebrow">02 / module V · discrete-time Markov chains</p>', unsafe_allow_html=True)
    st.subheader("Regime-specific transition engine")
    regime_view = st.segmented_control(
        "Select time regime",
        REGIMES,
        default=REGIMES[0],
        key="matrix_regime",
    )
    if regime_view is None:
        regime_view = REGIMES[0]
    matrix = art["markov_matrices"][regime_view]
    matrix_column, stationary_column = st.columns([1.35, 0.65], gap="large")
    with matrix_column:
        heatmap = go.Figure(
            go.Heatmap(
                z=matrix,
                x=STATES,
                y=STATES,
                zmin=0,
                zmax=1,
                colorscale=[[0, INDIGO], [0.5, BORDER], [1, BLUE]],
                text=np.round(matrix, 3),
                texttemplate="%{text:.2f}",
                hovertemplate="From %{y} to %{x}: %{z:.3f}<extra></extra>",
                colorbar=dict(title="Probability"),
            )
        )
        heatmap.update_layout(
            title=f"One-step transitions · {regime_view}",
            xaxis_title="Next state",
            yaxis_title="Current state",
            height=440,
        )
        st.plotly_chart(
            style_chart(heatmap),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False},
            key="transition_heatmap",
            alt=f"Markov transition probabilities for {regime_view}",
        )
    with stationary_column:
        st.markdown("#### Steady-state vector")
        steady = art["steady_states"][regime_view]
        steady_figure = go.Figure(
            go.Bar(
                x=steady,
                y=STATES,
                orientation="h",
                marker_color=[GREEN, AMBER, RED],
                text=[f"{value:.1%}" for value in steady],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:.1%}<extra></extra>",
            )
        )
        steady_figure.update_layout(
            height=210,
            xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
            yaxis=dict(autorange="reversed"),
            showlegend=False,
        )
        st.plotly_chart(
            style_chart(steady_figure),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False},
            key="steady_distribution",
            alt="Long-run probability of each traffic state",
        )
        st.code(
            f"π = [{steady[0]:.3f}, {steady[1]:.3f}, {steady[2]:.3f}]\nπP = π",
            language="text",
        )
        st.caption("Long-run state probabilities satisfying πP = π and Σπᵢ = 1.")
    row_sums = matrix.sum(axis=1)
    st.caption(
        "Rows are current states; columns are next states. "
        + " · ".join(
            f"Row {index} sum: {total:.2f}"
            for index, total in enumerate(row_sums)
        )
    )
    st.markdown(
        "**State definitions** · State 0 Free Flow (speed ≥ 50 km/h) · "
        "State 1 Slowing (30 ≤ speed < 50 km/h) · "
        "State 2 Congested (speed < 30 km/h)."
    )
    st.dataframe(
        pd.DataFrame(matrix, index=STATES, columns=STATES).style.format("{:.3f}"),
        width="stretch",
    )
    st.markdown("#### Markov-chain convergence")
    starting_state = st.segmented_control(
        "Starting traffic state",
        STATES,
        default=STATES[0],
        key="chain_starting_state",
    )
    if starting_state is None:
        starting_state = STATES[0]
    distribution_history = [np.eye(len(STATES))[STATES.index(starting_state)]]
    for _ in range(24):
        distribution_history.append(distribution_history[-1] @ matrix)
    chain_steps = np.arange(len(distribution_history))
    state_colors = (GREEN, AMBER, RED)
    chain_figure = go.Figure()
    for state_index, state_name in enumerate(STATES):
        chain_figure.add_scatter(
            x=chain_steps,
            y=[distribution[state_index] for distribution in distribution_history],
            name=state_name,
            mode="lines+markers",
            marker=dict(size=5),
            line=dict(color=state_colors[state_index], width=2.5),
            hovertemplate=f"Step %{{x}}<br>{state_name}: %{{y:.1%}}<extra></extra>",
        )
        chain_figure.add_hline(
            y=float(art["steady_states"][regime_view][state_index]),
            line_color=state_colors[state_index],
            line_dash="dot",
            line_width=1,
            opacity=0.55,
        )
    chain_figure.update_layout(
        title=f"State probability evolution from {starting_state}",
        xaxis_title="Five-minute transition step",
        yaxis_title="State probability",
        xaxis=dict(range=[0, 24], dtick=4),
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        hovermode="x unified",
        height=440,
    )
    with st.container(border=True):
        st.plotly_chart(
            style_chart(chain_figure),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False},
            key="markov_evolution",
            alt="Traffic-state probabilities converging toward the steady state",
        )
        st.caption(
            "The curves trace the 24-step evolution as the chart enters the viewport. "
            "Dotted lines show the computed steady-state probabilities."
        )

with tabs[2]:
    st.markdown('<p class="section-eyebrow">03 / module IV + AI · regression laboratory</p>', unsafe_allow_html=True)
    st.subheader("Test-set model benchmarks")
    benchmark_left, benchmark_right = st.columns([1.2, 0.8], gap="large")
    with benchmark_left:
        benchmark = go.Figure()
        benchmark.add_bar(
            name="Overall RMSE",
            x=metrics.Model,
            y=metrics["RMSE (km/h)"],
            marker_color=BLUE,
            hovertemplate="%{x}<br>Overall: %{y:.3f} km/h<extra></extra>",
        )
        benchmark.add_bar(
            name="Peak-hour RMSE",
            x=metrics.Model,
            y=metrics["Peak-Hour RMSE (km/h)"],
            marker_color=GREY,
            hovertemplate="%{x}<br>Peak hour: %{y:.3f} km/h<extra></extra>",
        )
        benchmark.update_layout(
            barmode="group",
            title="Speed prediction error by model",
            yaxis_title="RMSE (km/h)",
            height=440,
        )
        st.plotly_chart(
            style_chart(benchmark),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False},
            key="model_benchmark",
            alt="Overall and peak-hour RMSE for the four regression models",
        )
    with benchmark_right:
        st.markdown("#### Metrics explorer")
        selected_model = st.selectbox(
            "Inspect model",
            MODEL_NAMES,
            index=MODEL_NAMES.index("Hybrid Markov + XGBoost"),
            key="benchmark_model",
        )
        selected_metrics = metrics.loc[metrics.Model.eq(selected_model)].iloc[0]
        for labels in (("RMSE (km/h)", "MAE (km/h)"), ("MAPE (%)", "R2")):
            metric_columns = st.columns(2)
            for column, label in zip(metric_columns, labels):
                column.metric(label, f"{selected_metrics[label]:.4f}")
        st.caption(
            f"Peak-hour RMSE: {selected_metrics['Peak-Hour RMSE (km/h)']:.4f} km/h "
            f"· Latency: {selected_metrics['Inference Latency (ms/sample)']:.5f} ms/sample"
        )
        if selected_model == "Hybrid Markov + XGBoost":
            xgb_peak = float(
                metrics.loc[
                    metrics.Model.eq("XGBoost"), "Peak-Hour RMSE (km/h)"
                ].iloc[0]
            )
            st.info(
                f"Hybrid vs. standalone XGBoost at peak hour: "
                f"{xgb_peak - float(hybrid_row['Peak-Hour RMSE (km/h)']):+.3f} km/h "
                "RMSE difference."
            )

    test = art["test"]
    peak_indices = np.flatnonzero(
        test.regime.isin(["Morning Peak", "Evening Peak"]).to_numpy()
    )
    max_window = min(240, len(peak_indices))
    trajectory_window = st.slider(
        "Peak-hour trajectory window (5-minute samples)",
        min_value=30,
        max_value=max(30, max_window),
        value=min(120, max_window),
        step=30,
        key="trajectory_window",
    )
    max_start = max(0, len(peak_indices) - trajectory_window)
    trajectory_start = st.slider(
        "Scroll through peak-hour test samples",
        min_value=0,
        max_value=max_start,
        value=0,
        key="trajectory_start",
    )
    selected_indices = peak_indices[
        trajectory_start : trajectory_start + trajectory_window
    ]
    plotdf = test.iloc[selected_indices]
    trajectory_values = (
        plotdf.target_speed_5min.to_numpy(),
        art["test_predictions"]["Hybrid Markov + XGBoost"][selected_indices],
        art["test_predictions"]["XGBoost"][selected_indices],
    )
    trajectory_colors = (WHITE, BLUE, BORDER)
    trajectory_names = ("Actual", "Hybrid", "Standalone XGBoost")

    def trajectory_path(values, end):
        x_values = []
        y_values = []
        for index in range(end):
            if (
                index
                and plotdf.timestamp.iloc[index] - plotdf.timestamp.iloc[index - 1]
                > pd.Timedelta(minutes=5)
            ):
                x_values.append(None)
                y_values.append(None)
            x_values.append(plotdf.timestamp.iloc[index])
            y_values.append(float(values[index]))
        return x_values, y_values

    trajectory = go.Figure()
    for values, name, color in zip(
        trajectory_values, trajectory_names, trajectory_colors
    ):
        line_x, line_y = trajectory_path(values, len(plotdf))
        trajectory.add_scatter(
            x=line_x,
            y=line_y,
            name=name,
            mode="lines",
            line=dict(color=color, width=2 if name != "Standalone XGBoost" else 1.5),
        )
    trajectory.update_layout(
        title="Actual vs. predicted peak-hour speeds",
        xaxis_title="Timestamp",
        yaxis_title="Speed (km/h)",
        xaxis=dict(range=[plotdf.timestamp.iloc[0], plotdf.timestamp.iloc[-1]]),
        yaxis=dict(
            range=[
                float(np.floor(np.min(np.concatenate(trajectory_values)) - 2)),
                float(np.ceil(np.max(np.concatenate(trajectory_values)) + 2)),
            ]
        ),
        height=460,
        hovermode="x unified",
    )
    with st.container(border=True):
        st.plotly_chart(
            style_chart(trajectory),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False},
            key="peak_trajectory",
            alt="Actual and predicted speed trajectories during peak hours",
        )

    equation = f"ŷ = {art['ols_intercept']:.5f}" + "".join(
        f" {coefficient:+.5f}·{feature}"
        for feature, coefficient in art["ols_coefficients"].items()
    )
    with st.expander("Fitted OLS equation and full metric table", expanded=False):
        st.code(equation, language="text")
        st.caption(
            "Exact learned coefficients. Each z-feature is standardized with "
            "training-split statistics."
        )
        st.dataframe(
            metrics.style.format(
                {
                    column: "{:.4f}"
                    for column in metrics.select_dtypes(include="number").columns
                }
            ),
            width="stretch",
        )
        st.caption(
            "Regression metrics only. Peak-hour scores use Morning Peak and "
            "Evening Peak samples."
        )

with tabs[3]:
    st.markdown('<p class="section-eyebrow">04 / reproducibility · data & deliverables</p>', unsafe_allow_html=True)
    st.subheader("Dataset, preprocessing & export center")
    raw = art["raw"]
    clean = art["clean"]
    data_columns = st.columns(3)
    data_columns[0].metric("Rows", f"{len(clean):,}")
    data_columns[1].metric(
        "Raw missing values", f"{sum(art['imputation_counts'].values()):,}"
    )
    data_columns[2].metric(
        "Clean missing values", f"{int(clean.isna().sum().sum()):,}"
    )
    st.write(
        f"**Dataset:** {len(clean):,} rows × {len(clean.columns)} columns"
    )
    st.write(
        "**Chronological split:**",
        " · ".join(
            f"{name.title()}: {count:,} ({count / len(clean):.0%})"
            for name, count in art["split_sizes"].items()
        ),
    )
    st.write("**Imputation counts:**", art["imputation_counts"])
    st.write(
        "**Training-only 5-fold OLS CV RMSE:**",
        f"{np.mean(art['cv_rmse']):.3f} ± {np.std(art['cv_rmse']):.3f} km/h",
    )
    preview_columns = st.columns([1, 2], gap="large")
    with preview_columns[0]:
        st.markdown("#### Sensor health")
        st.json(
            {
                "raw_missing_values": int(raw.isna().sum().sum()),
                "clean_missing_values": int(clean.isna().sum().sum()),
                "raw_speed_mean": float(raw.speed_kmh.mean()),
                "clean_speed_mean": float(clean.speed_kmh.mean()),
            }
        )
        preview_regime = st.selectbox(
            "Preview time regime",
            ["All regimes", *REGIMES],
            key="preview_regime",
        )
        preview_rows = st.slider(
            "Preview rows", 10, 100, 30, step=10, key="preview_rows"
        )
    preview = clean
    if preview_regime != "All regimes":
        preview = preview.loc[preview.regime.eq(preview_regime)]
    with preview_columns[1]:
        st.markdown("#### Interactive data preview")
        st.dataframe(
            preview.head(preview_rows),
            width="stretch",
            height=390,
        )
    st.caption(
        "Linear interpolation → physical range clipping → chronological "
        "70/15/15 split → training-only feature scaling."
    )

    action_columns = st.columns(3)
    if action_columns[0].button(
        "Regenerate data & retrain",
        width="stretch",
        icon=":material/model_training:",
        key="retrain_all",
    ):
        load_artifacts.clear()
        with st.spinner("Regenerating the dataset and fitting all four models…"):
            generate_dataset()
            train_and_evaluate()
        st.success("All models and reports have been refreshed.")
        st.rerun()

    metrics_csv = metrics.to_csv(index=False).encode("utf-8")
    action_columns[1].download_button(
        "Download metrics (CSV)",
        metrics_csv,
        "metrics_summary.csv",
        "text/csv",
        width="stretch",
        icon=":material/download:",
    )
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for chart_name in CHART_NAMES:
            chart_path = OUTPUT_DIR / chart_name
            if not chart_path.exists():
                raise FileNotFoundError(
                    f"Expected chart was not generated: {chart_path}"
                )
            archive.write(chart_path, arcname=chart_name)
        archive.writestr("metrics_summary.csv", metrics_csv)
    action_columns[2].download_button(
        "Download charts + metrics (ZIP)",
        zip_buffer.getvalue(),
        "traffic_research_outputs.zip",
        "application/zip",
        width="stretch",
        icon=":material/folder_zip:",
    )
