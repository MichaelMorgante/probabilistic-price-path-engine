from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src importable when running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from mt5_loader import connect_mt5, load_mt5_candles, make_synthetic_ohlc
from simulator import estimate_mu_sigma, simulate_gbm_paths, simulate_bootstrap_paths
from probability_engine import pathwise_tp_sl_metrics, analytical_gbm_terminal_metrics
from charts import make_price_path_figure
from utils import append_probability_log, format_pct, format_price


st.set_page_config(
    page_title="Model 2 | Probabilistic Price Paths",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: #050b12;
        color: #d6dde8;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.05rem;
    }
    .model-header {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 18px 20px;
        background: linear-gradient(180deg, #08121d 0%, #050b12 100%);
        margin-bottom: 14px;
    }
    .small-muted {
        color: #8c98a8;
        font-size: 0.90rem;
    }
    .green-dot {
        color: #43d18d;
        font-weight: 800;
    }
    .panel {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 18px;
        background: #071018;
        min-height: 120px;
    }
    .metric-label {
        color: #9aa6b6;
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    .metric-big-green {
        color: #43d18d;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-big-red {
        color: #ff5b5b;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-big-blue {
        color: #6da8ff;
        font-size: 2.0rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .divider {
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 16px 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def timeframe_to_freq(tf: str) -> str:
    return {
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
    }.get(tf, "1min")


def run_model(
    df: pd.DataFrame,
    horizon: int,
    n_paths: int,
    vol_window: int,
    model_type: str,
    drift_mode: str,
    direction: str,
    tp_points: float,
    sl_points: float,
    seed: int,
):
    current_price = float(df["Close"].iloc[-1])
    mu, sigma, log_returns = estimate_mu_sigma(
        df["Close"],
        window=vol_window,
        drift_mode=drift_mode,
    )

    if model_type == "GBM":
        paths = simulate_gbm_paths(
            current_price=current_price,
            mu=mu,
            sigma=sigma,
            horizon=horizon,
            n_paths=n_paths,
            seed=seed,
        )
    else:
        paths = simulate_bootstrap_paths(
            current_price=current_price,
            historical_returns=log_returns,
            horizon=horizon,
            n_paths=n_paths,
            sample_window=vol_window,
            seed=seed,
        )

    metrics = pathwise_tp_sl_metrics(
        paths=paths,
        entry_price=current_price,
        direction=direction.lower(),
        tp_points=tp_points,
        sl_points=sl_points,
    )

    analytical_metrics = analytical_gbm_terminal_metrics(
        current_price=current_price,
        mu=mu,
        sigma=sigma,
        horizon=horizon,
        direction=direction.lower(),
        tp_points=tp_points,
        sl_points=sl_points,
    )

    metrics.update(analytical_metrics)

    metrics.update({
        "symbol": st.session_state.get("symbol", "NAS100"),
        "timeframe": st.session_state.get("timeframe", "M1"),
        "horizon": horizon,
        "n_paths": n_paths,
        "vol_window": vol_window,
        "model_type": model_type,
        "drift_mode": drift_mode,
        "sigma": sigma,
        "mu": mu,
        "last_candle_time": str(df["time"].iloc[-1]),
    })

    return paths, metrics


st.markdown(
    """
    <div class="model-header">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:1.35rem; font-weight:800; letter-spacing:0.08em;">
                    MODEL 2 <span class="green-dot">•</span> PROBABILISTIC PRICE PATHS
                </div>
                <div class="small-muted" style="margin-top:8px;">
                    Monte Carlo simulation of possible future price paths. Updates after each candle close / refresh.
                </div>
            </div>
            <div class="panel" style="min-height:0; padding:12px 16px;">
                <span class="small-muted">MODEL STATUS</span>
                <span class="green-dot"> ● ACTIVE</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


control_cols = st.columns([1.3, 1, 1, 1, 1, 1, 1])

with control_cols[0]:
    symbol = st.text_input("Symbol", value="NAS100", key="symbol")

with control_cols[1]:
    timeframe = st.selectbox(
        "Timeframe",
        ["M1", "M5", "M15", "M30", "H1"],
        index=0,
        key="timeframe",
    )

with control_cols[2]:
    horizon = st.selectbox("Sim horizon", [5, 10, 20, 30, 50], index=2)

with control_cols[3]:
    n_paths = st.selectbox("Paths", [100, 500, 1000, 5000], index=2)

with control_cols[4]:
    model_type = st.selectbox("Model", ["GBM", "Bootstrap"], index=0)

with control_cols[5]:
    direction = st.selectbox("Direction", ["Long", "Short"], index=0)

with control_cols[6]:
    use_mt5 = st.toggle("Use MT5", value=False)


settings_cols = st.columns([1, 1, 1, 1, 1])

with settings_cols[0]:
    vol_window = st.number_input(
        "Vol window",
        min_value=20,
        max_value=500,
        value=80,
        step=10,
    )

with settings_cols[1]:
    drift_mode = st.selectbox("Drift", ["zero", "historical"], index=0)

with settings_cols[2]:
    tp_points = st.number_input(
        "TP points",
        min_value=1.0,
        max_value=1000.0,
        value=45.0,
        step=1.0,
    )

with settings_cols[3]:
    sl_points = st.number_input(
        "SL points",
        min_value=1.0,
        max_value=1000.0,
        value=25.0,
        step=1.0,
    )

with settings_cols[4]:
    auto_refresh = st.toggle("Auto refresh", value=False)


data_status = "Synthetic data"
df = None

if use_mt5:
    ok, msg = connect_mt5()

    if ok:
        loaded = load_mt5_candles(symbol, timeframe=timeframe, bars=350)

        if loaded is not None and len(loaded) > vol_window + 20:
            df = loaded
            data_status = f"MT5 live data: {symbol}"
        else:
            data_status = "MT5 connected, but data unavailable. Using synthetic fallback."
    else:
        data_status = f"{msg} Using synthetic fallback."

if df is None:
    df = make_synthetic_ohlc(
        n=350,
        start_price=21500.0,
        seed=42,
        freq=timeframe_to_freq(timeframe),
    )


try:
    paths, metrics = run_model(
        df=df,
        horizon=int(horizon),
        n_paths=int(n_paths),
        vol_window=int(vol_window),
        model_type=model_type,
        drift_mode=drift_mode,
        direction=direction,
        tp_points=float(tp_points),
        sl_points=float(sl_points),
        seed=42,
    )

    append_probability_log(
        metrics=metrics,
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_type,
        output_path=PROJECT_ROOT / "reports" / "probability_logs.csv",
    )

    left, right = st.columns([4.8, 1.2])

    with left:
        last = df.iloc[-1]

        price_line = (
            f"**{symbol}** · **{timeframe}** · "
            f"O {last['Open']:,.1f} &nbsp;&nbsp; H {last['High']:,.1f} &nbsp;&nbsp; "
            f"L {last['Low']:,.1f} &nbsp;&nbsp; C {last['Close']:,.1f}"
        )

        st.markdown(price_line, unsafe_allow_html=True)

        fig = make_price_path_figure(df, paths, metrics)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### MODEL 2 OUTPUTS")
        st.markdown(
            f'<div class="small-muted">As of {metrics["last_candle_time"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        tp_class = (
            "metric-big-green"
            if metrics["p_tp_first"] >= metrics["p_sl_first"]
            else "metric-big-red"
        )

        st.markdown(
            '<div class="metric-label">P(TP before SL)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="{tp_class}">{format_pct(metrics["p_tp_first"])}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-label">P(SL before TP)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-big-red">{format_pct(metrics["p_sl_first"])}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-label">Expected Price</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-big-blue">{format_price(metrics["expected_price"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="small-muted">move: {metrics["expected_move"]:+.2f} pts</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-label">Expected Range</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:1.35rem;">'
            f'<span style="color:#ff5b5b;">{format_price(metrics["p5"])}</span>'
            f' — <span style="color:#43d18d;">{format_price(metrics["p95"])}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="small-muted">5% — 95%</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-label">Std. Deviation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:1.5rem;">{metrics["std_terminal"]:,.2f}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### Analytical GBM")
        st.markdown(
            '<div class="small-muted">Terminal-price benchmark, not TP/SL touch-first probability.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="metric-label">GBM Terminal TP Probability</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-big-green">{format_pct(metrics["gbm_analytical_terminal_tp_prob"])}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="metric-label">GBM Terminal SL Probability</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-big-red">{format_pct(metrics["gbm_analytical_terminal_sl_prob"])}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="metric-label">GBM Expected Price</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:1.35rem;">{format_price(metrics["gbm_analytical_expected_price"])}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="small-muted">'
            f'GBM move: {metrics["gbm_analytical_expected_move"]:+.2f} pts'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-label">Simulations</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div>{int(n_paths):,} paths</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="small-muted">{data_status}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Model failed: {e}")


if auto_refresh:
    time.sleep(10)
    st.rerun()