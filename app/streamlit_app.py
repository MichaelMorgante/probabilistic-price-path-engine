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
        "H4": "4h",
        "D1": "1D",
    }.get(tf, "1min")

def initialise_trade_state() -> None:
    if "locked_trade" not in st.session_state:
        st.session_state.locked_trade = None

    if "locked_trade_log" not in st.session_state:
        st.session_state.locked_trade_log = []

    if "latest_metrics_snapshot" not in st.session_state:
        st.session_state.latest_metrics_snapshot = None

initialise_trade_state()

def create_locked_trade(
    direction: str,
    metrics_snapshot: dict,
    tp_points: float,
    sl_points: float,
) -> dict:
    direction = direction.lower()

    entry_price = float(metrics_snapshot["entry_price"])

    if direction == "long":
        tp_level = entry_price + tp_points
        sl_level = entry_price - sl_points
    elif direction == "short":
        tp_level = entry_price - tp_points
        sl_level = entry_price + sl_points
    else:
        raise ValueError("direction must be 'long' or 'short'.")

    return {
        "status": "active",
        "skip_detection_once": True,
        "direction": direction,
        "entry_price": float(entry_price),
        "tp_level": float(tp_level),
        "sl_level": float(sl_level),
        "entry_time": str(metrics_snapshot["last_candle_time"]),
        "entry_timestamp": pd.Timestamp.now(),
        "p_tp_at_lock": float(metrics_snapshot["p_tp_first"]),
        "p_sl_at_lock": float(metrics_snapshot["p_sl_first"]),
        "p_tp_touched_at_lock": float(metrics_snapshot.get("p_tp_touched", 0.0)),
        "p_sl_touched_at_lock": float(metrics_snapshot.get("p_sl_touched", 0.0)),
        "model_type_at_lock": str(metrics_snapshot.get("model_type", "")),
        "horizon_at_lock": int(metrics_snapshot.get("horizon", 0)),
        "n_paths_at_lock": int(metrics_snapshot.get("n_paths", 0)),
        "exit_time": None,
        "exit_timestamp": None,
        "result": None,
        "exit_price": None,
        "candles_to_hit": None,
        "minutes_to_hit": None,
    }

def update_locked_trade_status(df: pd.DataFrame) -> None:
    locked_trade = st.session_state.locked_trade

    if locked_trade is None:
        return

    if locked_trade["status"] != "active":
        return

    entry_time = pd.to_datetime(locked_trade["entry_time"])
    trade_df = df[pd.to_datetime(df["time"]) > entry_time].copy()

    if trade_df.empty:
        return

    direction = locked_trade["direction"]
    tp_level = locked_trade["tp_level"]
    sl_level = locked_trade["sl_level"]

    for idx, row in trade_df.reset_index(drop=True).iterrows():
        high = float(row["High"])
        low = float(row["Low"])

        if direction == "long":
            tp_hit = high >= tp_level
            sl_hit = low <= sl_level
        else:
            tp_hit = low <= tp_level
            sl_hit = high >= sl_level

        if tp_hit and sl_hit:
            result = "SL"
            exit_price = sl_level
        elif tp_hit:
            result = "TP"
            exit_price = tp_level
        elif sl_hit:
            result = "SL"
            exit_price = sl_level
        else:
            continue

        exit_time = pd.to_datetime(row["time"])
        minutes_to_hit = (exit_time - entry_time).total_seconds() / 60

        locked_trade["status"] = "completed"
        locked_trade["result"] = result
        locked_trade["exit_price"] = float(exit_price)
        locked_trade["exit_time"] = str(row["time"])
        locked_trade["exit_timestamp"] = pd.Timestamp.now()
        locked_trade["candles_to_hit"] = int(idx + 1)
        locked_trade["minutes_to_hit"] = float(minutes_to_hit)

        st.session_state.locked_trade_log.append(locked_trade.copy())
        st.session_state.locked_trade = None
        break


def render_locked_trade_panel(current_price: float) -> None:
    locked_trade = st.session_state.locked_trade

    st.markdown(
        """
        <div class="panel" style="margin-top:16px;">
            <div style="font-size:1.1rem; font-weight:800; letter-spacing:0.08em;">
                LOCKED HYPOTHETICAL TRADE
            </div>
        """,
        unsafe_allow_html=True,
    )

    if locked_trade is None:
        st.markdown(
            '<div class="small-muted" style="margin-top:8px;">No active locked trade. Use Lock Long or Lock Short from the output panel.</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.locked_trade_log:
            latest = st.session_state.locked_trade_log[-1]
            result_colour = "#43d18d" if latest["result"] == "TP" else "#ff5b5b"
            st.markdown(
                f"""
                <div style="margin-top:10px;">
                    <span class="small-muted">Latest completed lock:</span>
                    <span style="color:{result_colour}; font-weight:800;"> {latest["result"]} hit</span>
                    <span class="small-muted"> after {latest["candles_to_hit"]} candles / {latest.get("minutes_to_hit", 0):.1f} mins.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
        return

    direction = locked_trade["direction"]
    entry_price = locked_trade["entry_price"]
    tp_level = locked_trade["tp_level"]
    sl_level = locked_trade["sl_level"]
    move = current_price - entry_price
    if direction == "short":
        move = entry_price - current_price

    direction_colour = "#43d18d" if direction == "long" else "#ff5b5b"

    st.markdown(
        f"""
        <div style="margin-top:10px; line-height:1.8;">
            <div><span class="small-muted">Mode:</span> <span style="color:{direction_colour}; font-weight:700;">LOCKED {direction.upper()}</span></div>
            <div><span class="small-muted">Entry:</span> {entry_price:,.2f}</div>
            <div><span class="small-muted">TP:</span> <span style="color:#43d18d;">{tp_level:,.2f}</span></div>
            <div><span class="small-muted">SL:</span> <span style="color:#ff5b5b;">{sl_level:,.2f}</span></div>
            <div><span class="small-muted">Current:</span> {current_price:,.2f}</div>
            <div><span class="small-muted">Move from entry:</span> {move:+.2f} pts</div>
            <div><span class="small-muted">Status:</span> {locked_trade["status"].upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

def render_locked_trade_log() -> None:
    if not st.session_state.locked_trade_log:
        return

    st.markdown(
        """
        <div class="panel" style="margin-top:16px;">
            <div style="font-size:1.1rem; font-weight:800; letter-spacing:0.08em;">
                LOCKED TRADE SESSION LOG
            </div>
        """,
        unsafe_allow_html=True,
    )

    rows = []
    for i, trade in enumerate(st.session_state.locked_trade_log, start=1):
        rows.append({
            "#": i,
            "Direction": trade["direction"].upper(),
            "Entry Time": trade["entry_time"],
            "Exit Time": trade["exit_time"],
            "Entry": round(trade["entry_price"], 2),
            "TP": round(trade["tp_level"], 2),
            "SL": round(trade["sl_level"], 2),
            "Result": trade["result"],
            "Candles": trade["candles_to_hit"],
            "Minutes": round(trade.get("minutes_to_hit", 0), 1),
            "P(TP) at lock": f'{trade["p_tp_at_lock"]:.1%}',
            "P(SL) at lock": f'{trade["p_sl_at_lock"]:.1%}',
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Clear Trade Log", use_container_width=True):
        st.session_state.locked_trade_log = []
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

def default_refresh_seconds(tf: str) -> int:
    return {
        "M1": 60,
        "M5": 300,
        "M15": 900,
        "M30": 1800,
        "H1": 3600,
        "H4": 14400,
        "D1": 86400,
    }.get(tf, 60)

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
    symbol = st.text_input("Symbol", value="US100.cash", key="symbol")

with control_cols[1]:
    timeframe = st.selectbox(
        "Timeframe",
        ["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
        index=0,
        key="timeframe",
    )

with control_cols[2]:
    horizon = st.selectbox("Sim horizon", [5, 10, 20, 30, 50, 60, 90], index=2)

with control_cols[3]:
    n_paths = st.selectbox("Paths", [100, 500, 1000, 5000, 10000, 25000, 50000], index=2)

with control_cols[4]:
    model_type = st.selectbox("Model", ["GBM", "Bootstrap"], index=1)

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
        value=90,
        step=10,
    )

with settings_cols[1]:
    drift_mode = st.selectbox("Drift", ["zero", "historical"], index=0)

with settings_cols[2]:
    tp_points = st.number_input(
        "TP points",
        min_value=1.0,
        max_value=1000.0,
        value=29.0,
        step=1.0,
    )

with settings_cols[3]:
    sl_points = st.number_input(
        "SL points",
        min_value=1.0,
        max_value=1000.0,
        value=29.0,
        step=1.0,
    )

with settings_cols[4]:
    auto_refresh = st.toggle("Auto refresh", value=False)

    refresh_mode = st.selectbox(
        "Refresh interval",
        ["Timeframe default", "10 seconds", "60 seconds", "5 minutes", "15 minutes", "30 minutes", "1 hour"],
        index=0,
    )

refresh_seconds_map = {
    "10 seconds": 10,
    "60 seconds": 60,
    "5 minutes": 300,
    "15 minutes": 900,
    "30 minutes": 1800,
    "1 hour": 3600,
}

refresh_seconds = (
    default_refresh_seconds(timeframe)
    if refresh_mode == "Timeframe default"
    else refresh_seconds_map[refresh_mode]
)


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

    st.session_state.latest_metrics_snapshot = metrics.copy()

    update_locked_trade_status(df)

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

        render_locked_trade_panel(current_price=float(df["Close"].iloc[-1]))
        render_locked_trade_log()

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### MODEL 2 OUTPUTS")
        st.markdown(
            f'<div class="small-muted">As of {metrics["last_candle_time"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.markdown("### Trade Lock")

        lock_col_1, lock_col_2 = st.columns(2)

        with lock_col_1:
            if st.button("Lock Long", use_container_width=True):
                snapshot = st.session_state.latest_metrics_snapshot or metrics.copy()
                st.session_state.locked_trade = create_locked_trade(
                    direction="long",
                    metrics_snapshot=snapshot,
                    tp_points=float(tp_points),
                    sl_points=float(sl_points),
                )
                st.rerun()

        with lock_col_2:
            if st.button("Lock Short", use_container_width=True):
                snapshot = st.session_state.latest_metrics_snapshot or metrics.copy()
                st.session_state.locked_trade = create_locked_trade(
                    direction="short",
                    metrics_snapshot=snapshot,
                    tp_points=float(tp_points),
                    sl_points=float(sl_points),
                )
                st.rerun()

        if st.session_state.locked_trade is not None:
            if st.button("Clear Locked Trade", use_container_width=True):
                st.session_state.locked_trade = None
                st.rerun()

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

        st.markdown(
            f'<div class="small-muted">P(TP at any time): <span style="color:#43d18d;">{format_pct(metrics["p_tp_touched"])}</span></div>',
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

        st.markdown(
            f'<div class="small-muted">P(SL at any time): <span style="color:#ff5b5b;">{format_pct(metrics["p_sl_touched"])}</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-label">No hit within horizon</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:1.15rem; color:#d6dde8;">{format_pct(metrics["p_neither"])}</div>',
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
        st.markdown("### Analytical GBM Benchmark")
        st.markdown(
            '<div class="small-muted">Terminal-price benchmark, not TP/SL touch-first probability.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="font-size:0.95rem; line-height:1.8; margin-top:8px;">
                <div><span class="small-muted">P(up at horizon):</span> {format_pct(metrics["gbm_analytical_p_terminal_up"])}</div>
                <div><span class="small-muted">Expected:</span> {format_price(metrics["gbm_analytical_expected_price"])}</div>
                <div><span class="small-muted">5%-95%:</span> {format_price(metrics["gbm_analytical_p5"])} — {format_price(metrics["gbm_analytical_p95"])}</div>
                <div><span class="small-muted">Terminal TP/SL:</span> {format_pct(metrics["gbm_analytical_terminal_tp_prob"])} / {format_pct(metrics["gbm_analytical_terminal_sl_prob"])}</div>
            </div>
            """,
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
    time.sleep(refresh_seconds)
    st.rerun()