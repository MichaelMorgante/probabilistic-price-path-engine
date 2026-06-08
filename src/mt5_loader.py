from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except Exception:
    mt5 = None
    MT5_AVAILABLE = False


def timeframe_map():
    if not MT5_AVAILABLE:
        return {}
    return {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }


def connect_mt5() -> tuple[bool, str]:
    if not MT5_AVAILABLE:
        return False, "MetaTrader5 package not installed."

    if not mt5.initialize():
        return False, f"MT5 initialize failed: {mt5.last_error()}"

    return True, "MT5 connected."


def load_mt5_candles(symbol: str, timeframe: str = "M1", bars: int = 300) -> Optional[pd.DataFrame]:
    if not MT5_AVAILABLE:
        return None

    tf_map = timeframe_map()
    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if not mt5.symbol_select(symbol, True):
        return None

    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "Volume",
    })

    return df[["time", "Open", "High", "Low", "Close", "Volume"]]


def make_synthetic_ohlc(
    n: int = 300,
    start_price: float = 21500.0,
    seed: int = 42,
    freq: str = "1min",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq=freq)

    vol = 0.00075
    shocks = rng.normal(0, vol, size=n)
    drift = 0.00002
    close = start_price * np.exp(np.cumsum(drift + shocks))

    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(3, 18, size=n)
    low = np.minimum(open_, close) - rng.uniform(3, 18, size=n)
    volume = rng.integers(100, 2000, size=n)

    return pd.DataFrame({
        "time": times,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })