from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from mt5_loader import load_mt5_candles, make_synthetic_ohlc


DataSource = Literal["mt5", "synthetic", "csv"]


REQUIRED_COLUMNS = ["time", "Open", "High", "Low", "Close", "Volume"]


def validate_ohlc_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and standardise the OHLC dataframe used by the engine.

    All downstream modules expect:

        time | Open | High | Low | Close | Volume
    """

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    out = df[REQUIRED_COLUMNS].copy()
    out["time"] = pd.to_datetime(out["time"])

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["time", "Open", "High", "Low", "Close"])
    out = out.sort_values("time").reset_index(drop=True)

    return out


def load_csv_candles(csv_path: str | Path) -> pd.DataFrame:
    """
    Load historical candles from a CSV file.

    The CSV must contain either:

        time, Open, High, Low, Close, Volume

    or lowercase equivalents:

        time, open, high, low, close, volume
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "tick_volume": "Volume",
    }

    df = df.rename(columns=rename_map)

    return validate_ohlc_schema(df)


def load_market_data(
    source: DataSource = "mt5",
    symbol: str = "US100.cash",
    timeframe: str = "M1",
    bars: int = 350,
    csv_path: Optional[str | Path] = None,
    synthetic_start_price: float = 21500.0,
    synthetic_seed: int = 42,
    synthetic_freq: str = "1min",
) -> pd.DataFrame:
    """
    Universal market-data loading interface.

    This keeps the rest of the project data-source agnostic.

    Supported sources:

        source="mt5"        -> loads candles from the active MetaTrader 5 terminal
        source="synthetic"  -> creates synthetic OHLC data
        source="csv"        -> loads candles from a CSV file

    Returns a standard OHLC dataframe:

        time | Open | High | Low | Close | Volume
    """

    source = source.lower()

    if source == "mt5":
        df = load_mt5_candles(
            symbol=symbol,
            timeframe=timeframe,
            bars=bars,
        )

        if df is None or df.empty:
            raise RuntimeError(
                f"MT5 returned no data for symbol={symbol}, timeframe={timeframe}."
            )

        return validate_ohlc_schema(df)

    if source == "synthetic":
        df = make_synthetic_ohlc(
            n=bars,
            start_price=synthetic_start_price,
            seed=synthetic_seed,
            freq=synthetic_freq,
        )

        return validate_ohlc_schema(df)

    if source == "csv":
        if csv_path is None:
            raise ValueError("csv_path must be provided when source='csv'.")

        return load_csv_candles(csv_path)

    raise ValueError(
        f"Unsupported source: {source}. Use one of: 'mt5', 'synthetic', 'csv'."
    )