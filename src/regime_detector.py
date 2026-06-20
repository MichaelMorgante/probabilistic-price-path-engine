from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _safe_float(value: float, fallback: float = 0.0) -> float:
    """
    Convert a numeric value to a finite float.
    Used so dashboard metrics do not break when volatility is tiny or missing.
    """
    value = float(value)
    if not np.isfinite(value):
        return fallback
    return value


def calculate_regime_features(
    df: pd.DataFrame,
    window: int = 90,
    short_window: int = 20,
    jump_threshold_sigma: float = 2.5,
) -> Dict[str, float]:
    """
    Calculate simple market-regime features from recent close prices.

    These features are designed for model selection, not trade entry signals.

    Main features:
    - vol_short: recent short-window volatility
    - vol_long: longer-window volatility
    - vol_ratio: short volatility divided by long volatility
    - latest_return_z: latest return measured in standard deviations
    - jump_intensity: fraction of recent returns beyond jump_threshold_sigma
    - trend_score: absolute mean return divided by volatility
    - range_expansion: latest candle range divided by recent average range
    """
    if "Close" not in df.columns:
        raise ValueError("df must contain a 'Close' column.")

    close = df["Close"].astype(float)
    log_returns = np.log(close).diff().dropna()

    if len(log_returns) < max(10, short_window):
        raise ValueError("Not enough returns to calculate regime features.")

    recent_returns = log_returns.tail(window)
    short_returns = log_returns.tail(short_window)

    vol_short = _safe_float(short_returns.std(ddof=1))
    vol_long = _safe_float(recent_returns.std(ddof=1))

    if vol_long <= 0:
        vol_ratio = 1.0
    else:
        vol_ratio = vol_short / vol_long

    latest_return = _safe_float(log_returns.iloc[-1])

    if vol_long <= 0:
        latest_return_z = 0.0
    else:
        latest_return_z = latest_return / vol_long

    mean_return = _safe_float(recent_returns.mean())

    if vol_long <= 0:
        trend_score = 0.0
    else:
        trend_score = abs(mean_return) / vol_long

    jump_mask = (recent_returns - mean_return).abs() > jump_threshold_sigma * vol_long
    jump_intensity = _safe_float(jump_mask.mean())

    range_expansion = 1.0
    if {"High", "Low"}.issubset(df.columns):
        candle_range = (df["High"].astype(float) - df["Low"].astype(float)).dropna()
        recent_range = candle_range.tail(window)
        if len(recent_range) >= 10:
            avg_range = _safe_float(recent_range.mean())
            latest_range = _safe_float(candle_range.iloc[-1])
            if avg_range > 0:
                range_expansion = latest_range / avg_range

    return {
        "vol_short": _safe_float(vol_short),
        "vol_long": _safe_float(vol_long),
        "vol_ratio": _safe_float(vol_ratio, fallback=1.0),
        "latest_return": _safe_float(latest_return),
        "latest_return_z": _safe_float(latest_return_z),
        "jump_intensity": _safe_float(jump_intensity),
        "trend_score": _safe_float(trend_score),
        "range_expansion": _safe_float(range_expansion, fallback=1.0),
        "window": float(window),
        "short_window": float(short_window),
        "jump_threshold_sigma": float(jump_threshold_sigma),
    }


def classify_market_regime(features: Dict[str, float]) -> Dict[str, str]:
    """
    Classify the current market regime and select the most appropriate model.

    Rule logic:
    - Jump-risk / shock behaviour -> Jump-Diffusion
    - High-volatility or range-expansion behaviour -> Bootstrap
    - Calm and relatively normal behaviour -> GBM
    - Unclear mixed behaviour -> Bootstrap
    """
    vol_ratio = features["vol_ratio"]
    latest_return_z = abs(features["latest_return_z"])
    jump_intensity = features["jump_intensity"]
    trend_score = features["trend_score"]
    range_expansion = features["range_expansion"]

    if latest_return_z >= 2.5 or jump_intensity >= 0.03:
        return {
            "regime_label": "Jump-risk / shock regime",
            "selected_model": "Jump-Diffusion (Experimental)",
            "reason": (
                "Selected Jump-Diffusion because recent returns show jump-risk "
                "or the latest return is unusually large versus recent volatility."
            ),
        }

    if vol_ratio >= 1.4 or range_expansion >= 1.5:
        return {
            "regime_label": "High-volatility empirical regime",
            "selected_model": "Bootstrap",
            "reason": (
                "Selected Bootstrap because short-term volatility or candle range "
                "has expanded relative to the recent baseline."
            ),
        }

    if vol_ratio <= 1.15 and trend_score < 0.20 and latest_return_z < 1.5:
        return {
            "regime_label": "Calm diffusion-like regime",
            "selected_model": "GBM",
            "reason": (
                "Selected GBM because recent volatility is stable and there is "
                "no strong jump or range-expansion signal."
            ),
        }

    return {
        "regime_label": "Mixed / uncertain regime",
        "selected_model": "Bootstrap",
        "reason": (
            "Selected Bootstrap as the conservative default because the current "
            "market state is mixed rather than clearly calm or jump-driven."
        ),
    }


def detect_market_regime(
    df: pd.DataFrame,
    window: int = 90,
    short_window: int = 20,
    jump_threshold_sigma: float = 2.5,
) -> Dict[str, float | str]:
    """
    Full regime detector used by the dashboard.

    Returns both:
    - numeric regime features
    - regime label
    - selected simulation model
    - human-readable reason
    """
    features = calculate_regime_features(
        df=df,
        window=window,
        short_window=short_window,
        jump_threshold_sigma=jump_threshold_sigma,
    )
    classification = classify_market_regime(features)

    return {
        **features,
        **classification,
    }