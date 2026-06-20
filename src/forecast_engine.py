from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _safe_float(value: float, fallback: float = 0.0) -> float:
    value = float(value)
    if not np.isfinite(value):
        return fallback
    return value


def build_market_state_features(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Build rolling market-state features from OHLC data.

    These features describe the local market regime around each candle.
    They are later used to find historical periods that looked similar
    to the current market state.
    """
    if "Close" not in df.columns:
        raise ValueError("df must contain a 'Close' column.")

    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype(float)
    log_returns = np.log(close).diff()

    out["return"] = log_returns
    out["volatility"] = log_returns.rolling(lookback).std()
    out["mean_return"] = log_returns.rolling(lookback).mean()

    out["trend_score"] = (
        out["mean_return"].abs() / out["volatility"].replace(0.0, np.nan)
    )

    out["latest_return_z"] = (
        log_returns / out["volatility"].replace(0.0, np.nan)
    )

    if {"High", "Low"}.issubset(df.columns):
        candle_range = df["High"].astype(float) - df["Low"].astype(float)
        out["range_ratio"] = (
            candle_range / candle_range.rolling(lookback).mean().replace(0.0, np.nan)
        )
    else:
        out["range_ratio"] = 1.0

    return out.replace([np.inf, -np.inf], np.nan)


def find_similar_market_states(
    df: pd.DataFrame,
    lookback: int = 20,
    horizon: int = 20,
    max_matches: int = 250,
    min_matches: int = 30,
) -> Tuple[pd.Index, Dict[str, float]]:
    """
    Find historical candles whose market-state features are closest to now.

    The method:
    1. Calculates rolling regime features.
    2. Takes the latest market state as the target.
    3. Compares past states using standardised Euclidean distance.
    4. Returns the closest historical examples that have enough forward data.

    This creates a conditional historical sample:
    'What happened after past markets that looked similar to now?'
    """
    features = build_market_state_features(df=df, lookback=lookback)

    feature_cols = [
        "volatility",
        "mean_return",
        "trend_score",
        "latest_return_z",
        "range_ratio",
    ]

    usable = features[feature_cols].dropna()

    if len(usable) < min_matches + horizon + 1:
        raise ValueError(
            "Not enough historical data to find similar market states. "
            "Use more candles, reduce lookback, or reduce horizon."
        )

    target_time = usable.index[-1]
    target = usable.loc[target_time]

    # Exclude the latest region because those candles do not have future data.
    candidate_features = usable.iloc[: -horizon]

    means = candidate_features.mean()
    stds = candidate_features.std(ddof=1).replace(0.0, np.nan)

    z_candidates = (candidate_features - means) / stds
    z_target = (target - means) / stds

    distances = np.sqrt(((z_candidates - z_target) ** 2).sum(axis=1))
    distances = distances.replace([np.inf, -np.inf], np.nan).dropna()

    if len(distances) < min_matches:
        raise ValueError(
            "Not enough valid candidate states after distance calculation."
        )

    n_matches = min(max_matches, len(distances))
    selected_index = distances.nsmallest(n_matches).index

    info = {
        "n_candidate_states": float(len(distances)),
        "n_matched_states": float(len(selected_index)),
        "mean_similarity_distance": _safe_float(distances.loc[selected_index].mean()),
        "median_similarity_distance": _safe_float(distances.loc[selected_index].median()),
        "lookback": float(lookback),
        "horizon": float(horizon),
    }

    return selected_index, info


def simulate_regime_conditioned_paths(
    df: pd.DataFrame,
    horizon: int = 20,
    n_paths: int = 1000,
    lookback: int = 20,
    max_matches: int = 250,
    min_matches: int = 30,
    seed: int | None = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Simulate price paths using historical forward returns after similar regimes.

    This is not an unconditional random bootstrap.

    Instead, it:
    - detects the current market state,
    - finds past market states that looked similar,
    - samples the forward return sequences that followed those states,
    - applies those return sequences to the current price.

    Output:
    - paths: shape (n_paths, horizon + 1)
    - info: diagnostics about the matched historical regimes
    """
    if "Close" not in df.columns:
        raise ValueError("df must contain a 'Close' column.")

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    close = df["Close"].astype(float)
    log_returns = np.log(close).diff()

    matched_index, info = find_similar_market_states(
        df=df,
        lookback=lookback,
        horizon=horizon,
        max_matches=max_matches,
        min_matches=min_matches,
    )

    rng = np.random.default_rng(seed)
    sampled_states = rng.choice(matched_index.to_numpy(), size=n_paths, replace=True)

    current_price = float(close.iloc[-1])
    paths = np.empty((n_paths, horizon + 1), dtype=float)
    paths[:, 0] = current_price

    forward_return_matrix = []

    index_positions = {idx: pos for pos, idx in enumerate(df.index)}

    for path_id, state_idx in enumerate(sampled_states):
        start_pos = index_positions[state_idx]

        # Forward returns after the matched state.
        # If matched state is t, use returns from t+1 to t+horizon.
        forward_returns = log_returns.iloc[start_pos + 1 : start_pos + horizon + 1]

        if len(forward_returns) < horizon:
            raise ValueError("Matched state did not have enough forward returns.")

        forward_returns = forward_returns.to_numpy(dtype=float)
        forward_return_matrix.append(forward_returns)

        cumulative_returns = np.cumsum(forward_returns)
        paths[path_id, 1:] = current_price * np.exp(cumulative_returns)

    forward_return_matrix = np.asarray(forward_return_matrix)

    terminal_prices = paths[:, -1]

    info.update(
        {
            "forecast_type": "Regime-Conditioned Forecast",
            "current_price": _safe_float(current_price),
            "expected_terminal_price": _safe_float(np.mean(terminal_prices)),
            "median_terminal_price": _safe_float(np.median(terminal_prices)),
            "mean_forward_return": _safe_float(np.mean(forward_return_matrix)),
            "median_forward_return": _safe_float(np.median(forward_return_matrix)),
            "forward_return_volatility": _safe_float(np.std(forward_return_matrix, ddof=1)),
        }
    )

    return paths, info