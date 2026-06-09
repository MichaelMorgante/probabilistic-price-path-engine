from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def compute_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def estimate_mu_sigma(
    close: pd.Series,
    window: int = 80,
    drift_mode: str = "zero",
) -> Tuple[float, float, pd.Series]:
    log_ret = compute_log_returns(close).dropna()
    recent = log_ret.tail(window)

    if len(recent) < 10:
        raise ValueError("Not enough returns to estimate volatility.")

    sigma = float(recent.std(ddof=1))

    if drift_mode == "historical":
        mu = float(recent.mean())
    elif drift_mode == "zero":
        mu = 0.0
    else:
        raise ValueError("drift_mode must be 'zero' or 'historical'.")

    return mu, sigma, log_ret


def simulate_gbm_paths(
    current_price: float,
    mu: float,
    sigma: float,
    horizon: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 1, size=(n_paths, horizon))
    log_steps = (mu - 0.5 * sigma**2) + sigma * eps
    log_paths = np.cumsum(log_steps, axis=1)
    prices = current_price * np.exp(log_paths)
    return np.column_stack([np.full(n_paths, current_price), prices])


def simulate_bootstrap_paths(
    current_price: float,
    historical_returns: pd.Series,
    horizon: int,
    n_paths: int,
    sample_window: int = 120,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = historical_returns.dropna().tail(sample_window).to_numpy()

    if len(returns) < 10:
        raise ValueError("Not enough historical returns for bootstrap.")

    sampled = rng.choice(returns, size=(n_paths, horizon), replace=True)
    log_paths = np.cumsum(sampled, axis=1)
    prices = current_price * np.exp(log_paths)
    return np.column_stack([np.full(n_paths, current_price), prices])