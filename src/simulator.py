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

def estimate_jump_parameters(
    historical_returns,
    window: int = 90,
    threshold_sigma: float = 2.5,
) -> dict:
    """
    Estimate simple threshold-based jump parameters from recent log returns.

    A jump is defined as:

        |r_t - mean(r)| > threshold_sigma * std(r)

    This is an experimental research estimate used by the jump-diffusion model.
    """

    recent_returns = historical_returns.dropna().tail(window)

    if len(recent_returns) < 2:
        return {
            "jump_intensity": 0.0,
            "jump_mean": 0.0,
            "jump_std": 0.0,
            "n_jumps": 0,
            "threshold_sigma": threshold_sigma,
        }

    mean_return = float(recent_returns.mean())
    std_return = float(recent_returns.std(ddof=1))

    if std_return <= 0:
        return {
            "jump_intensity": 0.0,
            "jump_mean": 0.0,
            "jump_std": 0.0,
            "n_jumps": 0,
            "threshold_sigma": threshold_sigma,
        }

    jump_mask = (recent_returns - mean_return).abs() > threshold_sigma * std_return
    jump_returns = recent_returns[jump_mask]

    return {
        "jump_intensity": float(jump_mask.mean()),
        "jump_mean": float(jump_returns.mean()) if len(jump_returns) else 0.0,
        "jump_std": float(jump_returns.std(ddof=1)) if len(jump_returns) > 1 else 0.0,
        "n_jumps": int(jump_mask.sum()),
        "threshold_sigma": threshold_sigma,
    }


def simulate_jump_diffusion_paths(
    current_price: float,
    mu: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    horizon: int,
    n_paths: int,
    seed: int | None = None,
):
    """
    Simulate jump-diffusion price paths.

    Return step:

        r_t = (mu - 0.5 sigma^2) + sigma Z_t + J_t N_t

    where:

        Z_t is a standard normal shock
        N_t is a Bernoulli jump indicator
        J_t is a random jump size
    """

    rng = np.random.default_rng(seed)

    normal_shocks = rng.normal(0.0, 1.0, size=(n_paths, horizon))
    gbm_steps = (mu - 0.5 * sigma**2) + sigma * normal_shocks

    jump_intensity = float(np.clip(jump_intensity, 0.0, 1.0))

    jump_occurs = rng.binomial(
        n=1,
        p=jump_intensity,
        size=(n_paths, horizon),
    )

    if jump_std > 0:
        jump_sizes = rng.normal(
            loc=jump_mean,
            scale=jump_std,
            size=(n_paths, horizon),
        )
    else:
        jump_sizes = np.full((n_paths, horizon), jump_mean)

    log_steps = gbm_steps + jump_occurs * jump_sizes
    log_paths = np.cumsum(log_steps, axis=1)

    future_prices = current_price * np.exp(log_paths)

    return np.column_stack([np.full(n_paths, current_price), future_prices])