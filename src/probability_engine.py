from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from scipy.stats import norm


def first_touch_outcomes(
    paths: np.ndarray,
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
    direction: str = "long",
) -> Dict[str, float]:
    """
    Calculate pathwise TP/SL probabilities.

    paths should have shape:
        (n_paths, horizon + 1)

    The first column is the current price.
    """

    if direction not in {"long", "short"}:
        raise ValueError("direction must be 'long' or 'short'.")

    if take_profit is None and stop_loss is None:
        raise ValueError("At least one of take_profit or stop_loss must be provided.")

    n_paths = paths.shape[0]

    tp_first = 0
    sl_first = 0
    neither = 0

    for path in paths:
        future = path[1:]

        if direction == "long":
            tp_hits = np.where(future >= take_profit)[0] if take_profit is not None else np.array([])
            sl_hits = np.where(future <= stop_loss)[0] if stop_loss is not None else np.array([])
        else:
            tp_hits = np.where(future <= take_profit)[0] if take_profit is not None else np.array([])
            sl_hits = np.where(future >= stop_loss)[0] if stop_loss is not None else np.array([])

        first_tp = tp_hits[0] if len(tp_hits) > 0 else None
        first_sl = sl_hits[0] if len(sl_hits) > 0 else None

        if first_tp is None and first_sl is None:
            neither += 1
        elif first_tp is not None and first_sl is None:
            tp_first += 1
        elif first_tp is None and first_sl is not None:
            sl_first += 1
        elif first_tp < first_sl:
            tp_first += 1
        elif first_sl < first_tp:
            sl_first += 1
        else:
            neither += 1

    return {
        "tp_first_prob": tp_first / n_paths,
        "sl_first_prob": sl_first / n_paths,
        "neither_prob": neither / n_paths,
        "tp_first_count": tp_first,
        "sl_first_count": sl_first,
        "neither_count": neither,
        "n_paths": n_paths,
    }


def expected_range(
    paths: np.ndarray,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> Dict[str, float]:
    """
    Calculate terminal expected range from simulated paths.

    Default gives the 5%-95% terminal range.
    """

    terminal_prices = paths[:, -1]

    return {
        "terminal_mean": float(np.mean(terminal_prices)),
        "terminal_median": float(np.median(terminal_prices)),
        "terminal_lower": float(np.quantile(terminal_prices, lower_q)),
        "terminal_upper": float(np.quantile(terminal_prices, upper_q)),
        "lower_q": lower_q,
        "upper_q": upper_q,
    }


def terminal_direction_probability(
    paths: np.ndarray,
    current_price: Optional[float] = None,
) -> Dict[str, float]:
    """
    Estimate probability that terminal price finishes above/below current price.
    """

    if current_price is None:
        current_price = float(paths[0, 0])

    terminal_prices = paths[:, -1]

    prob_up = float(np.mean(terminal_prices > current_price))
    prob_down = float(np.mean(terminal_prices < current_price))
    prob_flat = float(np.mean(terminal_prices == current_price))

    return {
        "prob_up": prob_up,
        "prob_down": prob_down,
        "prob_flat": prob_flat,
    }

def gbm_terminal_distribution_params(
    current_price: float,
    mu: float,
    sigma: float,
    horizon: int,
) -> Dict[str, float]:
    """
    Analytical GBM terminal log-price distribution.

    Under GBM:

        log(S_T / S_0) ~ N((mu - 0.5 sigma^2)T, sigma^2 T)

    Here horizon is measured in candle steps, because mu and sigma are
    estimated per candle return.
    """

    mean_log_return = (mu - 0.5 * sigma**2) * horizon
    std_log_return = sigma * np.sqrt(horizon)

    terminal_median = current_price * np.exp(mean_log_return)
    terminal_mean = current_price * np.exp(mu * horizon)

    return {
        "mean_log_return": float(mean_log_return),
        "std_log_return": float(std_log_return),
        "terminal_median": float(terminal_median),
        "terminal_mean": float(terminal_mean),
    }


def gbm_terminal_probability_above(
    current_price: float,
    level: float,
    mu: float,
    sigma: float,
    horizon: int,
) -> float:
    """
    Analytical probability that GBM terminal price finishes above a level.
    """

    if sigma <= 0:
        return float(current_price > level)

    mean_log_return = (mu - 0.5 * sigma**2) * horizon
    std_log_return = sigma * np.sqrt(horizon)

    z = (np.log(level / current_price) - mean_log_return) / std_log_return

    return float(1 - norm.cdf(z))


def gbm_terminal_probability_below(
    current_price: float,
    level: float,
    mu: float,
    sigma: float,
    horizon: int,
) -> float:
    """
    Analytical probability that GBM terminal price finishes below a level.
    """

    if sigma <= 0:
        return float(current_price < level)

    mean_log_return = (mu - 0.5 * sigma**2) * horizon
    std_log_return = sigma * np.sqrt(horizon)

    z = (np.log(level / current_price) - mean_log_return) / std_log_return

    return float(norm.cdf(z))


def gbm_terminal_range(
    current_price: float,
    mu: float,
    sigma: float,
    horizon: int,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> Dict[str, float]:
    """
    Analytical GBM terminal confidence range.

    Default gives the 5%-95% terminal price range.
    """

    mean_log_return = (mu - 0.5 * sigma**2) * horizon
    std_log_return = sigma * np.sqrt(horizon)

    lower_price = current_price * np.exp(mean_log_return + std_log_return * norm.ppf(lower_q))
    upper_price = current_price * np.exp(mean_log_return + std_log_return * norm.ppf(upper_q))

    return {
        "gbm_lower": float(lower_price),
        "gbm_upper": float(upper_price),
        "lower_q": lower_q,
        "upper_q": upper_q,
    }