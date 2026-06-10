from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import json
import pandas as pd


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def format_price(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def ensure_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def append_probability_log(
    metrics: Dict[str, float],
    symbol: str,
    timeframe: str,
    model_name: str,
    output_path: str | Path = "reports/logs/probability_log.csv",
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": pd.Timestamp.now(),
        "symbol": symbol,
        "timeframe": timeframe,
        "model_name": model_name,
        **metrics,
    }

    if extra:
        row.update(extra)

    df_new = pd.DataFrame([row])

    if output_path.exists():
        df_old = pd.read_csv(output_path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(output_path, index=False)

    return output_path


def save_json(
    payload: Dict,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, default=str)

    return output_path