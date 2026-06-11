from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def make_price_path_figure(
    df: pd.DataFrame,
    paths: np.ndarray,
    metrics: Dict[str, float],
    max_paths: int = 120,
) -> go.Figure:
    recent = df.tail(140).copy().reset_index(drop=True)
    n_recent = len(recent)
    horizon = paths.shape[1] - 1

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=list(range(n_recent)),
        open=recent["Open"],
        high=recent["High"],
        low=recent["Low"],
        close=recent["Close"],
        name="Recent candles",
        increasing_line_color="#43d18d",
        decreasing_line_color="#ff5b5b",
        showlegend=False,
    ))

    x_future = list(range(n_recent - 1, n_recent + horizon))

    rng = np.random.default_rng(7)
    idx = rng.choice(paths.shape[0], size=min(max_paths, paths.shape[0]), replace=False)

    for i in idx:
        end_price = paths[i, -1]
        line_color = (
            "rgba(67, 209, 141, 0.14)"
            if end_price >= metrics["entry_price"]
            else "rgba(255, 91, 91, 0.14)"
        )

        fig.add_trace(go.Scatter(
            x=x_future,
            y=paths[i],
            mode="lines",
            line=dict(width=1, color=line_color),
            hoverinfo="skip",
            showlegend=False,
        ))

    percentiles = np.percentile(paths, [5, 20, 50, 80, 95], axis=0)
    labels = ["5%", "20%", "50%", "80%", "95%"]
    colors = ["#ff5b5b", "#c65a68", "#6da8ff", "#43d18d", "#43d18d"]

    for pct, label, color in zip(percentiles, labels, colors):
        fig.add_trace(go.Scatter(
            x=x_future,
            y=pct,
            mode="lines",
            line=dict(width=1.5, dash="dot", color=color),
            name=label,
        ))

    fig.add_trace(go.Scatter(
        x=x_future + x_future[::-1],
        y=list(percentiles[4]) + list(percentiles[0][::-1]),
        fill="toself",
        fillcolor="rgba(67, 209, 141, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="5%-95% cone",
    ))

    fig.add_hline(
        y=metrics["tp_level"],
        line_dash="dash",
        line_color="#43d18d",
        annotation_text=f'TP {metrics["tp_level"]:,.2f}',
        annotation_position="right",
    )

    fig.add_hline(
        y=metrics["sl_level"],
        line_dash="dash",
        line_color="#ff5b5b",
        annotation_text=f'SL {metrics["sl_level"]:,.2f}',
        annotation_position="right",
    )

    fig.add_vline(
        x=n_recent - 1,
        line_dash="dash",
        line_color="rgba(255,255,255,0.45)",
        annotation_text="current candle close",
    )

    fig.update_layout(
        template="plotly_dark",
        height=650,
        margin=dict(l=20, r=20, t=35, b=25),
        paper_bgcolor="#071018",
        plot_bgcolor="#071018",
        font=dict(color="#d6dde8"),
        xaxis=dict(
            title="Candles",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            title="Price",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            side="right",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig