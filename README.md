# Probabilistic Price Path Engine

A Python research and dashboard project for live intraday price-path simulation, TP/SL probability estimation, and predictive market-state analysis using MT5 market data.

## Phase 1 Objective

The first version of this project converts a working notebook prototype into a clean Python project structure.

The engine will:

- fetch live or historical price data from MetaTrader 5
- estimate recent drift and volatility
- simulate future price paths using probabilistic methods
- estimate TP/SL hit probabilities
- visualise simulated paths, expected ranges, and trade levels
- provide a Streamlit dashboard for live monitoring

## Project Structure

```text
probabilistic-price-path-engine/
├── app/
│   └── streamlit_app.py
├── notebooks/
├── src/
│   ├── mt5_loader.py
│   ├── simulator.py
│   ├── probability_engine.py
│   ├── charts.py
│   └── utils.py
├── data/
├── reports/
├── live_artifacts/
├── requirements.txt
└── README.md
