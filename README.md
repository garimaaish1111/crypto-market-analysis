# Cryptocurrency Market Analysis System

A data-science dashboard for analysing digital-asset markets. It ingests crypto
price/volume history, market sentiment, on-chain activity, and traditional-asset
prices, then applies analytics, visualisation, and time-series modelling to
deliver **volatility & risk assessment**, **market-cycle phase identification**,
and **digital-vs-traditional correlation analysis** — packaged in an interactive
Streamlit dashboard for data-driven investment decisions.

---

## Features

| Area | What it does |
|------|--------------|
| **Volatility & risk** | Annualised and rolling volatility, historical VaR / CVaR, max drawdown, Sharpe and Sortino, volatility-regime classification, and a GARCH(1,1) model of volatility clustering |
| **Market cycles** | Accumulation → Markup → Distribution → Markdown phase identification from moving averages, RSI momentum, and drawdown |
| **Correlation** | Static correlation matrix + rolling correlation and beta between crypto and equities, gold, the dollar index, rates, and oil |
| **Forecasting** | ARIMA on log price, validated walk-forward one step ahead against a random-walk baseline and scored for skill, with an 80% interval |
| **Sentiment & on-chain** | Crypto Fear & Greed index and Bitcoin transaction / active-address activity |
| **Direction model** | A next-day direction classifier over technical, sentiment, on-chain and macro features, validated on time-ordered splits against the majority-class base rate, with a feature-block ablation |
| **Denomination** | Indian rupees throughout, with Indian digit grouping. Crypto is fetched natively in INR; dollar-quoted macro assets are converted, index levels and rates are not |
| **Resilience** | Live APIs, an on-disk cache shipped pre-populated with real responses, and a transparent generated fallback — the system runs with no credentials and no network |

## Data sources (all free tier)

- **CoinGecko** — crypto price, volume, market cap (no API key required)
- **Yahoo Finance** (`yfinance`) — S&P 500, gold, US dollar index, 10Y yield, crude oil
- **alternative.me** — crypto Fear & Greed index (sentiment signal)
- **Blockchain.info** — Bitcoin on-chain transaction & address metrics

If any feed is unreachable (offline, rate-limited), the system falls back to a
deterministic synthetic dataset so the dashboard always runs. The sidebar shows
which feeds are **live** vs **sample**.

---

## For the evaluator — running this from the submitted folder

**Nothing needs to be configured, and no API key is required.**

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` and loads in well under a second.

`data/cache/` ships pre-populated with **real API responses**, captured at
submission time from all five feeds — CoinGecko, Yahoo Finance, alternative.me,
Blockchain.info, and the USD/INR rate. All four history windows (90/180/270/365
days) are cached, so the sidebar works without a single network call. The
Datasets tab reports the exact date range each feed covers, so it is always
clear what the figures are as of.

Every number therefore comes from genuine market data, and the project runs
identically on a machine with no internet connection.

### Fetching fresh data instead (optional)

The cached responses are a snapshot. To pull current prices:

1. Create a free CoinGecko demo key at
   https://www.coingecko.com/en/developers/dashboard (no payment details).
2. Save it as `apikey.txt` in this folder, the key on a single line.
3. Delete `data/cache/*.pkl`, or press **Refresh data** in the sidebar.

Without a key the public tier rate-limits six consecutive requests and a cold
fetch takes over a minute; with one it takes about six seconds. This is why the
cache ships warm.

### Verifying the test suite

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

205 tests, covering the analysis layer at 92–97%. They run entirely offline.

---

## Quick start

**Windows:** double-click `run.bat`. **Mac / Linux:** `./run.sh`.

The first run creates a virtual environment and installs dependencies, which
takes a minute. Every run after that starts immediately.

### Where the environment lives, and why it matters

The launcher creates the environment **one level above the project folder**, not
inside it:

```
crypto\
├── .venv\                    <- created once, never replaced
└── crypto-market-analysis\   <- replace this folder freely
    ├── run.bat
    ├── app.py
    └── ...
```

A virtual environment stored *inside* the project is thrown away every time a
new version of the project is extracted, so every dependency gets reinstalled
for no reason. Keeping it in the parent folder means you can delete and replace
the entire project folder as often as you like and the environment is still
there.

So: **extract new versions next to `.venv`, not on top of it.**

The launcher looks for an environment in this order:

1. `%CRYPTO_VENV%` — set this environment variable to point anywhere you like
2. `..\.venv` — the shared location above, used by default
3. `.venv` — inside the project, supported for existing setups

It also checks that the core packages import cleanly on every start, and tops
them up if a requirement has been added. That check costs about a second and
means a new dependency never fails at import time.

### Manual setup

If you would rather not use the launcher:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # Mac / Linux
pip install -r requirements.txt
streamlit run app.py
```

Remember that `activate` applies to one terminal session. Open a new terminal
and you need to activate again before `streamlit run app.py` will work.

### Optional: CoinGecko key

Live crypto fetching is **on** by default (`CRYPTO_MODE = "live"` in
`config.py`). The anonymous free tier rate limits below the six requests this
dashboard needs, so create a free demo key at
https://www.coingecko.com/en/developers/dashboard and save it in a file called
`apikey.txt` in the project root. The launcher picks it up automatically, and
`apikey.txt` is gitignored.

Without a key some coins will fall back to generated data and the dashboard will
name which ones. To take the network out of the picture entirely — for offline
demos or for the test suite — set `CRYPTO_MODE = "simulated"` (or export it as an
environment variable, which the test suite and CI both do).

---

## Project structure

```
crypto-market-analysis/
├── app.py                     # Streamlit dashboard (entry point)
├── config.py                  # assets, endpoints, analysis parameters
├── requirements.txt
├── README.md
├── notebooks/
│   └── exploration.ipynb      # methodology walkthrough / reproducible analysis
└── src/
    ├── pipeline.py            # assembles all feeds into one MarketData bundle
    ├── data/
    │   ├── crypto.py          # CoinGecko price/volume ingestion
    │   ├── macro.py           # Yahoo Finance traditional assets
    │   ├── sentiment.py       # Fear & Greed + on-chain metrics
    │   ├── sample_data.py     # deterministic offline fallback
    │   └── cache.py           # on-disk TTL cache
    ├── analysis/
    │   ├── volatility.py      # risk metrics
    │   ├── cycles.py          # market-cycle phase logic
    │   ├── correlation.py     # cross-asset correlation & beta
    │   └── forecast.py        # ARIMA forecasting
    └── viz/
        └── charts.py          # Plotly chart builders
```

The design principle is a clean separation: **data → analysis → visualisation → app**.
Each analysis function is pure and testable on its own, so the dashboard file only
handles layout.

---

## Methodology notes

- **Market-cycle phases** are rule-based and fully explainable (no black box), which
  matters for a tool an investor is meant to trust. Rules combine trend (price vs
  50/200-day MAs), momentum (RSI-14), and drawdown from the peak.
- **Forecasting** models the **log-price** with ARIMA. Order is chosen by AIC on a
  training split; error is reported on a held-out test period *before* refitting on
  the full history to project ahead. Because crypto is close to a random walk, the
  forecast interval is presented as honest uncertainty rather than a price target.
- **Correlations** are shown both statically and as a **rolling** series, because
  crypto-vs-traditional correlation is regime-dependent and shifts over time.

---

## Extending it

- Add coins/macro assets by editing the dictionaries in `config.py`.
- Swap ARIMA for Prophet or an LSTM by adding a module under `src/analysis/` that
  returns the same `ForecastResult` shape — the dashboard needs no other changes.
- Deployable to Streamlit Community Cloud, a container, or IBM Cloud Code Engine
  (it's a standard Python app with no proprietary dependencies).

---

*Built for educational and analytical purposes. Not financial advice.*
