# What changed and why

Every change below fixes a defect that was in the original code, or closes a gap
between what the SRS promised and what the code did. Nothing here is cosmetic
except the colour scheme.

Files replaced: `config.py`, `app.py`, `src/data/cache.py`,
`src/data/sample_data.py`, `src/analysis/volatility.py`,
`src/analysis/cycles.py`, `src/analysis/correlation.py`,
`src/analysis/forecast.py`, `src/viz/charts.py`. New: `.streamlit/config.toml`.

Files unchanged: `src/pipeline.py`, `src/data/crypto.py`, `src/data/macro.py`,
`src/data/sentiment.py`, `requirements.txt`.

---

## 1. The Market Cycles tab produced nothing below 365 days

**Was:** a 200-day moving average needs 200 observations before it yields its
first value. On the 90-day and 180-day sidebar settings there were zero valid
values, so every day was labelled "Undetermined" and the *Price vs 200-day MA*
metric rendered as `nan%`. Even at the full 365 days only 45% of the window
carried a label.

**Now:** `config.ma_windows(n)` keeps the canonical 50/200 pair when there are at
least 300 observations and otherwise scales both windows down, preserving the 1:4
ratio. The windows actually in use are shown in the chart legend, the metric
labels, and the rule table, and an info panel appears whenever they differ from
the standard pair.

| Window | MA pair | Days labelled |
|---|---|---|
| 90 | 7 / 30 | 68% |
| 180 | 15 / 60 | 67% |
| 270 | 22 / 90 | 67% |
| 365 | 50 / 200 | 45% |

Shorter averages react faster and turn phases over more often, which is why the
info panel says so rather than hiding the substitution.

## 2. The "deterministic" sample data was not deterministic

**Was:** `seed = abs(hash(symbol)) % (2**32)`. Python randomises string hashing
per process, so Bitcoin's synthetic price history was different on every restart.
The module docstring, the SRS, and Section 5.2 of the report all claimed
reproducibility.

**Now:** seeds come from `zlib.crc32`, which is stable across processes. Verified
identical across three separate interpreter runs.

## 3. Sample data had no correlation structure, which silently invalidated the correlation tab

**Was:** every asset was an independent GBM. Offline, the correlation heatmap
showed approximately zero everywhere — not a finding, just sampling noise. Any
"Results and Observations" written from a sample-data run described randomness.

**Now:** all series are driven by a shared risk-on factor plus idiosyncratic
noise. Each asset has a beta onto that factor, and the crypto factor correlates
with the equity factor at 0.35, roughly the level observed since 2020. Offline
output now looks like this:

```
     S&P 500  Gold  US Dollar Index  US 10Y Yield  Crude Oil
BTC     0.37 -0.12            -0.07          0.09       0.19
ETH     0.35 -0.16            -0.06          0.09       0.13
```

Gold near zero, the dollar index negative, oil mildly positive — the qualitative
shape you would expect. This is simulated structure, not measured structure, and
a banner says so whenever any feed is on sample data.

**Also fixed:** sample macro prices are now generated on the full daily grid and
then filtered to business days, matching what Yahoo Finance actually returns.
Previously sample mode produced 365 macro rows against live mode's ~252, so the
two modes exercised different code paths through the correlation alignment.

## 4. The refresh button did not clear the cache it claimed to clear

**Was:** `st.cache_data.clear()` clears Streamlit's in-memory cache only. The
pickle files under `data/cache/` were untouched, so a failed fetch kept serving
its sample-data fallback for the full six-hour TTL and the button could not
dislodge it. SRS requirement FR-1.6 was not met.

**Now:** `cache.clear_all()` unlinks the pickle files and reports how many were
removed; the button calls it before clearing Streamlit's cache. `cache.entry_count()`
shows the current cache size on the Data & Sources tab.

## 5. The forecast cache returned results from the wrong history window

**Was:** `get_forecast(_price, symbol, horizon)`. The underscore prefix tells
Streamlit not to hash that argument, so the cache key was only `(symbol, horizon)`.
Changing the history window from 365 to 270 days returned the 365-day forecast.

**Now:** `days` is passed as an explicit fourth argument and forms part of the key.

## 6. The out-of-sample test was not measuring model quality

**Was:** the model was fitted on the training split and asked for one forecast
covering the entire ~55-day test period. For a near-random-walk series that
forecast is almost flat, so the RMSE measured how far the price drifted over two
months, not how good the model is. There was also no baseline, which left "RMSE
$4,200" uninterpretable.

**Now:**

- **Walk-forward validation.** Predictions are made one day ahead at a time.
  Each actual observation is appended to the model state (`refit=False`, so
  parameters stay fixed at their training values) before the next prediction.
  This is what a person checking the dashboard daily would actually experience.
- **A naive baseline.** The random-walk rule — tomorrow's price is today's price
  — is scored on the same days.
- **A skill score.** `1 − RMSE(model) / RMSE(naive)`. Positive means the model
  adds something.
- **A plain-English verdict** and a walk-forward chart showing predicted against
  actual.

Current output on sample data: ARIMA(0,1,1), skill **−0.88%**. The model does not
beat "assume no change." That is the correct and expected result for crypto, and
it now says so in words rather than leaving the reader to infer it.

## 7. Smaller correctness fixes

**RSI returned 50 for maximum strength.** When average loss is zero,
`.replace(0, np.nan)` made RSI `NaN`, and `.fillna(50)` turned an unbroken rally
into neutral momentum. Verified: a strictly rising series returned exactly 50.0.
Now returns 100 for pure gains, 50 only for genuinely flat stretches, and `NaN`
during the warm-up window rather than a fabricated neutral reading.

**Wilder's smoothing needed `adjust=False`.** `ewm(alpha=1/14)` defaults to
`adjust=True`, a different weighting scheme.

**Sortino used the wrong denominator.** `rets[rets < 0].std()` measures spread
about the mean of the losses. Proper downside deviation is
`sqrt(mean(min(r − target, 0)²))` across *all* observations against the risk-free
target. The old version inflated the ratio.

**"Drawdown vs ATH" was mislabelled.** `price.cummax()` is the running maximum
inside the loaded window — a one-year high on a 365-day history, not an all-time
high. Renamed to "Drawdown vs window high" everywhere.

**Yields were treated as prices.** A 10-year yield moving 4.00% → 4.10% was being
recorded as a "+2.5% return". Rate columns listed in `config.YIELD_ASSETS` are now
differenced in percentage points; everything else is percent-changed. Correlation
is scale-invariant, so the two mix without further adjustment.

**`warnings.filterwarnings("ignore")` at module scope** silenced every warning in
the whole process. Now scoped to the statsmodels calls with a context manager.

**`phase_series` was recomputed three times per tab** via a Python row loop with
`.loc` lookups. Now vectorised with `np.select` and computed once.

**Market cap wandered independently of price** in the sample generator, because
the supply multiplier was redrawn on every day. Now a fixed supply per asset.

**On-chain metrics shared one axis** despite differing by roughly 3×, flattening
the smaller series. Now on a secondary axis.

## 8. Phase rules tightened

Two rules fired in situations they were not meant to describe:

- An uptrend with soft momentum was labelled **Distribution** regardless of where
  price sat. Distribution describes supply meeting demand *near a top*, so it now
  also requires price to be within 10% of the peak. Otherwise the day reads as
  Transition.
- Any downtrend that was not both weak and deeply drawn down was labelled
  **Accumulation** — so a shallow, early decline read as a basing bottom.
  Accumulation now requires a drawdown past 25% already in place; shallower
  downtrends read as Markdown.

**This means Section 7.3 of your project report no longer matches the code.** The
updated table is reproduced in the dashboard's "How phases are identified" panel;
copy it across before you submit.

## 9. Colour scheme

Three colours, applied so that colour carries meaning rather than decoration:

| Token | Hex | Role |
|---|---|---|
| Ink | `#212121` | Observed data: price lines, actuals, bearish phase |
| Accent | `#00BCD4` | Wherever the tab wants attention: forecasts, current volatility, bullish phase |
| Muted | `#607D8B` | Context that should recede: moving averages, drawdown fill, distribution phase |

The phase ribbon needs six separable steps, so tints and shades are drawn from
the same three families — `#4DD0E1`, `#0097A7`, `#90A4AE`, `#ECEFF1`, `#CFD8DC`.
All eight values live in `config.py`; change them there and every chart follows.
`.streamlit/config.toml` carries the same scheme into Streamlit's own chrome
(sidebar, tabs, widgets), which Plotly theming does not reach.

---

## Still open

Deliberately not fixed, because each is a scope decision rather than a defect —
but be ready to name them:

- **On-chain data is displayed, not modelled.** Transaction count and active
  addresses appear on the Data & Sources tab and feed nothing. The Bitcoin-only
  limitation comes from the free Blockchain.info charts. Exchange inflows and
  outflows would be the single most valuable addition.
- **Sentiment is a composite index, not NLP.** Fear & Greed stands in for direct
  social-media text analysis, which needs a paid feed.
- **No GARCH.** Volatility is a rolling standard deviation, which reacts slowly
  and weights every day in the window equally. GARCH(1,1) would capture
  volatility clustering properly.
- **365 days is less than one cycle.** A crypto market cycle historically runs
  about four years. The free CoinGecko tier caps history at a year, so the phase
  labels describe the current regime rather than a position within a full cycle.
  The dashboard says this in the phase explainer.
- **No stationarity test.** `d=1` is assumed rather than established by an ADF
  test. Defensible for log prices, but worth stating.

---

# Update — browser launch and palette lock

## 10. The browser stopped opening automatically

The `.streamlit/config.toml` I supplied earlier contained:

```toml
[server]
headless = true
```

`headless = true` tells Streamlit it is running on a server with no display, so
it prints the URL and does not launch a browser. That setting is now removed, and
`run.bat` / `run.sh` are included so the whole sequence — activate the
environment, start the app — is one double-click.

If a browser still does not open on your machine (some Windows setups block the
launch regardless), open `http://localhost:8501` manually. The server is running
correctly either way; only the convenience launch is affected.

## 11. The palette is now closed at five colours

| Colour | Role |
|---|---|
| `#212121` | Observed data — price lines, actuals, bear phase, body text |
| `#00BCD4` | Focus — whatever the tab wants you to look at |
| `#0097A7` | Emphasis within the accent family — forecast line, accumulation |
| `#607D8B` | Context that should recede — drawdown fill, transition phase |
| `#90A4AE` | Tertiary — volume bars, base lines, distribution phase |

`#ECEFF1` and `#CFD8DC` survive in three places only: hairline gridlines, the
neutral midpoint of the correlation scale, and Streamlit's own panel background
(the sidebar and expanders, which need a surface tint to read as panels at all).

Two changes were needed to get there:

**The sixth phase colour is gone.** Six cycle states previously needed six
colours. Days inside the moving-average warm-up are now drawn as the bare price
line with no marker, which reads correctly as "not yet classifiable" rather than
implying a sixth phase. The bar chart still tints that bar, since one bar is a
small place.

**A custom Plotly template replaces `plotly_white`.** Plotly's stock templates
embed their own colourways and per-trace colour scales — the Plasma ramp for
heatmaps, a ten-hue qualitative set for lines — inside every figure they build.
Any trace that did not set a colour explicitly would have pulled from those. The
template is now constructed from scratch in `charts.py` and declares only this
scheme, so nothing outside it can reach the screen even if a future chart forgets
to be explicit.

Verified by dumping all twelve figures to JSON and extracting every hex value:
the only colours present are the five above plus the two tints.

---

# Update — crash, rate limiting, dark theme

## 12. `st.info(..., icon="◈")` crashed the app

`◈` (U+25C8) is a geometric shape, not an emoji, and Streamlit validates the
`icon` argument against a real emoji list. Every `icon=` argument has been
removed — from `st.set_page_config`, the sample-data warning, the moving-average
notice, and the forecast verdict. None of them carried meaning the text did not
already carry.

## 13. CoinGecko 429 — Too Many Requests

Six coins were requested back to back with no spacing. The free tier allows
roughly 5 to 15 calls per minute, so the last two or three reliably failed and
fell back to simulated data while the earlier ones stayed live. That is the worst
possible outcome: a risk table and correlation matrix silently mixing measured
and generated assets.

Three changes:

- **Throttling.** Live calls are spaced `COINGECKO_MIN_INTERVAL` seconds apart
  (2.5 by default). A cold cache now takes about 15 seconds for six coins; cache
  hits never touch the network at all.
- **Backoff.** A 429 is retried up to four times with exponential backoff,
  honouring the server's `Retry-After` header when it sends one.
- **Correct key headers.** Demo and Pro keys use different hosts *and* different
  header names. The old code sent a Pro header to the public host, which is
  silently ignored — so an apparently valid key still got rate limited. Set
  `COINGECKO_API_KEY` and leave `COINGECKO_PLAN` as `demo`; a free demo key takes
  two minutes to create and raises the ceiling substantially.

**Per-coin source reporting.** `load_all_crypto` now returns a symbol-to-source
map alongside the overall label, and the banner names exactly which coins are
simulated rather than reporting an undifferentiated "mixed" and leaving the
reader to guess which rows of the risk table are real.

## 14. Dark theme

The interface was white because light was the default, which under-used a palette
led by charcoal. `config.THEME` now switches the whole interface, and everything
downstream reads semantic tokens rather than raw colours:

| Token | Dark | Light |
|---|---|---|
| `PAGE_BG` | `#212121` | `#FFFFFF` |
| `TEXT` | `#ECEFF1` | `#212121` |
| `TEXT_MUTED` | `#90A4AE` | `#607D8B` |
| `BORDER` | `#607D8B` | `#CFD8DC` |
| `SERIES_PRIMARY` | `#ECEFF1` | `#212121` |

In dark mode the two pale tints become *text* rather than fill — thin glyphs, not
a wash of colour — which is what keeps them within the "small places" rule.

Panels are separated by `#607D8B` hairlines rather than a lifted background, so
no sixth grey is needed. The correlation heatmap's midpoint is now the page
colour itself: a correlation of zero fades into the background while strong
relationships in either direction come forward.

**Streamlit's own colours are overridden.** Metric deltas ship red and green, and
alerts ship yellow, blue and green. The arrow already carries direction and the
text already carries severity, so both are restyled to one neutral panel with a
cyan marker.

**To switch back to light:** set `THEME = "light"` in `config.py` and change the
three marked lines in `.streamlit/config.toml`. Every chart follows automatically.

---

# Update — CoinGecko disabled at the backend

## 15. `CRYPTO_MODE = "simulated"`

Live CoinGecko calls are switched off. In simulated mode `load_crypto` returns
generated data immediately — no request is made, so there is no exception to
catch, nothing to log, and no 429 to retry. The dashboard is unchanged: six coins,
every tab, every metric, computed by the same code on the same shape of data.

`config.CRYPTO_MODE = "live"` re-enables fetching, and the throttling, backoff
and demo-key support added earlier are all still in place waiting for it. Macro,
sentiment and on-chain feeds were never affected and remain live.

**A chosen mode and a failure are now different things.** They produce identical
data but mean different things, so the interface treats them differently:

| Source label | Meaning | How it surfaces |
|---|---|---|
| `live` | Fetched successfully | Green badge |
| `simulated` | `CRYPTO_MODE` is set to simulated | Blue badge, one calm caption |
| `mixed` / `sample` | A live call failed unexpectedly | Amber badge, warning banner naming the coins |

The alarming yellow banner now only appears for genuine failures. Deliberate
simulation gets a single line of explanation under the title.

## 16. Simulated prices anchored to their final value

Two problems made the generated market look wrong on screen.

**The level.** A random walk left to run for a year from a fixed starting price
drifts wherever it drifts — Bitcoin was ending near $233,000. Paths are now
rescaled so the *last* value is the anchor rather than the first. Multiplying a
series by a constant does not change a single return, so volatility, VaR,
drawdown, correlation, beta and every phase label are identical either way. Only
the level moves.

**The dispersion.** Volatilities were set at the textbook "crypto is 65%
volatile" figure, which over a full year produced a Bitcoin range of $34k to
$166k. They now sit near levels these assets have actually realised — Bitcoin in
the mid-forties, the smaller alts higher.

The simulated market now reads as:

| | Latest | Range | Annual vol | Max drawdown |
|---|---|---|---|---|
| BTC | $95,000 | $54k – $148k | 47% | −40% |
| ETH | $3,400 | $1.8k – $5.1k | 55% | −44% |
| SOL | $180 | $44 – $313 | 85% | −56% |

Crypto-to-S&P correlation lands around +0.25 to +0.35, gold near zero — the
qualitative shape you would expect, and enough to make the correlation tab say
something real about method even though the data is generated.

---

# Update — dependencies installed once, not per version

## 17. The virtual environment now lives outside the project folder

`run.bat` previously created `.venv` inside the project directory. Extracting a
new version of the project therefore discarded the environment along with
everything else, and every dependency was reinstalled from scratch — a minute of
downloading for a change of a few lines.

The launcher now creates and looks for the environment one level up:

```
crypto\
├── .venv\                    <- created once, never replaced
└── crypto-market-analysis\   <- replace this folder freely
```

Search order is `%CRYPTO_VENV%`, then `..\.venv`, then `.venv` inside the
project for existing setups. Nothing breaks if the old layout is still in place;
it is simply found last.

**Self-healing dependency check.** After activating an existing environment the
launcher imports the core packages and reinstalls from `requirements.txt` only
if one is missing. That costs about a second per start and means a newly added
requirement never surfaces as an ImportError halfway through startup.

---

## 18. One rate-limited ticker emptied the entire correlation tab

**Was:** on the deployed app the Correlation tab rendered a blank heatmap and a
crypto-to-macro table reading `None` in every cell, while every other tab worked
normally and the sidebar reported all four feeds live.

The cause was a partial provider failure, not a total one. When yfinance rate
limits a single ticker it still returns the frame — that one column just comes
back entirely NaN. `_fetch_macro` ends with `dropna(how="all")`, which only drops
rows where *every* column is NaN, so all 367 rows survived with a dead `Gold`
column among the live ones. `align_returns` then ended with a bare `.dropna()`,
which is `how="any"`: every row contained the NaN from that one column, so every
row was discarded and the returns frame came back with **0 rows**. Correlating an
empty frame yields all-NaN, which is the blank heatmap and the wall of `None`.

The Streamlit Cloud log for that run recorded exactly this:

```
1 Failed download:
['GC=F']: YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')
```

So a single rate-limited commodity took down the correlation analysis for all
eleven assets, and did it silently — nothing in the UI said a feed was missing,
because as far as the loader was concerned the fetch had succeeded.

**Now:** `correlation.drop_thin_columns()` removes under-populated columns
*before* the row-wise `dropna`, so a dead feed costs one asset instead of the
whole sample. Coverage is measured against the best-populated column rather than
the raw row count, so a genuinely short history is not mistaken for a broken
feed; the threshold is `config.MIN_CORRELATION_COVERAGE` (0.5).

Verified against live data by simulating the failure at the loader:

| Scenario | Before | After |
|---|---|---|
| All feeds healthy | 250 rows × 11 cols | 250 rows × 11 cols (unchanged) |
| Gold rate-limited | **0 rows — tab blank** | 250 rows × 10 cols |
| Gold + Crude Oil dead | **0 rows — tab blank** | 250 rows × 9 cols |
| Gold returns 10 points | **0 rows — tab blank** | 250 rows × 10 cols |
| Every macro feed dead | **0 rows — tab blank** | 250 rows × 6 cols (crypto-to-crypto still correlates) |

**And it now says so.** Following the same principle as the live-vs-sample
badges, a dropped asset is named in a warning above the matrix rather than
quietly vanishing from a smaller table. The *Compare against* picker for the
rolling-correlation chart is also built from the surviving columns, because
offering an excluded asset left the user selecting an option that rendered
nothing at all.
