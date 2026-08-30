# Engineering log

A record of the defects found in this system and the reasoning behind each fix.
It is kept because the *why* is the part that does not survive in a diff, and
because several of these are mistakes that are easy to make again.

Every entry below is covered by a regression test in `tests/`. The suite is the
enforcement mechanism; this document is the explanation.

---

## 1. Analytical correctness

### 1.1 RSI reported maximum strength as neutral

`avg_loss` of zero made the relative-strength ratio undefined, and filling the
resulting `NaN` with 50 meant an unbroken run of gains — the strongest possible
momentum reading — was reported as perfectly neutral.

The two degenerate cases are now separated: no losses with gains present is 100,
genuinely flat is 50, and the warm-up window stays `NaN` rather than being given
a fabricated value.

Wilder's smoothing also requires `adjust=False`. Pandas defaults `ewm` to
`adjust=True`, which is a different weighting scheme; the difference is largest
immediately after the warm-up and decays as the series lengthens.

*Tests:* `test_cycles.py::TestRSI`

### 1.2 Sortino used the wrong denominator

The implementation divided by `returns[returns < 0].std()` — the standard
deviation of the losing days. That measures dispersion about the mean of the
losses, which is not downside risk.

Downside deviation is `sqrt(mean(min(r − target, 0)²))`, taken across *all*
observations against the risk-free target. The distinction matters: adding
winning days must reduce measured downside risk, and under the old formula it
did not. The correct denominator is larger, so the corrected ratio is lower.

*Tests:* `test_volatility.py::TestDownsideDeviation`, `::TestSortino`

### 1.3 Yields were treated as prices

A 10-year Treasury yield moving from 4.00% to 4.10% was recorded as a `+2.5%`
return. It is a 10 basis-point change. Columns listed in `config.YIELD_ASSETS`
are now differenced in percentage points; everything else is percent-changed.
Correlation is scale-invariant, so the two mix without further adjustment.

*Tests:* `test_correlation.py::TestYieldHandling`

### 1.4 "Drawdown vs all-time high" was a window high

`price.cummax()` is the running maximum *inside the loaded window*. On a
365-day history that is a one-year peak, not an all-time high. The metric is
labelled "Drawdown vs window high" throughout.

### 1.5 Annualisation uses 365 days

Crypto trades every day of the year. Using the 252-day equity convention
understates annualised volatility by roughly 20%.

---

## 2. Validation methodology

### 2.1 The forecast was not being measured against anything

The model was fitted on a training split and asked for a single forecast
spanning the entire ~55-day test period. For a series close to a random walk
that forecast is nearly flat, so the resulting error measured how far the price
happened to drift over two months — not model quality. There was also no
baseline, leaving a reported "RMSE $4,200" uninterpretable.

Three changes:

- **Walk-forward validation.** Predictions are made one day ahead at a time,
  with each actual observation appended to the model state before the next
  prediction. Parameters stay fixed at their training values (`refit=False`),
  so no information from later in the test period leaks backwards.
- **A naive baseline.** The random-walk rule — tomorrow's price is today's —
  scored on exactly the same days.
- **A skill score.** `1 − RMSE(model) / RMSE(naive)`. Positive means the model
  adds something.

The dashboard states the outcome in plain language, including when the model
loses to the baseline, which for daily crypto returns is the usual result and a
correct finding rather than a defect.

*Tests:* `test_forecast.py`

### 2.2 The direction model is validated on time-ordered splits

`TimeSeriesSplit` trains on the past and tests on the future, always. A random
split on a time series lets a model learn from days that had not happened yet;
it is the most common route to a financial machine-learning result that cannot
be reproduced. Every feature is computed from information available at the close
of its own day, and the target is the sign of the *next* day's move.

Accuracy is reported against the majority-class base rate measured on the same
held-out days, not against 50%. An asset that rose on 54% of days makes 54% the
number to beat.

*Tests:* `test_direction.py` — including a test that truncating the series does
not change any feature value on the days that remain, which is what look-ahead
would violate.

---

## 3. Data integrity

### 3.1 One rate-limited ticker emptied the whole correlation analysis

When yfinance rate-limits a single ticker it still returns the frame; that one
column simply comes back entirely `NaN`. Correlation needs every column
populated on the same day, so the row-wise `dropna` then discarded *every* row
and the analysis ran on an empty sample — a blank heatmap and a table of `None`,
with nothing in the interface indicating a feed was missing.

`correlation.drop_thin_columns()` removes under-populated columns *before* rows
are dropped, so a dead feed costs one asset instead of the entire matrix.
Coverage is measured against the best-populated column rather than the row count,
so a genuinely short history is not mistaken for a broken feed. Any excluded
asset is named in the interface.

| Scenario | Before | After |
|---|---|---|
| All feeds healthy | 250 × 11 | 250 × 11 |
| One ticker rate-limited | **0 rows** | 250 × 10 |
| Two tickers dead | **0 rows** | 250 × 9 |
| Every macro feed dead | **0 rows** | 250 × 6, crypto still correlates |

*Tests:* `test_correlation.py::TestPartialProviderFailure`

### 3.2 Preprocessing is centralised and reported

Cleaning happens in `src/data/preprocessing.py` rather than as scattered
`dropna` and `astype` calls, and every step reports what it did. The order is
deliberate: de-duplicate the index before anything counts a day twice; coerce
types before values are compared; collapse to one row per day; remove impossible
values before a return turns a zero into an infinity; forward-fill gaps —
forward only, because back-filling moves a later observation into an earlier day
and is look-ahead.

Outliers are **flagged, never removed**. Value-at-Risk and CVaR exist to
describe the tail of the return distribution; trimming that tail before
measuring it produces a comfortable number that is wrong in the one direction
that matters.

Traditional assets are not resampled onto a daily grid. Those markets are
genuinely closed at weekends, and inventing rows would manufacture zero-return
days that drag every measured correlation toward zero.

A column's dtype is not used to decide what is numeric. Under pandas 2 a string
column arrives as `object`; under pandas 3 it arrives as a dedicated `str`
dtype, so a check for `object` silently skipped the Fear & Greed value — the one
column that actually needed coercing. Coercion is attempted on every non-numeric
column and kept only where it parses, which behaves identically on both.

*Tests:* `test_preprocessing.py`

### 3.3 Generated fallback data is reproducible and correlated

Two properties were wrong in the original generator.

Seeds derived from Python's built-in `hash()` are randomised per process, so the
"deterministic" fallback produced different data on every restart. Seeds now come
from `zlib.crc32`, which is stable across processes — asserted by a test that
generates the same series in three separate interpreters and compares digests.

Assets were also independent geometric Brownian motions, so the offline
correlation matrix showed approximately zero everywhere: sampling noise
presented as a finding. Every series is now driven by a shared risk-on factor
plus idiosyncratic noise, with per-asset betas. This is *simulated* structure,
not measured structure, and the interface says so whenever a feed is generated.

*Tests:* `test_sample_data.py`

---

## 4. Data availability

### 4.1 Rate limiting made the system appear broken

Six coins requested back-to-back against CoinGecko's keyless tier reliably
earned HTTP 429 on the last few. Each 429 cost 30 seconds of backoff, and a
measured cold start took **74.8 seconds** — most of it spent recovering from a
rate limit that wider spacing would have avoided. Reported as "the API is not
connecting", it was in fact connecting and succeeding, slowly, behind a
motionless spinner.

Four changes:

- **Spacing targets the worst documented limit, not the average.** Keyless
  access is documented at roughly 10–30 calls per minute; 2.5-second spacing
  (24/min) sat inside that band on paper but not in practice. At 6 seconds
  (10/min) six coins complete in about 30–36 seconds with no 429 at all —
  slower per call, faster overall, because it never pays the backoff tax.
- **A demo key removes the problem entirely.** Spacing drops to 0.8 seconds and
  the same six coins complete in **6.3 seconds**.
- **Progress is reported per coin**, so a slow fetch is visibly working.
- **Failures are not cached.** Caching a fallback pinned the system to generated
  data for the full TTL even after connectivity returned.

### 4.2 Stale real data is preferred over fresh synthetic data

If a fetch fails and an expired response for the same key is still on disk, the
expired response is served rather than the generated fallback.

This system is distributed as a folder and read at an unknown later date,
possibly with no network. Given the choice between real data carrying an older
date and invented data carrying today's, real is the more honest thing to
present — and the Datasets tab states the date range of every feed, so nothing
is passed off as more current than it is.

*Tests:* `test_data_layer.py::TestStaleBeatsSynthetic`

### 4.3 Distribution ships with a populated cache

`data/cache/` contains real responses from all five feeds across all four
history windows, so the system starts in under a second with no API key and no
network connection.

---

## 5. Presentation

### 5.1 Denomination

The interface is denominated in Indian rupees. CoinGecko supports `inr` as a
native `vs_currency`, so crypto prices are *fetched* in rupees — no conversion
and no cross-rate error. Dollar-quoted traditional assets are converted at the
daily USD/INR rate.

Two macro series are deliberately not converted: the US Dollar Index is an index
level, and the 10-year yield is a rate in percentage points. Neither is a price
in dollars.

Converting before differencing means returns carry the USD/INR move as a shared
component. That is not a distortion — it is the correct frame for an investor
whose portfolio is denominated in rupees, whose actual gain on gold is the gold
move *and* the currency move.

Indian digit grouping is applied: 7456772 renders as `74,56,772`, not
`7,456,772`.

*Tests:* `test_currency.py`, including a check that no hardcoded currency symbol
survives anywhere in the presentation layer.

### 5.2 Adaptive moving averages

A 200-day moving average produces no values at all until its 200th observation,
so on a 90- or 180-day window every day was labelled "Undetermined" and the
associated metric rendered as `nan%`.

`config.ma_windows()` keeps the canonical 50/200 pair when there is enough
history and otherwise scales both windows down, preserving the 1:4 ratio. The
windows actually in use are reported alongside every label, so a reading is
never quietly computed on a different basis than the reader assumes.

| Window | Pair | Days labelled |
|---|---|---|
| 90 | 7 / 30 | 68% |
| 180 | 15 / 60 | 67% |
| 270 | 22 / 90 | 67% |
| 365 | 50 / 200 | 45% |

### 5.3 Cycle rules

Two rules described situations they were not meant to describe. *Distribution*
now additionally requires price to be within 10% of the peak, because
distribution is supply meeting demand near a top rather than any stalling
uptrend. *Accumulation* now requires a drawdown past 25% to already be in place,
because a shallow early decline is not a basing bottom.

### 5.4 Visual scheme

Five colours carry the entire interface, applied so that colour signals meaning
rather than decoration. A Plotly template is constructed from scratch rather
than extending a stock one, because the stock templates embed their own
colourways and per-trace colour scales that any unstyled trace would silently
inherit. A test extracts every hex value from every figure and asserts that
nothing outside the scheme reaches the screen.

*Tests:* `test_charts.py::TestPaletteIsClosed`

---

## 6. Known limitations

Stated because they bound what the results mean.

- **History is capped at one year.** CoinGecko's free tier limit. A full crypto
  cycle historically runs about four years, so phase labels describe the current
  regime rather than a position within a complete cycle.
- **On-chain metrics are Bitcoin-only**, a limitation of the free
  Blockchain.info charts. Exchange inflow and outflow would be the single most
  valuable addition.
- **Sentiment is a composite index, not text analysis.** The Fear & Greed index
  stands in for direct social-media processing, which requires a paid feed.
- **`d=1` is assumed rather than established** by an ADF test. Defensible for
  log prices, but not demonstrated.
- **The direction model shows no reliable lift.** Reported as measured. On this
  sample only the macro feature block produces a positive lift, and at a
  magnitude inside the noise for the sample size.
