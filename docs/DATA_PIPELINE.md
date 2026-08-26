# Data Pipeline

## Providers

| Provider | Source | Status |
|---|---|---|
| `SyntheticProvider` | Seeded DGP with known ground truth | Default; used for all reported results |
| `YFinanceProvider` | Yahoo Finance prices | Implemented; blocked by network policy in the build environment |
| `FredProvider` | FRED public CSV (risk-free rates) | Implemented; blocked |
| `EdgarProvider` | SEC EDGAR company facts | Implemented; blocked |

Every financial data host returned HTTP 403 during the build, so the synthetic provider is the default. All results are labelled synthetic-data validation. Switching is one config line: `settings.data.provider = "yfinance"`.

## The synthetic DGP

Not noise — a structured process with **known ground truth**, which is what makes `tests/test_factor_recovery.py` possible.

- **Market regimes**: 3-state Markov chain — bull (drift +14%, vol 11%), chop (+2%, 16%), bear (−18%, 34%)
- **Trend state**: AR(1), φ=0.985 (~46-session half-life)
- **Value / quality states**: slow AR(1), φ=0.9985
- **True premia**: trend 3.0%, value 4.5%, quality 3.5% annualised
- **Fundamentals**: quarterly, published with 45–75 day lag (median 61)
- **Benchmark**: cap-weighted from the same universe

Latent states are exposed via `bundle.ground_truth`, which is how the momentum negative result was diagnosed rather than guessed at.

## Point-in-time discipline

The pipeline's most important property. Every fundamental record carries two dates:

- `period_end` — the fiscal period described
- `available_from` — when it became public

`MarketDataBundle.fundamentals_asof(d)` is the **only sanctioned accessor**. It filters `available_from <= d` and returns the latest filing per ticker.

The naive alternative — indexing by `period_end` — leaks ~2 months of hindsight into every value factor and is enough to manufacture an impressive, entirely fictitious backtest. `stamp_availability` also clamps impossible dates where a source claims availability before period end.

Verified: 2,400 records, **0** availability violations.

## Cleaning

**`normalize_index`** — strips timezones, normalises to date, sorts, and drops duplicate timestamps keeping the **last** (a republished bar is a correction).

**`limited_ffill`** — bounded forward fill (default 5 sessions). Unbounded ffill is a real hazard: a delisted stock keeps "trading" at its last price forever, producing a zero-volatility, zero-return asset that optimisers love. The bound means a genuinely dead stock goes NaN and drops out.

**`drop_sparse_columns`** — removes assets below a coverage threshold.

**`flag_abnormal_returns`** — flags moves beyond a threshold for review rather than silently deleting them; a 50% move may be a bad tick or may be real.

One bug found here: pandas 3.0's `stack()` retains NaN, so without `.dropna()` every non-flagged cell was reported as an anomaly. Caught by `test_flag_abnormal_returns_empty_when_clean`.

## Validation

`validate_bundle()` checks missing data, duplicate timestamps, non-positive prices, stale prices, abnormal returns, calendar alignment, price/benchmark alignment, and fundamental PIT integrity. Issues are typed with severity.

Full run: 40 assets, 156,480 observations, **0 missing, 0 duplicates, 0 errors**, 1 informational note.

`assert_no_lookahead_in_frame` and `check_monotone_availability` are available as inline guards.

## Caching

Pickle-based, keyed by a hash of provider parameters, with exponential-backoff retry for network providers. Cache keys include the seed and universe size, so changing either produces a different key rather than a stale hit.
