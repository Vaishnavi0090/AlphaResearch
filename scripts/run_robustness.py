#!/usr/bin/env python3
"""Run ONE robustness sweep and append its rows to disk.

Split by sweep so each invocation is short and results survive interruption:

    python scripts/run_robustness.py cost
    python scripts/run_robustness.py frequency
    python scripts/run_robustness.py weights
    python scripts/run_robustness.py constraints
    python scripts/run_robustness.py subperiod
    python scripts/run_robustness.py --render

Rows accumulate in evaluation/results/robustness.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from finrisk.backtesting.engine import WalkForwardBacktester  # noqa: E402
from finrisk.backtesting.strategies import all_strategies  # noqa: E402
from finrisk.config.settings import default_settings  # noqa: E402
from finrisk.data.market_data import align_to_calendar, load_bundle  # noqa: E402
from finrisk.evaluation.runner import EvaluationRun, _row, _subperiods  # noqa: E402

OUT = ROOT / "evaluation" / "results" / "robustness.json"
STRATEGY = "F_finrisk_full"


def load_rows() -> list[dict]:
    if OUT.exists():
        return json.loads(OUT.read_text()).get("rows", [])
    return []


def save_rows(rows: list[dict], notes: list[str]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"name": "robustness", "rows": rows, "notes": notes},
            indent=2,
            default=str,
        )
    )


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    sweep = sys.argv[1]

    settings = default_settings()
    bundle = align_to_calendar(load_bundle(settings.data))
    strat = all_strategies()[STRATEGY]
    rows = load_rows()
    done = {r["label"] for r in rows}

    if sweep == "--render":
        run = EvaluationRun("robustness", "", {}, rows=rows)
        from finrisk.evaluation.runner import render_robustness

        print(render_robustness(run))
        pd.DataFrame(rows).to_csv(ROOT / "evaluation" / "results" / "robustness.csv", index=False)
        return 0

    configs: list[tuple[str, str, object, object]] = []

    if sweep == "cost":
        for bps in (0.0, 5.0, 10.0, 20.0, 40.0):
            s = settings.model_copy(deep=True)
            s.backtest.costs.commission_bps = bps / 2
            s.backtest.costs.spread_bps = bps / 2
            s.backtest.costs.slippage_bps = 0.0
            configs.append((f"cost_{bps:g}bps", "transaction_cost", bps, s))

    elif sweep == "frequency":
        for freq in ("W", "M", "Q"):
            s = settings.model_copy(deep=True)
            s.backtest.rebalance_frequency = freq
            configs.append((f"rebalance_{freq}", "rebalance_frequency", freq, s))

    elif sweep == "weights":
        schemes = {
            "baseline_30_25_25_20": {
                "momentum": 0.30,
                "value": 0.25,
                "quality": 0.25,
                "risk": 0.20,
            },
            "equal_all": {"momentum": 0.25, "value": 0.25, "quality": 0.25, "risk": 0.25},
            "momentum_tilt": {"momentum": 0.55, "value": 0.15, "quality": 0.15, "risk": 0.15},
            "value_tilt": {"momentum": 0.15, "value": 0.55, "quality": 0.15, "risk": 0.15},
            "quality_tilt": {"momentum": 0.15, "value": 0.15, "quality": 0.55, "risk": 0.15},
            "defensive": {"momentum": 0.15, "value": 0.20, "quality": 0.25, "risk": 0.40},
        }
        for name, w in schemes.items():
            s = settings.model_copy(deep=True)
            s.factors.composite_weights = w
            configs.append((f"weights_{name}", "factor_weights", json.dumps(w), s))

    elif sweep == "constraints":
        for top_n, max_w, vol in (
            (5, 0.35, 0.15),
            (10, 0.20, 0.15),
            (20, 0.12, 0.15),
            (10, 0.20, 0.10),
            (10, 0.20, 0.25),
        ):
            s = settings.model_copy(deep=True)
            s.portfolio.top_n = top_n
            s.portfolio.max_weight = max_w
            s.portfolio.target_volatility = vol
            configs.append(
                (
                    f"cons_n{top_n}_w{max_w:g}_v{vol:g}",
                    "constraints",
                    f"top_n={top_n},max_w={max_w},vol={vol}",
                    s,
                )
            )

    elif sweep == "subperiod":
        for label, sub in _subperiods(bundle, settings).items():
            key = f"period_{label}"
            if key in done:
                print(f"skip {key}")
                continue
            t0 = time.perf_counter()
            res = WalkForwardBacktester(settings).run(sub, strat)
            rows.append(_row(res, label=key, extra={"sweep": "subperiod", "value": label}))
            print(
                f"{key:<34} CAGR {res.metrics.cagr:>7.2%}  "
                f"Sharpe {res.metrics.sharpe:>5.2f}  "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
            save_rows(rows, [])
        return 0

    else:
        print(f"unknown sweep {sweep!r}")
        return 2

    # Sweeps that share a context where the factor/regime config is unchanged.
    ctx_cache: dict[str, object] = {}
    for label, sweep_name, value, s in configs:
        if label in done:
            print(f"skip {label}")
            continue
        t0 = time.perf_counter()
        key = json.dumps(
            {
                "f": s.factors.model_dump(mode="json"),
                "r": s.regimes.model_dump(mode="json"),
                "q": s.backtest.rebalance_frequency,
            },
            sort_keys=True,
        )
        bt = WalkForwardBacktester(s)
        if key not in ctx_cache:
            ctx_cache[key] = bt.prepare_context(bundle)
        res = bt.run(bundle, strat, ctx_cache[key])
        rows.append(_row(res, label=label, extra={"sweep": sweep_name, "value": value}))
        print(
            f"{label:<34} CAGR {res.metrics.cagr:>7.2%}  "
            f"Sharpe {res.metrics.sharpe:>5.2f}  "
            f"vol {res.metrics.ann_volatility:>6.1%}  "
            f"turn {res.metrics.avg_turnover:>5.1%}  "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
        save_rows(rows, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
