# FINRISK AI — Evaluation Report

## Dataset

- Provider: `synthetic`
- Assets: 40
- Period: 2010-01-04 to 2024-12-31 (3912 sessions)
- Benchmark: `BENCH`

**Data notes:**

- Synthetic DGP, seed=7, n_assets=40.
- Results are DGP validation, NOT historical performance.
- True annualised premia: {'trend': 0.03, 'value': 0.045, 'quality': 0.035}
- Calendar-aligned to 3912 sessions.

## Strategy comparison

```
Strategy Comparison
======================================================================================================================
Strategy                                CAGR     Vol  Sharpe  Sortino    MaxDD  Calmar   CVaR95   Beta   Turn    Cost
----------------------------------------------------------------------------------------------------------------------
benchmark_buy_hold                    1.91%   14.3%    0.05     0.07   -55.2%    0.03   -1.98%   1.00   0.0%   0.00%
A_equal_weight_universe               2.39%   14.8%    0.08     0.12   -52.0%    0.05   -2.06%   1.00   3.0%   0.93%
B_factor_equal_weight                 6.18%   15.2%    0.33     0.47   -30.3%    0.20   -2.10%   0.95  21.5%   6.72%
C_factor_regime_equal_weight          6.06%   12.8%    0.35     0.50   -26.1%    0.23   -1.78%   0.78  21.6%   6.75%
D_factor_mean_variance                6.36%   13.4%    0.36     0.52   -23.5%    0.27   -1.84%   0.81  22.2%   6.92%
D2_factor_risk_parity                 6.40%   14.5%    0.35     0.50   -28.4%    0.23   -2.00%   0.91  21.3%   6.64%
E_factor_optimized_risk_constrained   6.69%   15.5%    0.35     0.51   -26.2%    0.26   -2.10%   0.90  26.5%   8.25%
F_finrisk_full                        6.81%   13.3%    0.40     0.57   -23.7%    0.29   -1.82%   0.74  24.7%   7.71%
----------------------------------------------------------------------------------------------------------------------
Turn = mean one-way turnover per rebalance. Cost = cumulative transaction-cost drag over the full period.
```

## Ablation study

```
Ablation Study
================================================================================================================
Rung                                      CAGR   dCAGR     Vol  Sharpe   dShrp    MaxDD   CVaR95    Turn
----------------------------------------------------------------------------------------------------------------
A_equal_weight_universe                 2.39%      —   14.8%    0.08      —   -52.0%   -2.06%    3.0%
B_factor_equal_weight                   6.18%  3.80%   15.2%    0.33   0.24   -30.3%   -2.10%   21.5%
C_factor_regime_equal_weight            6.06% -0.12%   12.8%    0.35   0.02   -26.1%   -1.78%   21.6%
D_factor_mean_variance                  6.36%  0.31%   13.4%    0.36   0.01   -23.5%   -1.84%   22.2%
D2_factor_risk_parity                   6.40%  0.03%   14.5%    0.35  -0.01   -28.4%   -2.00%   21.3%
E_factor_optimized_risk_constrained     6.69%  0.30%   15.5%    0.35   0.00   -26.2%   -2.10%   26.5%
F_finrisk_full                          6.81%  0.12%   13.3%    0.40   0.04   -23.7%   -1.82%   24.7%
----------------------------------------------------------------------------------------------------------------

Deltas are versus the PREVIOUS rung, so each row isolates one component. D2 is an alternative to D, not a further step; its delta is vs D.
```

## Reproduction

```bash
python scripts/run_all.py --full
```

Every row carries the `config_digest` of the settings that produced it.
