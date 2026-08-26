# FINRISK AI — Evaluation Report

## Dataset

- Provider: `synthetic`
- Assets: 20
- Period: 2014-01-02 to 2020-12-31 (1826 sessions)
- Benchmark: `BENCH`

**Data notes:**

- Synthetic DGP, seed=7, n_assets=20.
- Results are DGP validation, NOT historical performance.
- True annualised premia: {'trend': 0.03, 'value': 0.045, 'quality': 0.035}
- Calendar-aligned to 1826 sessions.

## Strategy comparison

```
Strategy Comparison
======================================================================================================================
Strategy                                CAGR     Vol  Sharpe  Sortino    MaxDD  Calmar   CVaR95   Beta   Turn    Cost
----------------------------------------------------------------------------------------------------------------------
benchmark_buy_hold                    8.58%   16.6%    0.46     0.65   -29.5%    0.29   -2.30%   1.00   0.0%   0.00%
A_equal_weight_universe               6.75%   14.5%    0.39     0.55   -20.8%    0.32   -1.98%   0.82   3.6%   0.43%
B_factor_equal_weight                 6.70%   15.2%    0.37     0.53   -23.1%    0.29   -2.02%   0.82  16.9%   2.03%
C_factor_regime_equal_weight          6.93%   13.0%    0.43     0.61   -19.3%    0.36   -1.71%   0.69  18.7%   2.25%
D_factor_mean_variance                6.81%   14.5%    0.39     0.55   -25.7%    0.27   -1.94%   0.75  20.5%   2.46%
D2_factor_risk_parity                 6.78%   14.8%    0.38     0.54   -23.7%    0.29   -1.99%   0.80  17.2%   2.06%
E_factor_optimized_risk_constrained   8.74%   15.1%    0.50     0.72   -24.8%    0.35   -1.96%   0.73  24.1%   2.89%
F_finrisk_full                        8.84%   12.9%    0.57     0.82   -18.4%    0.48   -1.66%   0.61  23.7%   2.85%
----------------------------------------------------------------------------------------------------------------------
Turn = mean one-way turnover per rebalance. Cost = cumulative transaction-cost drag over the full period.
```

## Ablation study

```
Ablation Study
================================================================================================================
Rung                                      CAGR   dCAGR     Vol  Sharpe   dShrp    MaxDD   CVaR95    Turn
----------------------------------------------------------------------------------------------------------------
A_equal_weight_universe                 6.75%      —   14.5%    0.39      —   -20.8%   -1.98%    3.6%
B_factor_equal_weight                   6.70% -0.04%   15.2%    0.37  -0.01   -23.1%   -2.02%   16.9%
C_factor_regime_equal_weight            6.93%  0.23%   13.0%    0.43   0.06   -19.3%   -1.71%   18.7%
D_factor_mean_variance                  6.81% -0.13%   14.5%    0.39  -0.04   -25.7%   -1.94%   20.5%
D2_factor_risk_parity                   6.78% -0.03%   14.8%    0.38  -0.01   -23.7%   -1.99%   17.2%
E_factor_optimized_risk_constrained     8.74%  1.96%   15.1%    0.50   0.12   -24.8%   -1.96%   24.1%
F_finrisk_full                          8.84%  0.11%   12.9%    0.57   0.07   -18.4%   -1.66%   23.7%
----------------------------------------------------------------------------------------------------------------

Deltas are versus the PREVIOUS rung, so each row isolates one component. D2 is an alternative to D, not a further step; its delta is vs D.
```

## Reproduction

```bash
python scripts/run_all.py --full
```

Every row carries the `config_digest` of the settings that produced it.
