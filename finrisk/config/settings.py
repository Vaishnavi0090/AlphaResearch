"""Configuration objects for FINRISK AI.

Everything that a researcher might want to change lives here, not in code.
Configs are Pydantic models so that YAML/JSON configs are validated at load
time rather than failing deep inside a backtest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "data_cache"

TRADING_DAYS = 252
"""Annualisation constant. 252 is the conventional US equity trading-day count."""


class DataConfig(BaseModel):
    """Where prices/fundamentals come from and how they are cached."""

    provider: Literal["synthetic", "yfinance"] = "synthetic"
    universe: list[str] = Field(default_factory=list)
    benchmark: str = "BENCH"
    start: str = "2010-01-04"
    end: str = "2024-12-31"
    cache_dir: Path = DEFAULT_CACHE_DIR
    use_cache: bool = True
    # Lag applied to fundamental data to model reporting delay (see
    # docs/DATA_LEAKAGE_AUDIT.md). 90 calendar days is a conservative proxy
    # for the gap between fiscal period end and public filing.
    fundamental_publication_lag_days: int = 90
    risk_free_annual: float = 0.02
    synthetic_seed: int = 7
    # Universe size for the synthetic provider. Configurable so the same code
    # path serves a fast smoke run and the full evaluation.
    synthetic_assets: int = Field(40, ge=5, le=500)

    @field_validator("cache_dir")
    @classmethod
    def _abs(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()


class FactorConfig(BaseModel):
    """Factor construction and composite weighting."""

    momentum_lookbacks: list[int] = Field(default_factory=lambda: [21, 63, 126, 252])
    momentum_skip_days: int = 21  # skip most recent month (short-term reversal)
    volatility_lookback: int = 63
    beta_lookback: int = 252
    drawdown_lookback: int = 252
    winsorize_quantile: float = 0.02
    min_cross_section: int = 5
    # Composite weights. NOT claimed to be optimal — Phase 24 ablations and
    # Phase 25 robustness sweeps test alternatives.
    composite_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "momentum": 0.30,
            "value": 0.25,
            "quality": 0.25,
            "risk": 0.20,
        }
    )

    @model_validator(mode="after")
    def _check_weights(self) -> FactorConfig:
        if not self.composite_weights:
            raise ValueError("composite_weights must not be empty")
        total = sum(self.composite_weights.values())
        if abs(total) < 1e-12:
            raise ValueError("composite_weights must not sum to zero")
        return self

    def normalised_weights(self) -> dict[str, float]:
        total = sum(self.composite_weights.values())
        return {k: v / total for k, v in self.composite_weights.items()}


class RegimeConfig(BaseModel):
    """Gaussian HMM regime detector settings."""

    n_states: int = 3
    features: list[str] = Field(default_factory=lambda: ["ret_21d", "vol_21d"])
    train_min_obs: int = 500
    max_iter: int = 120
    tol: float = 1e-5
    seed: int = 0
    refit_every: int = 252  # re-estimate HMM parameters once per year


class PortfolioConfig(BaseModel):
    """Portfolio construction constraints."""

    method: Literal[
        "equal_weight",
        "mean_variance",
        "max_sharpe",
        "risk_parity",
        "risk_constrained",
        "cvar_constrained",
    ] = "risk_constrained"
    top_n: int = 10
    max_weight: float = 0.20
    min_weight: float = 0.0
    max_sector_weight: float = 0.45
    target_volatility: float | None = 0.15
    max_cvar_95: float | None = None
    risk_aversion: float = 5.0
    # One-way turnover ceiling per rebalance, as a fraction of NAV. Caps
    # trading costs directly rather than hoping a cost penalty discourages
    # them. None = unconstrained. Requires prev_weights to bind.
    max_turnover: float | None = None
    # Ceiling on any single asset's share of total portfolio volatility
    # (Euler component / total). A weight cap does NOT bound risk share: a
    # 20% position in a name twice as volatile as its peers can carry 35%+
    # of the risk. This constrains what actually matters.
    max_risk_contribution: float | None = None
    max_leverage: float = 1.0
    cash_allowed: bool = True
    covariance_estimator: Literal["sample", "ledoit_wolf"] = "ledoit_wolf"
    covariance_lookback: int = 252

    @model_validator(mode="after")
    def _check(self) -> PortfolioConfig:
        if self.max_weight <= 0 or self.max_weight > 1:
            raise ValueError("max_weight must be in (0, 1]")
        if self.min_weight < 0 or self.min_weight >= self.max_weight:
            raise ValueError("min_weight must be in [0, max_weight)")
        if self.top_n * self.max_weight < 1.0 - 1e-9 and not self.cash_allowed:
            raise ValueError("Infeasible: top_n * max_weight < 1 with cash_allowed=False")
        return self


class CostConfig(BaseModel):
    """Transaction-cost model. Simplified — see docs/BACKTESTING.md."""

    commission_bps: float = 2.0
    spread_bps: float = 5.0
    slippage_bps: float = 3.0

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.spread_bps + self.slippage_bps

    @property
    def cost_per_unit_turnover(self) -> float:
        """Cost as a fraction of traded notional."""
        return self.total_bps / 10_000.0


class BacktestConfig(BaseModel):
    """Walk-forward evaluation protocol."""

    train_window: int = 504  # 2y of daily data used to estimate moments
    rebalance_frequency: Literal["M", "Q", "W"] = "M"
    warmup: int = 504  # bars consumed before the first decision is allowed
    initial_capital: float = 1_000_000.0
    costs: CostConfig = Field(default_factory=CostConfig)
    # Signals computed at close of day t are executed at close of day t+1.
    execution_lag_days: int = 1

    @model_validator(mode="after")
    def _check(self) -> BacktestConfig:
        if self.execution_lag_days < 1:
            raise ValueError(
                "execution_lag_days must be >= 1; executing on the signal bar " "is look-ahead bias"
            )
        return self


class RAGConfig(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    # "hybrid" fuses BM25 with dense retrieval via Reciprocal Rank Fusion.
    # "bm25" and "dense" isolate a single path, which is how the ablation in
    # docs/RAG.md measures what the dense half actually contributes.
    retriever: Literal["bm25", "dense", "hybrid"] = "hybrid"
    # "auto" prefers sentence-transformers, falls back to corpus-fitted LSA.
    encoder: Literal["auto", "lsa", "sentence-transformer", "hashing"] = "auto"
    embedding_dim: int = 128
    # RRF constant from the original paper; larger values flatten top ranks.
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    # Candidates pulled from each retriever before fusion and reranking.
    candidate_multiplier: int = 4
    docs_dir: Path = REPO_ROOT / "data_cache" / "documents"


class AgentConfig(BaseModel):
    llm_provider: Literal["stub", "anthropic"] = "stub"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2000
    temperature: float = 0.0
    enable_investment_committee: bool = True


class Settings(BaseModel):
    """Root config object."""

    name: str = "finrisk-default"
    data: DataConfig = Field(default_factory=DataConfig)
    factors: FactorConfig = Field(default_factory=FactorConfig)
    regimes: RegimeConfig = Field(default_factory=RegimeConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    random_seed: int = 42

    @classmethod
    def from_file(cls, path: str | Path) -> Settings:
        raw = Path(path).read_text()
        if str(path).endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ImportError("pyyaml required for YAML configs") from exc
            return cls.model_validate(yaml.safe_load(raw))
        return cls.model_validate(json.loads(raw))

    def to_file(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))


def default_settings() -> Settings:
    return Settings()
