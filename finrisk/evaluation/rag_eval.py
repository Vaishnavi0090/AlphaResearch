"""Retrieval evaluation (recall@k, MRR, nDCG) with ground-truth labels.

Retrieval quality claims need an evaluation set, and a hand-written one is
circular — you write queries you already know the retriever answers. So the
gold labels here are derived mechanically from the corpus:

* **Fact probes.** The synthetic corpus is generated from known fundamentals,
  so a query naming a ticker and a metric has exactly one correct source
  document. Gold labels come from the generator, not from judgement.
* **Paraphrase probes.** Queries phrased with vocabulary that deliberately does
  *not* appear in the target text, testing semantic rather than lexical match.

Both probe families run against every retrieval mode, so "hybrid is better" is
a measurement rather than an assumption — and if the measurement says
otherwise, that is the result reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from finrisk.rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)


@dataclass
class Probe:
    """One query with its known-correct document(s)."""

    query: str
    # Matched against Citation.source (the filename), NOT doc_id: doc_id is a
    # content hash, so it cannot be predicted from the generator's inputs.
    gold_sources: set[str]
    kind: str = "fact"
    note: str = ""
    # Restricting to one issuer is what makes the task discriminating: the
    # retriever must pick the right FILING YEAR from that issuer's filings
    # rather than being handed the answer by the ticker in the query.
    ticker: str | None = None


@dataclass
class RetrievalScore:
    mode: str
    n_probes: int
    recall_at_1: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    mean_latency_ms: float
    by_kind: dict[str, float] = field(default_factory=dict)


def _dcg(rels: list[float]) -> float:
    return float(sum(r / np.log2(i + 2) for i, r in enumerate(rels)))


def evaluate_retriever(
    retriever: DocumentRetriever, probes: list[Probe], mode: str, k: int = 5
) -> RetrievalScore:
    """Standard IR metrics for one retrieval mode."""
    import time

    r1 = r5 = rr = ndcg = 0.0
    latencies: list[float] = []
    kind_hits: dict[str, list[float]] = {}

    for probe in probes:
        t0 = time.perf_counter()
        hits = retriever.retrieve(probe.query, top_k=k, mode=mode, ticker=probe.ticker)
        latencies.append((time.perf_counter() - t0) * 1000)

        # Deduplicate by source, preserving rank order. Several chunks from the
        # same filing are one retrieved DOCUMENT; counting each as a separate
        # relevant hit inflates DCG above the ideal (observed nDCG of 1.365).
        seen: set[str] = set()
        ranked: list[str] = []
        for h in hits:
            if h.source not in seen:
                seen.add(h.source)
                ranked.append(h.source)
        rels = [1.0 if d in probe.gold_sources else 0.0 for d in ranked]

        hit1 = float(bool(ranked) and ranked[0] in probe.gold_sources)
        hit5 = float(any(rels))
        r1 += hit1
        r5 += hit5
        first = next((i for i, rel in enumerate(rels) if rel > 0), None)
        rr += 1.0 / (first + 1) if first is not None else 0.0
        ideal = _dcg([1.0] * min(len(probe.gold_sources), k))
        ndcg += _dcg(rels) / ideal if ideal > 0 else 0.0
        kind_hits.setdefault(probe.kind, []).append(hit5)

    n = max(len(probes), 1)
    return RetrievalScore(
        mode=mode,
        n_probes=len(probes),
        recall_at_1=r1 / n,
        recall_at_5=r5 / n,
        mrr=rr / n,
        ndcg_at_5=ndcg / n,
        mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        by_kind={kk: float(np.mean(v)) for kk, v in kind_hits.items()},
    )


def build_fact_probes(bundle: Any, n_tickers: int = 12) -> list[Probe]:
    """Queries whose correct answer is known from the generator.

    Each query is scoped to one issuer and quotes a NUMBER that appears in
    exactly one of that issuer's filings, so the retriever must discriminate
    between several near-identical documents on numeric content alone. This is
    where exact lexical matching is expected to win.
    """
    probes: list[Probe] = []
    fund = bundle.fundamentals
    if not len(fund):
        return probes
    f = fund.copy()
    f["period_end"] = pd.to_datetime(f["period_end"])

    for ticker in list(bundle.tickers)[:n_tickers]:
        sub = f[f["ticker"] == ticker].sort_values("period_end")
        if len(sub) < 4:
            continue
        for year in sorted(sub["period_end"].dt.year.unique())[-3:]:
            rows = sub[sub["period_end"].dt.year == year]
            if not len(rows):
                continue
            last = rows.iloc[-1]
            eps = float(last["eps"])
            fcf = float(last["free_cash_flow_per_share"])
            gold = {f"{ticker}_10K_{int(year)}.txt"}
            probes.append(
                Probe(
                    query=f"free cash flow per share of {fcf:.2f}",
                    gold_sources=gold,
                    kind="fact",
                    note="numeric discrimination",
                    ticker=ticker,
                )
            )
            probes.append(
                Probe(
                    query=f"diluted earnings per share {eps:.2f} fiscal {int(year)}",
                    gold_sources=gold,
                    kind="fact",
                    note="numeric + period",
                    ticker=ticker,
                )
            )
    return probes


def build_paraphrase_probes(bundle: Any, n_tickers: int = 12) -> list[Probe]:
    """Queries worded to avoid the corpus's own vocabulary.

    Scoped to one issuer and one fiscal year, phrased the way a person asks
    rather than the way a filing is written. If dense retrieval adds nothing
    here, it adds nothing anywhere.
    """
    probes: list[Probe] = []
    fund = bundle.fundamentals
    if not len(fund):
        return probes
    f = fund.copy()
    f["period_end"] = pd.to_datetime(f["period_end"])

    for ticker in list(bundle.tickers)[:n_tickers]:
        sub = f[f["ticker"] == ticker].sort_values("period_end")
        if len(sub) < 4:
            continue
        for year in sorted(sub["period_end"].dt.year.unique())[-3:]:
            gold = {f"{ticker}_10K_{int(year)}.txt"}
            for template, note in (
                ("{y} what could go wrong for this business", "risk paraphrase"),
                ("{y} how much cash the operations actually throw off", "cash paraphrase"),
                ("{y} how profitable is it on shareholder capital", "roe paraphrase"),
            ):
                probes.append(
                    Probe(
                        query=template.format(y=int(year)),
                        gold_sources=gold,
                        kind="paraphrase",
                        note=note,
                        ticker=ticker,
                    )
                )
    return probes


def run_retrieval_ablation(
    retriever: DocumentRetriever,
    probes: list[Probe],
    modes: tuple[str, ...] = ("bm25", "dense", "hybrid"),
    k: int = 5,
) -> pd.DataFrame:
    """Score every retrieval mode on the same probes."""
    rows = []
    for mode in modes:
        sc = evaluate_retriever(retriever, probes, mode, k)
        rows.append(
            {
                "mode": sc.mode,
                "n_probes": sc.n_probes,
                "recall@1": sc.recall_at_1,
                "recall@5": sc.recall_at_5,
                "mrr": sc.mrr,
                "ndcg@5": sc.ndcg_at_5,
                "latency_ms": sc.mean_latency_ms,
                **{f"recall@5_{kk}": v for kk, v in sc.by_kind.items()},
            }
        )
    return pd.DataFrame(rows).set_index("mode")


def render_retrieval_report(df: pd.DataFrame, encoder: str = "") -> str:
    if df.empty:
        return "No retrieval evaluation produced."
    lines = [
        "Retrieval Ablation" + (f" (dense encoder: {encoder})" if encoder else ""),
        "=" * 92,
        f"{'Mode':<10}{'R@1':>8}{'R@5':>8}{'MRR':>8}{'nDCG@5':>9}"
        f"{'R@5 fact':>11}{'R@5 para':>11}{'ms':>8}",
        "-" * 92,
    ]
    for mode, r in df.iterrows():
        lines.append(
            f"{mode:<10}{r['recall@1']:>8.2f}{r['recall@5']:>8.2f}{r['mrr']:>8.3f}"
            f"{r['ndcg@5']:>9.3f}{r.get('recall@5_fact', float('nan')):>11.2f}"
            f"{r.get('recall@5_paraphrase', float('nan')):>11.2f}"
            f"{r['latency_ms']:>8.1f}"
        )
    lines.append("-" * 92)
    lines.append(
        "Gold labels are derived from the corpus generator, not hand-written, so "
        "the probe set is not tuned to any one retriever."
    )
    return "\n".join(lines)
