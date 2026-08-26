"""Hybrid RAG: embeddings, vector store, fusion, provenance.

The load-bearing tests here are the ones that measure retrieval quality rather
than asserting it, and the one that proves dense retrieval finds paraphrases
BM25 cannot. Where dense retrieval does NOT help (the template-generated
synthetic corpus), that is asserted too, so the documented finding cannot
silently rot.
"""

from __future__ import annotations

import numpy as np
import pytest

from finrisk.config.settings import RAGConfig
from finrisk.data.documents import FinancialDocument
from finrisk.evaluation.rag_eval import (
    Probe,
    build_fact_probes,
    build_paraphrase_probes,
    evaluate_retriever,
    render_retrieval_report,
    run_retrieval_ablation,
)
from finrisk.rag.embeddings import (
    HashingEncoder,
    LSAEncoder,
    build_encoder,
)
from finrisk.rag.retriever import DocumentRetriever, chunk_document
from finrisk.rag.vector_store import InMemoryVectorStore, reciprocal_rank_fusion

# A corpus with genuine linguistic variety — each document says something
# different in different words. This is what dense retrieval is for.
PARAPHRASE_CORPUS = [
    (
        "d1",
        "Gross margin contracted 240 basis points as input cost inflation outpaced pricing actions.",
    ),
    (
        "d2",
        "The company redeemed its senior notes and reduced net borrowings, lowering interest expense.",
    ),
    (
        "d3",
        "Headcount reductions and facility consolidation drove restructuring charges in the quarter.",
    ),
    (
        "d4",
        "Order backlog expanded as customers accelerated capital spending in the infrastructure market.",
    ),
    ("d5", "Litigation reserves were increased following an adverse ruling in the patent dispute."),
    ("d6", "Foreign exchange translation reduced reported revenue by 180 basis points."),
]


def _docs(pairs):
    return [FinancialDocument(i, f"{i}.txt", "TST", "10-K", t) for i, t in pairs]


# ---------------------------------------------------------------- encoders
def test_lsa_produces_unit_norm_vectors():
    enc = LSAEncoder(dim=8).fit([t for _, t in PARAPHRASE_CORPUS])
    V = enc.encode([t for _, t in PARAPHRASE_CORPUS])
    assert V.shape[0] == len(PARAPHRASE_CORPUS)
    assert np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-6)


def test_lsa_is_deterministic():
    texts = [t for _, t in PARAPHRASE_CORPUS]
    a = LSAEncoder(dim=8, seed=0).fit(texts).encode(texts)
    b = LSAEncoder(dim=8, seed=0).fit(texts).encode(texts)
    assert np.allclose(a, b)


def test_lsa_dim_capped_by_corpus_size():
    """SVD cannot yield more components than the corpus supports."""
    enc = LSAEncoder(dim=512).fit([t for _, t in PARAPHRASE_CORPUS])
    assert enc.dim < 512
    assert enc.encode(["margin"]).shape[1] == enc.dim


def test_lsa_places_related_text_closer_than_unrelated():
    texts = [t for _, t in PARAPHRASE_CORPUS]
    enc = LSAEncoder(dim=5).fit(texts)
    V = enc.encode(texts)
    q = enc.encode(["debt repayment and interest burden"])[0]
    sims = V @ q
    assert int(np.argmax(sims)) == 1, "the borrowings document should rank first"


def test_encode_before_fit_raises():
    with pytest.raises(RuntimeError):
        LSAEncoder().encode(["anything"])


def test_hashing_encoder_is_deterministic_and_normed():
    enc = HashingEncoder(dim=64)
    a = enc.encode(["revenue margin"])
    b = enc.encode(["revenue margin"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)


def test_build_encoder_falls_back_gracefully():
    enc = build_encoder("auto", dim=16)
    assert hasattr(enc, "fit") and hasattr(enc, "encode")
    assert enc.name in {"lsa", "hashing", "sentence-transformer"}


# ---------------------------------------------------------------- store
def test_vector_store_search_orders_by_cosine():
    store = InMemoryVectorStore(dim=3)
    store.add(
        ["a", "b", "c"],
        np.array([[1.0, 0, 0], [0, 1.0, 0], [0.7071, 0.7071, 0]]),
        [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}],
    )
    hits = store.search(np.array([1.0, 0.0, 0.0]), top_k=3)
    assert [h[0] for h in hits] == ["a", "c", "b"]
    assert hits[0][1] == pytest.approx(1.0)


def test_vector_store_filters_before_scoring():
    """Post-filtering would silently return fewer than top_k."""
    store = InMemoryVectorStore(dim=2)
    store.add(
        ["a", "b", "c"],
        np.array([[1.0, 0], [0.9, 0.1], [0, 1.0]]),
        [{"ticker": "X"}, {"ticker": "Y"}, {"ticker": "X"}],
    )
    hits = store.search(np.array([1.0, 0.0]), 2, where=lambda m: m["ticker"] == "X")
    assert len(hits) == 2
    assert {h[0] for h in hits} == {"a", "c"}


def test_vector_store_updates_rather_than_duplicates():
    store = InMemoryVectorStore(dim=2)
    store.add(["a"], np.array([[1.0, 0.0]]), [{"v": 1}])
    store.add(["a"], np.array([[0.0, 1.0]]), [{"v": 2}])
    assert len(store) == 1
    assert store.get("a")["v"] == 2


def test_vector_store_rejects_dim_mismatch():
    store = InMemoryVectorStore(dim=3)
    with pytest.raises(ValueError):
        store.add(["a"], np.array([[1.0, 0.0]]), [{}])
    with pytest.raises(ValueError):
        store.search(np.array([1.0, 0.0]))


def test_vector_store_rejects_length_mismatch():
    store = InMemoryVectorStore(dim=2)
    with pytest.raises(ValueError):
        store.add(["a", "b"], np.array([[1.0, 0.0]]), [{}])


def test_empty_store_returns_nothing():
    assert InMemoryVectorStore(dim=4).search(np.zeros(4)) == []


def test_vector_store_roundtrip(tmp_path):
    store = InMemoryVectorStore(dim=2)
    store.add(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]), [{"t": "A"}, {"t": "B"}])
    store.save(tmp_path / "idx")
    loaded = InMemoryVectorStore.load(tmp_path / "idx")
    assert len(loaded) == 2
    assert loaded.search(np.array([1.0, 0.0]), 1)[0][0] == "a"


# ---------------------------------------------------------------- fusion
def test_rrf_rewards_appearing_in_both_rankings():
    """A document both retrievers like beats one only a single retriever found.

    Note what is NOT asserted: that a consistently-second document beats one
    ranked first-then-third. With k=60 those scores are 0.032258 vs 0.032266 —
    RRF gives no such guarantee, and asserting it would encode a false property
    of the algorithm.
    """
    fused = reciprocal_rank_fusion([["a", "b", "x"], ["a", "b", "y"]], k=60)
    assert fused["a"] > fused["b"] > fused["x"]
    assert fused["x"] > 0 and fused["y"] > 0
    # Present in both lists beats present in one at the same rank.
    single = reciprocal_rank_fusion([["z"], []], k=60)
    assert fused["a"] > single["z"]


def test_rrf_respects_weights():
    lex = reciprocal_rank_fusion([["a", "z"], ["z", "a"]], 60, [10.0, 1.0])
    assert lex["a"] > lex["z"]


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], 60, [1.0, 1.0])


# ---------------------------------------------------------------- retriever
def test_hybrid_retriever_reports_diagnostics():
    r = DocumentRetriever(RAGConfig(embedding_dim=8))
    r.add_documents(_docs(PARAPHRASE_CORPUS))
    r.retrieve("margin")
    d = r.diagnostics()
    assert d["mode"] == "hybrid"
    assert d["dense_ready"] is True
    assert d["encoder"] in {"lsa", "hashing", "sentence-transformer"}
    assert d["n_chunks"] == r.n_chunks


def test_dense_finds_paraphrase_that_bm25_misses():
    """The core justification for adding dense retrieval at all.

    'profitability decline' shares no content word with 'Gross margin
    contracted ... as input cost inflation outpaced pricing actions'.
    """
    r = DocumentRetriever(RAGConfig(top_k=1, embedding_dim=5))
    r.add_documents(_docs(PARAPHRASE_CORPUS))
    query = "why did profitability decline"
    bm25 = r.retrieve(query, 1, mode="bm25")
    dense = r.retrieve(query, 1, mode="dense")
    assert not bm25 or bm25[0].doc_id != "d1", "fixture no longer tests paraphrase"
    assert dense and dense[0].doc_id == "d1"


def test_hybrid_retains_dense_wins():
    r = DocumentRetriever(RAGConfig(top_k=1, embedding_dim=5))
    r.add_documents(_docs(PARAPHRASE_CORPUS))
    hits = r.retrieve("why did profitability decline", 1, mode="hybrid")
    assert hits and hits[0].doc_id == "d1"


def test_retrieval_mode_is_recorded_on_citations():
    r = DocumentRetriever(RAGConfig(embedding_dim=8))
    r.add_documents(_docs(PARAPHRASE_CORPUS))
    for mode in ("bm25", "dense", "hybrid"):
        hits = r.retrieve("margin", 2, mode=mode)
        assert hits
        assert all(h.retrieval_mode for h in hits)


def test_bm25_mode_does_not_build_dense_index():
    r = DocumentRetriever(RAGConfig(retriever="bm25", embedding_dim=8))
    r.add_documents(_docs(PARAPHRASE_CORPUS))
    r.retrieve("margin")
    assert r.diagnostics()["dense_ready"] is False


def test_adding_documents_invalidates_dense_index():
    r = DocumentRetriever(RAGConfig(embedding_dim=5))
    r.add_documents(_docs(PARAPHRASE_CORPUS[:3]))
    r.retrieve("margin")
    before = r.n_chunks
    r.add_documents(_docs(PARAPHRASE_CORPUS[3:]))
    assert r.retrieve("backlog", 3)
    assert r.n_chunks > before
    assert r.diagnostics()["dense_ready"] is True


def test_ticker_filter_applies_to_every_mode():
    docs = [
        FinancialDocument("a1", "AAA_10K_2023.txt", "AAA", "10-K", "Revenue rose sharply."),
        FinancialDocument("b1", "BBB_10K_2023.txt", "BBB", "10-K", "Revenue rose sharply."),
    ]
    r = DocumentRetriever(RAGConfig(embedding_dim=2))
    r.add_documents(docs)
    for mode in ("bm25", "dense", "hybrid"):
        hits = r.retrieve("revenue", 5, ticker="AAA", mode=mode)
        assert hits, mode
        assert all(h.ticker == "AAA" for h in hits), mode


# ---------------------------------------------------------------- provenance
def test_citations_carry_section_and_page():
    doc = FinancialDocument(
        "x",
        "AAA_10K_2023.txt",
        "AAA",
        "10-K",
        "Item 1A. Risk Factors. " + "Commodity prices may fluctuate materially. " * 40,
    )
    r = DocumentRetriever(RAGConfig(embedding_dim=4))
    r.add_documents([doc])
    hits = r.retrieve("commodity prices", 1)
    assert hits
    assert hits[0].section == "Item 1A - Risk Factors"
    assert hits[0].page is not None and hits[0].page >= 1


def test_section_filter_narrows_results():
    doc = FinancialDocument(
        "x",
        "AAA_10K_2023.txt",
        "AAA",
        "10-K",
        "Item 1A. Risk Factors. Commodity exposure is significant. " * 20
        + "Item 7. Management's discussion. Liquidity remained strong. " * 20,
    )
    r = DocumentRetriever(RAGConfig(embedding_dim=4))
    r.add_documents([doc])
    hits = r.retrieve("commodity liquidity", 5, section="Risk Factors")
    assert hits
    assert all("Risk Factors" in (h.section or "") for h in hits)


def test_chunk_metadata_is_complete():
    doc = FinancialDocument("x", "AAA_10K_2023.txt", "AAA", "10-K", "Revenue grew. " * 50)
    meta = chunk_document(doc)[0].metadata()
    for key in ("chunk_id", "doc_id", "source", "ticker", "doc_type", "page"):
        assert key in meta


# ---------------------------------------------------------------- evaluation
def test_ndcg_never_exceeds_one():
    """Regression: counting several chunks from one filing as separate relevant
    documents pushed nDCG to 1.365."""
    r = DocumentRetriever(RAGConfig(embedding_dim=5))
    r.add_documents(_docs(PARAPHRASE_CORPUS))
    probes = [Probe("margin contraction", {"d1.txt"}, "fact")]
    for mode in ("bm25", "dense", "hybrid"):
        sc = evaluate_retriever(r, probes, mode, k=5)
        assert sc.ndcg_at_5 <= 1.0 + 1e-9


def test_probe_builders_produce_scoped_probes(full_bundle):
    facts = build_fact_probes(full_bundle, n_tickers=3)
    paras = build_paraphrase_probes(full_bundle, n_tickers=3)
    assert facts and paras
    assert all(p.ticker for p in facts + paras), "probes must be issuer-scoped"
    assert all(p.gold_sources for p in facts + paras)
    assert all(p.kind == "paraphrase" for p in paras)


def test_ablation_scores_every_mode(full_bundle, full_settings):
    from finrisk.data.documents import load_corpus

    docs = load_corpus(full_settings.rag.docs_dir)
    if not docs:
        pytest.skip("corpus not generated; run scripts/run_all.py")
    r = DocumentRetriever(full_settings.rag)
    r.add_documents(docs)
    probes = build_fact_probes(full_bundle, 4) + build_paraphrase_probes(full_bundle, 4)
    df = run_retrieval_ablation(r, probes, k=3)
    assert set(df.index) == {"bm25", "dense", "hybrid"}
    assert (df["ndcg@5"] <= 1.0 + 1e-9).all()
    # Documented finding: on this TEMPLATE-generated corpus, hybrid does not
    # beat BM25 — the prose barely varies, so there is no semantics to exploit.
    # Hybrid must at least stay close, which is what justifies it as a default
    # that is safe across corpus types.
    assert df.loc["hybrid", "mrr"] >= df.loc["bm25", "mrr"] - 0.05
    assert "Retrieval Ablation" in render_retrieval_report(df)
