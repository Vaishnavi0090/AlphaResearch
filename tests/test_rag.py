"""RAG: chunking, BM25 scoring, filters and citation traceability."""

from __future__ import annotations

from pathlib import Path

import pytest

from finrisk.config.settings import RAGConfig
from finrisk.data.documents import FinancialDocument, infer_metadata, load_corpus
from finrisk.rag.retriever import (
    BM25Index,
    DocumentRetriever,
    chunk_document,
    rerank_by_coverage,
    tokenize,
)


def _doc(text: str, ticker: str | None = "AAA", doc_type: str = "10-K") -> FinancialDocument:
    return FinancialDocument(
        doc_id="d1",
        source="AAA_10K_2023.txt",
        ticker=ticker,
        doc_type=doc_type,
        text=text,
    )


# ---------------------------------------------------------------- tokenizer
def test_tokenizer_preserves_numbers_and_percentages():
    toks = tokenize("Revenue rose 3.2% to $1,250 million in fiscal 2023.")
    assert "3.2%" in toks
    assert "1250" in toks
    assert "2023" in toks


def test_tokenizer_removes_stopwords():
    assert "the" not in tokenize("the revenue")


def test_tokenizer_distinguishes_similar_numbers():
    assert tokenize("rose 3%") != tokenize("rose 30%")


# ---------------------------------------------------------------- chunking
def test_chunking_produces_overlapping_chunks():
    text = " ".join(f"Sentence number {i} about revenue and margin." for i in range(200))
    chunks = chunk_document(_doc(text), chunk_size=400, overlap=100)
    assert len(chunks) > 3
    assert all(c.tokens for c in chunks)
    assert all(c.doc_id == "d1" for c in chunks)


def test_chunk_ids_are_unique():
    text = " ".join(f"Fact {i} here." for i in range(300))
    ids = [c.chunk_id for c in chunk_document(_doc(text), 300, 50)]
    assert len(ids) == len(set(ids))


def test_chunking_empty_document():
    assert chunk_document(_doc("   ")) == []


def test_chunking_short_document_yields_one_chunk():
    chunks = chunk_document(_doc("Only one sentence."), 900, 150)
    assert len(chunks) == 1


def test_chunk_metadata_propagates():
    c = chunk_document(_doc("A sentence."), 900, 150)[0]
    assert c.ticker == "AAA" and c.doc_type == "10-K" and c.source.endswith(".txt")


# ---------------------------------------------------------------- metadata
@pytest.mark.parametrize(
    "name,ticker,dtype",
    [
        ("AAPL_10K_2023.txt", "AAPL", "10-K"),
        ("SY00_10K_2021.txt", "SY00", "10-K"),
        ("MSFT_earnings_2024Q1.txt", "MSFT", "earnings_release"),
        ("BRKB_transcript_2023.txt", "BRKB", "earnings_call"),
    ],
)
def test_infer_metadata(name, ticker, dtype):
    t, d = infer_metadata(Path(name))
    assert t == ticker
    assert d == dtype


def test_infer_metadata_alphanumeric_ticker_not_dropped():
    """Regression: a letters-only regex silently made ticker filters a no-op."""
    assert infer_metadata(Path("SY07_10K_2022.txt"))[0] == "SY07"


# ---------------------------------------------------------------- BM25
def test_bm25_ranks_relevant_chunk_first():
    idx = BM25Index()
    idx.add(chunk_document(_doc("Return on equity was 14 percent this year.")))
    d2 = FinancialDocument(
        doc_id="d2",
        source="b.txt",
        ticker="BBB",
        doc_type="10-K",
        text="Supply chain disruption affected freight costs materially.",
    )
    idx.add(chunk_document(d2))
    hits = idx.search("return on equity", top_k=2)
    assert hits
    assert hits[0][0].doc_id == "d1"


def test_bm25_empty_index_and_empty_query():
    assert BM25Index().search("anything") == []
    idx = BM25Index()
    idx.add(chunk_document(_doc("Some content here.")))
    assert idx.search("the and of") == []


def test_bm25_idf_is_non_negative():
    idx = BM25Index()
    for i in range(5):
        idx.add(
            chunk_document(
                FinancialDocument(
                    doc_id=f"d{i}",
                    source=f"{i}.txt",
                    ticker="X",
                    doc_type="10-K",
                    text="revenue revenue revenue margin",
                )
            )
        )
    assert idx._idf("revenue") >= 0.0
    assert idx._idf("nonexistentterm") == 0.0


def test_bm25_saturates_term_frequency():
    """A term repeated 50x must not score 50x a single occurrence."""
    idx = BM25Index()
    once = FinancialDocument("d1", "a.txt", "A", "10-K", "goodwill impairment occurred.")
    many = FinancialDocument("d2", "b.txt", "B", "10-K", "goodwill " * 50 + "impairment occurred.")
    idx.add(chunk_document(once))
    idx.add(chunk_document(many))
    scores = {c.doc_id: s for c, s in idx.search("goodwill", top_k=2)}
    assert scores["d2"] < 10 * scores["d1"]


def test_rerank_prefers_broader_coverage():
    idx = BM25Index()
    broad = FinancialDocument(
        "d1", "broad.txt", "A", "10-K", "revenue margin leverage growth all present."
    )
    narrow = FinancialDocument("d2", "narrow.txt", "B", "10-K", "revenue " * 12 + "only.")
    idx.add(chunk_document(broad))
    idx.add(chunk_document(narrow))
    q = "revenue margin leverage growth"
    reranked = rerank_by_coverage(idx.search(q, top_k=5), q, 2)
    assert reranked[0][0].doc_id == "d1"


# ---------------------------------------------------------------- retriever
@pytest.fixture(scope="module")
def retriever():
    r = DocumentRetriever(RAGConfig(top_k=3))
    docs = [
        FinancialDocument(
            doc_id=f"d{i}",
            source=f"T{i}_10K_2023.txt",
            ticker=f"T{i}",
            doc_type="10-K" if i % 2 == 0 else "earnings_release",
            text=(
                f"Company T{i} reported return on equity of {10 + i} percent. "
                f"Total debt per share was {i}.50 against equity of {i * 2}.00. "
                "Risk factors include commodity price volatility and rate shocks."
            ),
        )
        for i in range(6)
    ]
    r.add_documents(docs)
    return r


def test_retriever_indexes_documents(retriever):
    assert retriever.n_documents == 6
    assert retriever.n_chunks >= 6


def test_retriever_returns_citations_with_provenance(retriever):
    cits = retriever.retrieve("return on equity", top_k=3)
    assert cits
    for c in cits:
        assert c.doc_id and c.source and c.chunk_id and c.excerpt
        assert c.score > 0


def test_ticker_filter_is_strict(retriever):
    cits = retriever.retrieve("return on equity", top_k=5, ticker="T3")
    assert cits
    assert all(c.source.startswith("T3_") for c in cits)


def test_doc_type_filter(retriever):
    cits = retriever.retrieve("risk factors", top_k=5, doc_type="10-K")
    assert cits
    assert all("10K" in c.source for c in cits)


def test_unknown_ticker_returns_nothing(retriever):
    assert retriever.retrieve("return on equity", ticker="ZZZZ") == []


def test_context_block_is_tagged_and_bounded(retriever):
    cits = retriever.retrieve("return on equity", top_k=3)
    block = retriever.context_block(cits, max_chars=400)
    assert "[S1]" in block and "chunk=" in block
    assert len(block) <= 500


def test_empty_retriever_returns_nothing():
    assert DocumentRetriever(RAGConfig()).retrieve("anything") == []


def test_load_corpus_missing_directory_is_safe(tmp_path):
    assert load_corpus(tmp_path / "does_not_exist") == []
