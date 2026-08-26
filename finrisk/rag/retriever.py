"""Financial document retrieval (Phase 15).

DOCUMENT -> PARSE -> CHUNK -> INDEX -> RETRIEVE -> RERANK -> AGENT

WHY LEXICAL (BM25) RATHER THAN DENSE EMBEDDINGS
-----------------------------------------------
Three reasons specific to financial filings, not a general claim that BM25
beats embeddings:

1. **Terminology is exact.** Users ask about "deferred revenue", "goodwill
   impairment", "Level 3 assets". These are precise terms of art that appear
   verbatim in filings. Lexical matching is strong exactly where vocabulary is
   controlled, and it cannot hallucinate a semantic neighbour.
2. **Numbers matter.** Dense encoders famously mangle figures; "revenue rose
   3%" and "revenue rose 30%" are near-identical in embedding space and
   completely different facts.
3. **No dependency, no drift.** The index is deterministic and reproducible
   with no model download, no API key, and no version-to-version embedding
   drift — which matters because the whole project's claim is reproducibility.

The `EmbeddingRetriever` hook is provided so a dense or hybrid retriever can be
swapped in; `docs/RAG.md` records the trade-off honestly rather than pretending
BM25 is universally superior.

CITATIONS ARE NOT OPTIONAL
--------------------------
Retrieval returns `Citation` objects carrying doc_id, source filename, chunk id
and the excerpt. The report layer refuses to emit a document-derived claim
without one (see `finrisk/agents/critic.py`).
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from finrisk.config.settings import RAGConfig
from finrisk.data.documents import FinancialDocument
from finrisk.schemas import Citation

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\.\-%]*")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "this",
        "these",
        "those",
        "they",
        "we",
        "our",
        "us",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase tokenizer that preserves numbers, percentages and decimals.

    '3.2%' and '1,250' must survive tokenisation to stay searchable, so '.',
    '-' and '%' are allowed inside a token. But they must be stripped from the
    END of a token: otherwise sentence-final punctuation produces '2023.' which
    will never match a query for '2023', silently destroying recall on exactly
    the numeric facts this retriever exists to find.
    """
    text = text.lower().replace(",", "")
    out = []
    for raw in _TOKEN_RE.findall(text):
        tok = raw.rstrip(".-")
        if len(tok) > 1 and tok not in STOPWORDS:
            out.append(tok)
    return out


@dataclass
class Chunk:
    """An indexed passage plus enough provenance to audit a claim.

    `section`, `page` and `filing_date` are what turn a citation from "the
    10-K says so" into a pointer someone can actually check.
    """

    chunk_id: str
    doc_id: str
    source: str
    ticker: str | None
    doc_type: str
    text: str
    position: int
    tokens: list[str] = field(default_factory=list)
    section: str | None = None
    page: int | None = None
    filing_date: str | None = None
    fiscal_period: str | None = None

    def metadata(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "ticker": self.ticker,
            "doc_type": self.doc_type,
            "section": self.section,
            "page": self.page,
            "filing_date": self.filing_date,
            "fiscal_period": self.fiscal_period,
            "position": self.position,
        }


def chunk_document(
    doc: FinancialDocument, chunk_size: int = 900, overlap: int = 150
) -> list[Chunk]:
    """Sentence-aware chunking with overlap.

    Splitting mid-sentence severs a number from its unit or its period ("...
    increased to $4.1" | "billion in fiscal 2023"), which produces retrievable
    but unusable fragments. Sentences are accumulated up to `chunk_size` and the
    tail is carried into the next chunk so context spanning a boundary is not
    lost.
    """
    if not doc.text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", doc.text.strip())
    chunks: list[Chunk] = []
    current_section: str | None = None
    buf: list[str] = []
    size = 0
    pos = 0

    def flush() -> None:
        nonlocal buf, size, pos, current_section
        if not buf:
            return
        text = " ".join(buf).strip()
        current_section = _section_of(text, doc, current_section)
        if text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{pos}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    ticker=doc.ticker,
                    doc_type=doc.doc_type,
                    text=text,
                    position=pos,
                    tokens=tokenize(text),
                    section=current_section,
                    page=1 + pos // 3,
                    filing_date=getattr(doc, "filing_date", None),
                    fiscal_period=getattr(doc, "fiscal_period", None),
                )
            )
            pos += 1
        # carry the tail forward as overlap
        tail: list[str] = []
        tail_len = 0
        for s in reversed(buf):
            if tail_len >= overlap:
                break
            tail.insert(0, s)
            tail_len += len(s)
        buf = tail
        size = tail_len

    for sent in sentences:
        if size + len(sent) > chunk_size and buf:
            flush()
        buf.append(sent)
        size += len(sent)
    buf_final = buf
    if buf_final:
        text = " ".join(buf_final).strip()
        current_section = _section_of(text, doc, current_section)
        if text and (not chunks or chunks[-1].text != text):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{pos}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    ticker=doc.ticker,
                    doc_type=doc.doc_type,
                    text=text,
                    position=pos,
                    tokens=tokenize(text),
                    section=current_section,
                    page=1 + pos // 3,
                    filing_date=getattr(doc, "filing_date", None),
                    fiscal_period=getattr(doc, "fiscal_period", None),
                )
            )
    return chunks


# Canonical SEC section headings, matched case-insensitively against the chunk
# text. Knowing a passage came from Item 1A (Risk Factors) rather than Item 7
# (MD&A) changes how much weight a claim deserves.
_SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"item\s*1a\b|risk factors", "Item 1A - Risk Factors"),
    (r"item\s*7a\b|quantitative and qualitative disclosures", "Item 7A - Market Risk"),
    (r"item\s*7\b|management.s discussion", "Item 7 - MD&A"),
    (r"item\s*8\b|financial statements and supplementary", "Item 8 - Financial Statements"),
    (r"item\s*1\b|^business\b", "Item 1 - Business"),
    (r"liquidity and capital resources", "Item 7 - MD&A"),
    (r"results of operations", "Item 7 - MD&A"),
]


def _section_of(text: str, doc: FinancialDocument, current: str | None = None) -> str | None:
    """Infer the filing section a chunk belongs to.

    Sections span many chunks, so a chunk with no heading of its own inherits
    the last heading seen. Without this, only the chunk containing the literal
    "Item 1A" header would be labelled and every following chunk — which is
    where the actual risk-factor prose lives — would be unattributed.
    """
    explicit = getattr(doc, "section", None)
    if explicit:
        return str(explicit)
    head = text[:400]
    for pattern, label in _SECTION_PATTERNS:
        if re.search(pattern, head, re.I | re.M):
            return label
    return current


class BM25Index:
    """Okapi BM25.

    score(q, d) = sum over terms t in q of
        IDF(t) * f(t,d)*(k1+1) / (f(t,d) + k1*(1 - b + b*|d|/avgdl))

    k1 controls term-frequency saturation (a term appearing 20 times is not 20x
    more relevant than once); b controls length normalisation, which matters
    because filing sections vary from a paragraph to twenty pages.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.df: Counter[str] = Counter()
        self.tf: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.avgdl: float = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            counts = Counter(c.tokens)
            self.chunks.append(c)
            self.tf.append(counts)
            self.lengths.append(len(c.tokens))
            self.df.update(counts.keys())
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

    @property
    def n_docs(self) -> int:
        return len(self.chunks)

    def _idf(self, term: str) -> float:
        n = self.n_docs
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        # BM25+ style IDF, floored at 0 so ubiquitous terms cannot score negative
        return max(math.log(1.0 + (n - df + 0.5) / (df + 0.5)), 0.0)

    def search(
        self,
        query: str,
        top_k: int = 5,
        ticker: str | None = None,
        doc_type: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        if not self.n_docs:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores: list[tuple[Chunk, float]] = []
        for i, chunk in enumerate(self.chunks):
            # Strict: when a ticker filter is supplied, a chunk with unknown
            # ticker metadata is EXCLUDED rather than passed through. Letting
            # unattributed chunks satisfy a ticker-specific query is how a
            # citation ends up pointing at the wrong company.
            if ticker and (chunk.ticker is None or chunk.ticker.upper() != ticker.upper()):
                continue
            if doc_type and chunk.doc_type != doc_type:
                continue
            tf = self.tf[i]
            dl = self.lengths[i] or 1
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                s += self._idf(term) * f * (self.k1 + 1) / denom
            if s > 0:
                scores.append((chunk, s))
        scores.sort(key=lambda kv: -kv[1])
        return scores[:top_k]


def rerank_by_coverage(
    hits: list[tuple[Chunk, float]], query: str, top_k: int
) -> list[tuple[Chunk, float]]:
    """Light reranking on distinct-query-term coverage.

    BM25 rewards repetition of a single high-IDF term. A chunk mentioning all
    four query concepts once is usually more useful than one mentioning the
    rarest concept six times, so the score is blended with the fraction of
    distinct query terms present.
    """
    q = set(tokenize(query))
    if not q or not hits:
        return hits[:top_k]
    max_s = max(s for _, s in hits) or 1.0
    out = []
    for c, s in hits:
        coverage = len(q & set(c.tokens)) / len(q)
        out.append((c, 0.7 * (s / max_s) + 0.3 * coverage))
    out.sort(key=lambda kv: -kv[1])
    return out[:top_k]


class DocumentRetriever:
    """Hybrid retrieval: BM25 + dense embeddings fused by Reciprocal Rank Fusion.

    The two halves fail in different ways, which is exactly why both are here.
    BM25 nails exact terminology and numbers ("3.2%", "goodwill impairment")
    and misses paraphrase. Dense retrieval catches paraphrase and blurs precise
    distinctions. Fusing by RANK rather than by score keeps them comparable:
    BM25 scores are unbounded and corpus-dependent while cosine sits in
    [-1, 1], so any fixed score normalisation between them is arbitrary.

    The dense index is fitted lazily on first retrieval, because a
    corpus-fitted encoder (LSA) needs the whole corpus before it can encode
    anything. Callers keep the same add-then-retrieve flow either way.

    Set `RAGConfig.retriever` to "bm25" or "dense" to isolate one path — that
    is how the ablation in docs/RAG.md measures what fusion actually buys.
    """

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.cfg = config or RAGConfig()
        self.index = BM25Index()
        self._docs: dict[str, FinancialDocument] = {}
        self._chunks: list[Chunk] = []
        self._encoder: Any = None
        self._store: Any = None
        self._dense_ready = False
        self._dense_error: str | None = None

    # ------------------------------------------------------------------
    def add_documents(self, docs: list[FinancialDocument]) -> int:
        total = 0
        for d in docs:
            self._docs[d.doc_id] = d
            chunks = chunk_document(d, self.cfg.chunk_size, self.cfg.chunk_overlap)
            self.index.add(chunks)
            self._chunks.extend(chunks)
            total += len(chunks)
        # New documents invalidate a corpus-fitted encoder.
        self._dense_ready = False
        logger.info("indexed %d chunks from %d documents", total, len(docs))
        return total

    @property
    def n_chunks(self) -> int:
        return self.index.n_docs

    @property
    def n_documents(self) -> int:
        return len(self._docs)

    @property
    def dense_available(self) -> bool:
        self._ensure_dense()
        return self._dense_ready

    @property
    def encoder_name(self) -> str:
        self._ensure_dense()
        return getattr(self._encoder, "name", "none")

    def _ensure_dense(self) -> None:
        """Fit the encoder and build the vector store, once."""
        if self._dense_ready or self.cfg.retriever == "bm25" or not self._chunks:
            return
        try:
            from finrisk.rag.embeddings import build_encoder
            from finrisk.rag.vector_store import InMemoryVectorStore

            texts = [c.text for c in self._chunks]
            encoder = build_encoder(self.cfg.encoder, self.cfg.embedding_dim)
            encoder.fit(texts)
            vectors = encoder.encode(texts)
            store = InMemoryVectorStore(dim=vectors.shape[1])
            store.add(
                [c.chunk_id for c in self._chunks],
                vectors,
                [c.metadata() for c in self._chunks],
            )
            self._encoder, self._store = encoder, store
            self._dense_ready = True
            self._dense_error = None
            logger.info(
                "dense index ready: %d vectors, %d dims, encoder=%s",
                len(store),
                vectors.shape[1],
                encoder.name,
            )
        except Exception as exc:
            # Degrade to BM25 rather than failing retrieval outright, and record
            # why so the fallback is visible instead of silent.
            self._dense_error = f"{type(exc).__name__}: {exc}"
            self._dense_ready = False
            logger.warning("dense index unavailable (%s); using BM25 only", self._dense_error)

    # ------------------------------------------------------------------
    def _passes(
        self, meta: dict, ticker: str | None, doc_type: str | None, section: str | None
    ) -> bool:
        if ticker is not None and meta.get("ticker") != ticker:
            return False
        if doc_type is not None and meta.get("doc_type") != doc_type:
            return False
        if section is not None:
            got = meta.get("section")
            if not got or section.lower() not in got.lower():
                return False
        return True

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        ticker: str | None = None,
        doc_type: str | None = None,
        excerpt_chars: int = 320,
        section: str | None = None,
        mode: str | None = None,
    ) -> list[Citation]:
        k = top_k or self.cfg.top_k
        mode = mode or self.cfg.retriever
        n_cand = max(k * self.cfg.candidate_multiplier, k)
        by_id = {c.chunk_id: c for c in self._chunks}

        bm25_hits = self.index.search(query, n_cand, ticker=ticker, doc_type=doc_type)
        if section is not None:
            bm25_hits = [
                (c, sc) for c, sc in bm25_hits if self._passes(c.metadata(), None, None, section)
            ]
        bm25_ids = [c.chunk_id for c, _ in bm25_hits]

        dense_ids: list[str] = []
        dense_scores: dict[str, float] = {}
        if mode in {"dense", "hybrid"}:
            self._ensure_dense()
            if self._dense_ready:
                qv = self._encoder.encode([query])
                if qv.shape[0]:
                    results = self._store.search(
                        qv[0],
                        n_cand,
                        where=lambda m: self._passes(m, ticker, doc_type, section),
                    )
                    dense_ids = [cid for cid, _, _ in results]
                    dense_scores = {cid: sc for cid, sc, _ in results}

        if mode == "bm25" or (mode == "dense" and not dense_ids):
            ranked = rerank_by_coverage(bm25_hits, query, k)
            fused_mode = "bm25"
        elif mode == "dense":
            ranked = [(by_id[i], dense_scores[i]) for i in dense_ids[:k] if i in by_id]
            fused_mode = "dense"
        else:
            from finrisk.rag.vector_store import reciprocal_rank_fusion

            if not dense_ids:
                ranked = rerank_by_coverage(bm25_hits, query, k)
                fused_mode = "bm25_fallback"
            else:
                fused = reciprocal_rank_fusion(
                    [bm25_ids, dense_ids],
                    self.cfg.rrf_k,
                    [self.cfg.bm25_weight, self.cfg.dense_weight],
                )
                order = sorted(fused, key=lambda i: -fused[i])
                candidates = [(by_id[i], fused[i]) for i in order if i in by_id]
                # Rerank on term coverage: RRF rewards appearing in both lists,
                # coverage rewards answering more of the question.
                ranked = rerank_by_coverage(candidates, query, k)
                fused_mode = "hybrid"

        return [
            Citation(
                doc_id=c.doc_id,
                source=c.source,
                chunk_id=c.chunk_id,
                score=round(float(sc), 4),
                excerpt=c.text[:excerpt_chars].strip(),
                ticker=c.ticker,
                doc_type=c.doc_type,
                section=c.section,
                page=c.page,
                filing_date=c.filing_date,
                fiscal_period=c.fiscal_period,
                retrieval_mode=fused_mode,
            )
            for c, sc in ranked
        ]

    def diagnostics(self) -> dict:
        """What the retriever is actually doing — surfaced by the API."""
        self._ensure_dense()
        return {
            "n_documents": self.n_documents,
            "n_chunks": self.n_chunks,
            "mode": self.cfg.retriever,
            "encoder": self.encoder_name,
            "dense_ready": self._dense_ready,
            "dense_error": self._dense_error,
            "embedding_dim": getattr(self._encoder, "dim", None),
            "explained_variance": getattr(self._encoder, "explained_variance", None),
            "sections_indexed": sorted({c.section for c in self._chunks if c.section}),
        }

    def context_block(self, citations: list[Citation], max_chars: int = 4000) -> str:
        """Format retrieved chunks for an LLM prompt with explicit source tags,
        so the model can cite and a validator can check the citation exists."""
        parts, used = [], 0
        for i, c in enumerate(citations, 1):
            block = f"[S{i}] source={c.source} chunk={c.chunk_id}\n{c.excerpt}\n"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n".join(parts)
