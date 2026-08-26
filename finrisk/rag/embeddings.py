"""Dense embeddings for hybrid retrieval.

BM25 matches words. Dense retrieval matches *meaning* — the point is to find a
passage about "margin compression" when the query says "profitability
pressure", which lexical search cannot do.

Two encoders behind one protocol:

* `LSAEncoder` (default) — TF-IDF followed by truncated SVD, i.e. Latent
  Semantic Analysis. Genuine dense vectors: terms that co-occur across the
  corpus collapse onto shared latent dimensions, so semantically related
  passages land near each other even with no shared vocabulary. It is fitted on
  the corpus itself, needs no downloads, is deterministic, and runs in
  milliseconds.

* `SentenceTransformerEncoder` (optional) — a pretrained bi-encoder, stronger
  because it carries semantics learned from a far larger corpus.

Why LSA is the default rather than a pretrained model: model weights are not
reachable in this environment (huggingface.co is blocked), and a retrieval
layer that silently fails without network access is worse than one that is
weaker but always works. The trade-off is real and is measured rather than
asserted — `tests/test_hybrid_rag.py` checks that dense retrieval finds
paraphrases BM25 misses, and `docs/RAG.md` reports where it does not help.

A corpus-fitted encoder has one genuine limitation worth stating: its semantics
come only from co-occurrence within the indexed documents, so it generalises
less well than a pretrained model and needs refitting when the corpus changes
materially.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class DenseEncoder(Protocol):
    """Anything that turns text into unit-norm vectors."""

    name: str
    dim: int

    def fit(self, corpus: list[str]) -> DenseEncoder: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    """Unit-norm rows so dot product == cosine similarity."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


class LSAEncoder:
    """TF-IDF + truncated SVD. Offline, deterministic, corpus-fitted."""

    name = "lsa"

    def __init__(
        self,
        dim: int = 128,
        min_df: int = 1,
        max_df: float = 0.85,
        ngram_range: tuple[int, int] = (1, 2),
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.seed = seed
        # Typed as Any because they are built lazily in fit(); annotating them
        # as Optional would force a None-check on every use even though encode()
        # already guards with an explicit RuntimeError.
        self._vectorizer: Any = None
        self._svd: Any = None
        self._fitted = False

    def fit(self, corpus: list[str]) -> LSAEncoder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not corpus:
            logger.warning("LSAEncoder.fit called with an empty corpus")
            return self

        # sublinear_tf dampens repeated terms the same way BM25 saturation
        # does; without it a boilerplate-heavy filing dominates its own vector.
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            min_df=min(self.min_df, max(len(corpus) // 50, 1)),
            max_df=self.max_df if len(corpus) > 10 else 1.0,
            ngram_range=self.ngram_range,
            token_pattern=r"(?u)\b\w[\w.%-]*\b",  # keep 3.2%, 12.5, Q1-2023
            strip_accents="unicode",
        )
        X = self._vectorizer.fit_transform(corpus)

        # SVD cannot produce more components than min(n_docs, n_features) - 1.
        n_components = max(min(self.dim, min(X.shape) - 1), 2)
        if n_components < self.dim:
            logger.info("LSA dim reduced %d -> %d (corpus too small)", self.dim, n_components)
        self._svd = TruncatedSVD(
            n_components=n_components, random_state=self.seed, algorithm="randomized"
        )
        # A degenerate corpus (identical or near-identical documents) has zero
        # total variance, and sklearn divides by it when computing the
        # explained-variance ratio. The fit itself is still valid, so the
        # warning is suppressed here and `explained_variance` reports 0.0
        # rather than NaN.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self._svd.fit(X)
        self.dim = n_components
        self._fitted = True
        logger.info(
            "LSA fitted: %d docs, %d terms, %d dims, %.1f%% variance retained",
            X.shape[0],
            X.shape[1],
            self.dim,
            100.0 * self.explained_variance,
        )
        return self

    @property
    def explained_variance(self) -> float:
        if not self._fitted:
            return 0.0
        total = float(np.nansum(self._svd.explained_variance_ratio_))
        return total if np.isfinite(total) else 0.0

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LSAEncoder.encode called before fit()")
        if not texts:
            return np.zeros((0, self.dim), dtype=float)
        return _l2_normalize(self._svd.transform(self._vectorizer.transform(texts)).astype(float))


class HashingEncoder:
    """Dependency-free fallback: signed hashing of token n-grams.

    Not semantic — it is a random projection of the bag of words, so it behaves
    like a compressed lexical index. Present only so the dense path degrades to
    *something* deterministic if scikit-learn is unavailable, and named
    honestly so nobody mistakes it for a semantic model.
    """

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def fit(self, corpus: list[str]) -> HashingEncoder:
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=float)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _l2_normalize(out)


class SentenceTransformerEncoder:  # pragma: no cover - needs model weights
    """Pretrained bi-encoder. Used when the package and weights are present."""

    name = "sentence-transformer"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.model_name = model_name

    def fit(self, corpus: list[str]) -> SentenceTransformerEncoder:
        return self  # pretrained: nothing to fit

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=float)
        return _l2_normalize(
            np.asarray(self._model.encode(texts, show_progress_bar=False), dtype=float)
        )


def build_encoder(kind: str = "auto", dim: int = 128, seed: int = 0) -> DenseEncoder:
    """Select an encoder, preferring the strongest one actually available.

    'auto' tries sentence-transformers, falls back to LSA, then to hashing.
    Each fallback is logged so the active encoder is never a mystery.
    """
    if kind in {"auto", "sentence-transformer"}:
        try:
            enc = SentenceTransformerEncoder()
            logger.info("dense encoder: sentence-transformers")
            return enc
        except Exception as exc:
            if kind == "sentence-transformer":
                logger.warning("sentence-transformers unavailable (%s); using LSA", exc)
            else:
                logger.info("sentence-transformers unavailable; using LSA")

    if kind in {"auto", "lsa"}:
        try:
            import sklearn  # noqa: F401

            return LSAEncoder(dim=dim, seed=seed)
        except ImportError:
            logger.warning("scikit-learn unavailable; falling back to hashing encoder")

    return HashingEncoder(dim=max(dim, 256))
