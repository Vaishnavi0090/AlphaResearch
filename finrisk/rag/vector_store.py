"""Vector store for dense retrieval.

Deliberately in-memory and dependency-free rather than FAISS/Chroma/pgvector.
The corpus here is hundreds to low thousands of chunks; a NumPy matrix
multiply searches that exhaustively in under a millisecond, which is *exact*
where an ANN index is approximate. Adding a vector database would introduce a
service dependency, a persistence format and an approximation error in
exchange for speed this workload does not need.

The `VectorStore` interface is kept narrow (`add`, `search`, `save`, `load`) so
swapping in FAISS later is a contained change — see `docs/RAG.md`.

Metadata filtering happens *before* scoring, not after. Filtering afterwards
silently returns fewer than `top_k` results, which then looks like a relevance
problem rather than a filter problem.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class InMemoryVectorStore:
    """Exact cosine-similarity search over unit-norm vectors."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._ids: list[str] = []
        self._meta: list[dict[str, Any]] = []
        self._id_pos: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def is_empty(self) -> bool:
        return len(self._ids) == 0

    def add(self, ids: list[str], vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        if len(ids) != len(vectors) or len(ids) != len(metadata):
            raise ValueError(
                f"length mismatch: {len(ids)} ids, {len(vectors)} vectors, "
                f"{len(metadata)} metadata"
            )
        if not ids:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != store dim {self.dim}")
        # Re-adding an existing id updates it rather than creating a duplicate
        # that would compete with itself in search results.
        keep_ids, keep_vecs, keep_meta = [], [], []
        for i, cid in enumerate(ids):
            pos = self._id_pos.get(cid)
            if pos is not None:
                self._vectors[pos] = vectors[i]
                self._meta[pos] = metadata[i]
            else:
                keep_ids.append(cid)
                keep_vecs.append(vectors[i])
                keep_meta.append(metadata[i])
        if keep_ids:
            start = len(self._ids)
            self._vectors = np.vstack([self._vectors, np.asarray(keep_vecs, dtype=np.float32)])
            self._ids.extend(keep_ids)
            self._meta.extend(keep_meta)
            for offset, cid in enumerate(keep_ids):
                self._id_pos[cid] = start + offset

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        where: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return (id, cosine similarity, metadata), best first."""
        # Dim is validated BEFORE the empty check: a mismatched query is a
        # programming error whether or not the store happens to hold vectors,
        # and returning [] would hide it until the store filled up.
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query dim {q.shape[0]} != store dim {self.dim}")
        if self.is_empty or top_k <= 0:
            return []

        # Filter first: post-filtering would quietly return fewer than top_k.
        if where is None:
            idx = np.arange(len(self._ids))
        else:
            idx = np.array([i for i, m in enumerate(self._meta) if where(m)], dtype=int)
            if idx.size == 0:
                return []

        sims = self._vectors[idx] @ q
        k = min(top_k, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k] if k < sims.shape[0] else np.arange(k)
        top = top[np.argsort(-sims[top])]
        return [(self._ids[idx[j]], float(sims[j]), self._meta[idx[j]]) for j in top]

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        pos = self._id_pos.get(chunk_id)
        return self._meta[pos] if pos is not None else None

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p.with_suffix(".npz"), vectors=self._vectors, ids=np.array(self._ids))
        p.with_suffix(".meta.json").write_text(json.dumps(self._meta, default=str))
        return p.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> InMemoryVectorStore:
        p = Path(path).with_suffix(".npz")
        data = np.load(p, allow_pickle=False)
        vectors = data["vectors"]
        store = cls(dim=int(vectors.shape[1]) if vectors.size else 0)
        store._vectors = vectors.astype(np.float32)
        store._ids = [str(x) for x in data["ids"]]
        store._meta = json.loads(Path(path).with_suffix(".meta.json").read_text())
        store._id_pos = {cid: i for i, cid in enumerate(store._ids)}
        return store


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> dict[str, float]:
    """Fuse ranked id lists by Reciprocal Rank Fusion.

        score(d) = sum_i  w_i / (k + rank_i(d))

    RRF is used instead of normalising and adding the raw scores because BM25
    scores are unbounded and corpus-dependent while cosine similarities sit in
    [-1, 1]. Any fixed normalisation between them is arbitrary and shifts as
    the corpus changes; RRF only uses *rank*, so it is scale-free and stable.

    k=60 is the value from the original RRF paper. Larger k flattens the
    contribution of top ranks.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return scores
