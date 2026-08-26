"""Parquet/pickle cache with retry helpers.

Financial APIs are rate-limited and flaky. Every network call in this project
goes through `retry` and every fetched frame through `DataCache` so that a
research loop does not hammer a provider.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")


def cache_key(*parts: Any) -> str:
    """Stable short hash of the arguments identifying a dataset."""
    blob = json.dumps([str(p) for p in parts], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def retry(
    fn: Callable[[], T],
    attempts: int = 4,
    base_delay: float = 0.6,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Exponential-backoff retry. Used for every external data call."""
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return fn()
        except exceptions as exc:
            last = exc
            if i == attempts - 1:
                break
            delay = base_delay * (backoff**i)
            logger.warning(
                "attempt %d/%d failed (%s); retrying in %.1fs",
                i + 1,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last is not None
    raise last


class DataCache:
    """Filesystem cache. Frames go to parquet when pyarrow is available,
    otherwise pickle. Objects go to pickle."""

    def __init__(self, root: str | Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    # -- frames --------------------------------------------------------
    def _frame_path(self, key: str) -> Path:
        return self.root / f"{key}.pkl"

    def get_frame(self, key: str) -> pd.DataFrame | None:
        if not self.enabled:
            return None
        p = self._frame_path(key)
        if p.exists():
            self.hits += 1
            return pd.read_pickle(p)
        self.misses += 1
        return None

    def put_frame(self, key: str, df: pd.DataFrame) -> None:
        if not self.enabled:
            return
        df.to_pickle(self._frame_path(key))

    def frame(self, key: str, builder: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        cached = self.get_frame(key)
        if cached is not None:
            return cached
        built = builder()
        self.put_frame(key, built)
        return built

    # -- arbitrary objects --------------------------------------------
    def object(self, key: str, builder: Callable[[], Any]) -> Any:
        import pickle

        p = self.root / f"{key}.obj.pkl"
        if self.enabled and p.exists():
            self.hits += 1
            with p.open("rb") as fh:
                return pickle.load(fh)
        self.misses += 1
        obj = builder()
        if self.enabled:
            with p.open("wb") as fh:
                pickle.dump(obj, fh)
        return obj

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

    def clear(self) -> int:
        n = 0
        if self.root.exists():
            for p in self.root.glob("*.pkl"):
                p.unlink()
                n += 1
        return n
