"""Application observability (Phase 30).

Deliberately dependency-free. LangSmith tracing is *supported* (set
`LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` and LangGraph emits traces
automatically) but never required: a research platform that cannot run without
an observability SaaS is not reproducible.

What is tracked: stage timings, tool call counts and latency, cache hit rates,
LLM calls and token usage where the backend reports it, and errors.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Span:
    name: str
    duration_ms: float
    ok: bool = True
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Telemetry:
    """Collects spans and counters for one session/run."""

    spans: list[Span] = field(default_factory=list)
    counters: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str, **meta: Any) -> Iterator[dict[str, Any]]:
        t0 = time.perf_counter()
        payload: dict[str, Any] = dict(meta)
        try:
            yield payload
        except Exception as exc:
            self.spans.append(
                Span(
                    name,
                    (time.perf_counter() - t0) * 1000,
                    False,
                    f"{type(exc).__name__}: {exc}",
                    payload,
                )
            )
            self.incr("errors")
            raise
        else:
            self.spans.append(Span(name, (time.perf_counter() - t0) * 1000, True, None, payload))

    def incr(self, key: str, by: float = 1.0) -> None:
        self.counters[key] = self.counters.get(key, 0.0) + by

    def record_tools(self, stats: dict[str, Any]) -> None:
        self.incr("tool_calls", float(stats.get("n_calls", 0)))
        self.incr("tool_failures", float(stats.get("n_failed", 0)))
        self.incr("tool_latency_ms", float(stats.get("total_latency_ms", 0.0)))

    def record_llm(self, client: Any) -> None:
        self.incr("llm_calls", float(getattr(client, "n_calls", 0)))
        self.incr("llm_input_tokens", float(getattr(client, "total_input_tokens", 0) or 0))
        self.incr("llm_output_tokens", float(getattr(client, "total_output_tokens", 0) or 0))

    def record_cache(self, stats: dict[str, int]) -> None:
        self.incr("cache_hits", float(stats.get("hits", 0)))
        self.incr("cache_misses", float(stats.get("misses", 0)))

    @property
    def total_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans)

    def summary(self) -> dict[str, Any]:
        hits = self.counters.get("cache_hits", 0.0)
        misses = self.counters.get("cache_misses", 0.0)
        return {
            "total_ms": round(self.total_ms, 1),
            "n_spans": len(self.spans),
            "slowest": [
                {"name": name, "ms": ms}
                for name, ms in sorted(
                    ((s.name, round(s.duration_ms, 1)) for s in self.spans),
                    key=lambda pair: -pair[1],
                )[:8]
            ],
            "failed_spans": [s.name for s in self.spans if not s.ok],
            "counters": {k: round(v, 2) for k, v in sorted(self.counters.items())},
            "cache_hit_rate": round(hits / (hits + misses), 4) if (hits + misses) else None,
            "langsmith_enabled": os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true",
        }

    def render(self) -> str:
        s = self.summary()
        lines = [
            "Observability",
            "=" * 56,
            f"Total: {s['total_ms']:.0f} ms across {s['n_spans']} spans",
        ]
        for sp in s["slowest"]:
            lines.append(f"  {sp['name']:<34}{sp['ms']:>10.1f} ms")
        if s["counters"]:
            lines += ["-" * 56, "Counters:"]
            for k, v in s["counters"].items():
                lines.append(f"  {k:<34}{v:>10.2f}")
        if s["cache_hit_rate"] is not None:
            lines.append(f"  {'cache_hit_rate':<34}{s['cache_hit_rate']:>10.2%}")
        if s["failed_spans"]:
            lines.append(f"FAILED: {', '.join(s['failed_spans'])}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.summary(), indent=2))
        return p


TELEMETRY = Telemetry()
