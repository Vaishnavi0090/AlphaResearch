"""Financial document loading for the RAG layer.

Supports plain text / markdown / HTML on disk. Each document keeps a stable
`doc_id` so retrieved chunks stay traceable to a source file, which is what
makes a generated research claim auditable.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
HTML_SUFFIXES = {".html", ".htm"}


@dataclass
class FinancialDocument:
    doc_id: str
    source: str
    ticker: str | None
    doc_type: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.text)


def _strip_html(raw: str) -> str:
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"&nbsp;?", " ", raw)
    raw = re.sub(r"&amp;", "&", raw)
    return re.sub(r"\s+", " ", raw).strip()


def infer_metadata(path: Path) -> tuple[str | None, str]:
    """Infer (ticker, doc_type) from a filename like `AAPL_10K_2023.md`.

    Convention over configuration: filenames carry the metadata so a user can
    drop filings into a folder and get correct citations with no config.
    """
    stem = path.stem
    parts = re.split(r"[_\-\.]", stem)
    # A ticker must START with a letter but may contain digits (BRK.B, SY00,
    # 7203.T). Requiring letters-only silently drops every alphanumeric ticker
    # and makes the ticker filter a no-op.
    ticker = (
        parts[0].upper() if parts and re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,7}", parts[0]) else None
    )
    lowered = stem.lower()
    for key, label in (
        ("10k", "10-K"),
        ("10-q", "10-Q"),
        ("10q", "10-Q"),
        ("8k", "8-K"),
        ("annual", "annual_report"),
        ("quarter", "quarterly_report"),
        ("transcript", "earnings_call"),
        ("earnings", "earnings_release"),
        ("press", "press_release"),
    ):
        if key in lowered:
            return ticker, label
    return ticker, "document"


def load_document(path: str | Path) -> FinancialDocument:
    p = Path(path)
    raw = p.read_text(errors="ignore")
    text = _strip_html(raw) if p.suffix.lower() in HTML_SUFFIXES else raw
    ticker, doc_type = infer_metadata(p)
    doc_id = hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:12]
    return FinancialDocument(
        doc_id=doc_id,
        source=p.name,
        ticker=ticker,
        doc_type=doc_type,
        text=text,
        metadata={"path": str(p), "bytes": p.stat().st_size},
    )


def load_corpus(directory: str | Path) -> list[FinancialDocument]:
    d = Path(directory)
    if not d.exists():
        logger.warning("document directory %s does not exist", d)
        return []
    docs: list[FinancialDocument] = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES | HTML_SUFFIXES:
            try:
                docs.append(load_document(p))
            except Exception as exc:
                logger.warning("failed to load %s: %s", p, exc)
    logger.info("loaded %d documents from %s", len(docs), d)
    return docs
