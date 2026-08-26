"""SEC filing ingestion: 10-K, 10-Q and 8-K.

Parses real EDGAR filings into `FinancialDocument`s with the metadata that
makes a citation auditable: accession number, CIK, form type, filing date,
period of report, and the Item section each passage came from.

Design constraints worth stating:

* **Offline-first.** Parsing works on local files with no network. Fetching
  from EDGAR is a separate, optional function. A pipeline that only works with
  internet access is not reproducible, and EDGAR is unreachable from this
  environment anyway (HTTP 403).
* **Filing date, not period end.** `filing_date` is when the document became
  public and is what point-in-time discipline requires. `period_of_report` is
  the fiscal period it describes. Confusing the two leaks roughly two months of
  hindsight — the same error the fundamentals pipeline guards against.
* **Section-aware.** Item 1A (Risk Factors) and Item 7 (MD&A) are extracted
  separately, because "the 10-K says so" is not traceability if you cannot say
  which part.

EDGAR requires a descriptive User-Agent with contact details; requests without
one are rejected. Set `SEC_USER_AGENT` (see `.env.example`).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finrisk.data.documents import FinancialDocument

logger = logging.getLogger(__name__)

FORM_TYPES = {"10-K", "10-Q", "8-K", "20-F", "40-F"}

# Item headings, in the order they appear in a filing. Matching is anchored to
# line starts because "item 1a" appears inside prose constantly.
_ITEM_PATTERNS: list[tuple[str, str, str]] = [
    ("10-K", r"^\s*item\s*1\s*[.\-—:]?\s*business\b", "Item 1 - Business"),
    ("10-K", r"^\s*item\s*1a\s*[.\-—:]?\s*risk\s*factors\b", "Item 1A - Risk Factors"),
    ("10-K", r"^\s*item\s*1b\s*[.\-—:]?", "Item 1B - Unresolved Staff Comments"),
    ("10-K", r"^\s*item\s*2\s*[.\-—:]?\s*propert", "Item 2 - Properties"),
    ("10-K", r"^\s*item\s*3\s*[.\-—:]?\s*legal", "Item 3 - Legal Proceedings"),
    ("10-K", r"^\s*item\s*5\s*[.\-—:]?\s*market\b", "Item 5 - Market for Common Equity"),
    ("10-K", r"^\s*item\s*7\s*[.\-—:]?\s*management", "Item 7 - MD&A"),
    ("10-K", r"^\s*item\s*7a\s*[.\-—:]?", "Item 7A - Market Risk"),
    ("10-K", r"^\s*item\s*8\s*[.\-—:]?\s*financial\s*statements", "Item 8 - Financial Statements"),
    ("10-K", r"^\s*item\s*9a\s*[.\-—:]?\s*controls", "Item 9A - Controls and Procedures"),
    ("10-Q", r"^\s*item\s*1\s*[.\-—:]?\s*financial\s*statements", "Item 1 - Financial Statements"),
    ("10-Q", r"^\s*item\s*2\s*[.\-—:]?\s*management", "Item 2 - MD&A"),
    ("10-Q", r"^\s*item\s*3\s*[.\-—:]?\s*quantitative", "Item 3 - Market Risk"),
    ("10-Q", r"^\s*item\s*1a\s*[.\-—:]?\s*risk\s*factors", "Item 1A - Risk Factors"),
    ("8-K", r"^\s*item\s*2\.02", "Item 2.02 - Results of Operations"),
    ("8-K", r"^\s*item\s*5\.02", "Item 5.02 - Departure of Directors"),
    ("8-K", r"^\s*item\s*7\.01", "Item 7.01 - Regulation FD Disclosure"),
    ("8-K", r"^\s*item\s*8\.01", "Item 8.01 - Other Events"),
]

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\xa0]+")
_BLANK_RE = re.compile(r"\n{3,}")


@dataclass
class FilingMetadata:
    """Everything needed to cite a filing precisely."""

    accession: str | None = None
    cik: str | None = None
    ticker: str | None = None
    company: str | None = None
    form_type: str = "10-K"
    filing_date: str | None = None  # when it became public (PIT-relevant)
    period_of_report: str | None = None  # the fiscal period described
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    source_path: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accession": self.accession,
            "cik": self.cik,
            "ticker": self.ticker,
            "company": self.company,
            "form_type": self.form_type,
            "filing_date": self.filing_date,
            "period_of_report": self.period_of_report,
            "fiscal_year": self.fiscal_year,
            "fiscal_period": self.fiscal_period,
        }


def html_to_text(raw: str) -> str:
    """Strip markup while preserving block structure.

    Deliberately regex-based rather than pulling in BeautifulSoup/lxml: EDGAR
    documents are enormous and mostly tables, a DOM parse is slow and memory
    hungry, and everything downstream needs is plain text with line breaks.
    """
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw)
    text = re.sub(r"(?i)<br[^>]*>", "\n", text)
    text = re.sub(r"(?i)</(p|div|tr|table|h[1-6]|li)>", "\n", text)
    text = re.sub(r"(?i)</t[dh]>", " ", text)
    text = _TAG_RE.sub(" ", text)
    for entity, char in (
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
        ("&#39;", "'"),
        ("&rsquo;", "'"),
        ("&mdash;", "—"),
    ):
        text = text.replace(entity, char)
    text = re.sub(r"&#\d+;", " ", text)
    text = _WS_RE.sub(" ", text)
    return _BLANK_RE.sub("\n\n", text).strip()


def parse_sec_header(raw: str) -> FilingMetadata:
    """Read the SEC-HEADER block present in full submission files."""
    meta = FilingMetadata()

    def grab(pattern: str) -> str | None:
        match = re.search(pattern, raw[:20000], re.I)
        return match.group(1).strip() if match else None

    meta.accession = grab(r"ACCESSION NUMBER:\s*([\d-]+)")
    meta.cik = grab(r"CENTRAL INDEX KEY:\s*(\d+)")
    meta.company = grab(r"COMPANY CONFORMED NAME:\s*(.+)")
    form = grab(r"CONFORMED SUBMISSION TYPE:\s*([\w-]+)")
    if form:
        meta.form_type = form.upper()
    filed = grab(r"FILED AS OF DATE:\s*(\d{8})")
    if filed:
        meta.filing_date = f"{filed[:4]}-{filed[4:6]}-{filed[6:]}"
    period = grab(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})")
    if period:
        meta.period_of_report = f"{period[:4]}-{period[4:6]}-{period[6:]}"
        meta.fiscal_year = int(period[:4])
        meta.fiscal_period = f"{period[:4]}Q{(int(period[4:6]) - 1) // 3 + 1}"
    return meta


def split_into_sections(text: str, form_type: str = "10-K") -> dict[str, str]:
    """Split filing text into Item sections.

    Returns section label -> text. Falls back to a single 'Full Document'
    entry when no headings are found, which is the honest outcome for an
    exhibit or a filing whose formatting defeats the patterns — better than
    silently returning nothing.
    """
    patterns = [(p, label) for form, p, label in _ITEM_PATTERNS if form == form_type]
    if not patterns:
        patterns = [(p, label) for _, p, label in _ITEM_PATTERNS]

    marks: list[tuple[int, str]] = []
    for pattern, label in patterns:
        for match in re.finditer(pattern, text, re.I | re.M):
            marks.append((match.start(), label))
    if not marks:
        return {"Full Document": text}

    marks.sort()
    # A filing repeats its Item headings in the table of contents. The real
    # section is the LAST occurrence, since the TOC sits at the front.
    last: dict[str, int] = {}
    for pos, label in marks:
        last[label] = pos
    ordered = sorted(((pos, label) for label, pos in last.items()))

    sections: dict[str, str] = {}
    for i, (pos, label) in enumerate(ordered):
        end = ordered[i + 1][0] if i + 1 < len(ordered) else len(text)
        body = text[pos:end].strip()
        if len(body) > 200:  # discard heading-only fragments
            sections[label] = body
    return sections or {"Full Document": text}


def parse_filing(
    raw: str,
    metadata: FilingMetadata | None = None,
    min_section_chars: int = 200,
) -> list[FinancialDocument]:
    """Turn one filing into per-section documents.

    One document per section rather than one per filing: a 10-K runs to
    hundreds of pages, and retrieving "the 10-K" is useless. Section-level
    documents let a citation point at Item 1A specifically.
    """
    meta = metadata or FilingMetadata()
    if "<SEC-HEADER>" in raw or "ACCESSION NUMBER:" in raw:
        parsed = parse_sec_header(raw)
        for key, value in parsed.as_dict().items():
            if value and not getattr(meta, key, None):
                setattr(meta, key, value)

    text = html_to_text(raw) if "<" in raw[:2000] else raw
    sections = split_into_sections(text, meta.form_type)

    docs: list[FinancialDocument] = []
    stem = (
        f"{meta.ticker or meta.cik or 'UNKNOWN'}_"
        f"{meta.form_type.replace('-', '')}_"
        f"{meta.fiscal_year or (meta.filing_date or '')[:4] or 'NA'}"
    )
    for label, body in sections.items():
        if len(body) < min_section_chars:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        doc = FinancialDocument(
            doc_id=f"{stem}:{slug}",
            source=f"{stem}.txt#{slug}",
            ticker=meta.ticker,
            doc_type=meta.form_type,
            text=body,
        )
        # Attached dynamically so chunking picks them up (see Chunk.metadata).
        doc.section = label  # type: ignore[attr-defined]
        doc.filing_date = meta.filing_date  # type: ignore[attr-defined]
        doc.fiscal_period = meta.fiscal_period  # type: ignore[attr-defined]
        doc.accession = meta.accession  # type: ignore[attr-defined]
        docs.append(doc)

    logger.info(
        "parsed %s %s (%s): %d sections",
        meta.ticker or meta.cik,
        meta.form_type,
        meta.filing_date,
        len(docs),
    )
    return docs


def load_filings_from_dir(
    directory: str | Path, ticker_map: dict[str, str] | None = None
) -> list[FinancialDocument]:
    """Parse every filing in a directory.

    Filenames of the form `TICKER_FORM_YEAR.*` (e.g. `AAPL_10K_2023.htm`)
    supply metadata when the file has no SEC header.
    """
    d = Path(directory)
    if not d.exists():
        logger.info("filings directory %s does not exist", d)
        return []

    docs: list[FinancialDocument] = []
    for path in sorted(d.glob("*")):
        if path.suffix.lower() not in {".txt", ".htm", ".html", ".xml"}:
            continue
        meta = FilingMetadata(source_path=str(path))
        parts = path.stem.split("_")
        if parts:
            meta.ticker = (ticker_map or {}).get(parts[0], parts[0]).upper()
        if len(parts) > 1:
            candidate = parts[1].upper()
            normalised = (
                f"{candidate[:2]}-{candidate[2:]}" if candidate.startswith("10") else candidate
            )
            if normalised in FORM_TYPES:
                meta.form_type = normalised
        if len(parts) > 2 and parts[2][:4].isdigit():
            meta.fiscal_year = int(parts[2][:4])
        try:
            docs.extend(parse_filing(path.read_text(errors="ignore"), meta))
        except Exception as exc:
            logger.warning("failed to parse %s: %s", path.name, exc)
    logger.info("loaded %d section documents from %s", len(docs), d)
    return docs


def fetch_filing_index(
    cik: str, form_type: str = "10-K", limit: int = 5, user_agent: str | None = None
) -> list[FilingMetadata]:  # pragma: no cover - network
    """List recent filings for a CIK via EDGAR's submissions API.

    Optional and network-dependent. EDGAR is unreachable in this environment
    (HTTP 403), so the offline path above is the default and the tested one.
    """
    import json
    import os
    import urllib.request

    agent = user_agent or os.environ.get("SEC_USER_AGENT")
    if not agent:
        raise RuntimeError(
            "SEC requires a descriptive User-Agent with contact details. "
            "Set SEC_USER_AGENT (see .env.example)."
        )
    cik_padded = str(cik).zfill(10)
    req = urllib.request.Request(
        f"https://data.sec.gov/submissions/CIK{cik_padded}.json",
        headers={"User-Agent": agent},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    recent = data.get("filings", {}).get("recent", {})
    out: list[FilingMetadata] = []
    for i, form in enumerate(recent.get("form", [])):
        if form != form_type:
            continue
        out.append(
            FilingMetadata(
                accession=recent["accessionNumber"][i],
                cik=cik_padded,
                company=data.get("name"),
                form_type=form,
                filing_date=recent["filingDate"][i],
                period_of_report=recent.get("reportDate", [None] * (i + 1))[i],
                extras={"primary_document": recent["primaryDocument"][i]},
            )
        )
        if len(out) >= limit:
            break
    return out
