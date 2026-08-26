"""SEC filing ingestion: header parsing, section splitting, provenance.

The tests that matter here are the ones guarding point-in-time discipline
(filing date is not period end) and the table-of-contents trap — a 10-K lists
every Item heading in its TOC before the real sections, so naive splitting
returns the table of contents as the risk-factors section.
"""

from __future__ import annotations

import pytest

from finrisk.config.settings import RAGConfig
from finrisk.data.sec_filings import (
    FORM_TYPES,
    FilingMetadata,
    html_to_text,
    load_filings_from_dir,
    parse_filing,
    parse_sec_header,
    split_into_sections,
)
from finrisk.rag.retriever import DocumentRetriever

HEADER = """<SEC-HEADER>
ACCESSION NUMBER:\t\t0000320193-23-000106
CONFORMED SUBMISSION TYPE:\t10-K
CONFORMED PERIOD OF REPORT:\t20230930
FILED AS OF DATE:\t\t20231103
COMPANY CONFORMED NAME:\tApple Inc.
CENTRAL INDEX KEY:\t\t0000320193
</SEC-HEADER>
"""


def _section(heading: str, n: int = 12) -> str:
    return (
        f"<p>{heading}</p><p>"
        + (f"Detailed disclosure about operations and outlook sentence {n}. " * 12)
        + "</p>"
    )


def _filing(header: bool = True) -> str:
    toc = "<p>TABLE OF CONTENTS</p><p>Item 1A. Risk Factors ..... 5</p><p>Item 7. Management ..... 30</p>"
    body = (
        _section("Item 1. Business", 1)
        + _section("Item 1A. Risk Factors", 2)
        + _section("Item 7. Management's Discussion and Analysis", 3)
        + _section("Item 8. Financial Statements", 4)
    )
    return (HEADER if header else "") + f"<html><body>{toc}{body}</body></html>"


# ---------------------------------------------------------------- html
def test_html_to_text_strips_markup_and_entities():
    out = html_to_text("<p>Revenue &amp; margin rose 3&#37;.</p><script>evil()</script>")
    assert "<" not in out and ">" not in out
    assert "&" in out and "evil()" not in out


def test_html_to_text_preserves_block_breaks():
    out = html_to_text("<p>First block.</p><p>Second block.</p>")
    assert "\n" in out
    assert "First block." in out and "Second block." in out


def test_html_to_text_handles_plain_text():
    assert html_to_text("Just plain text.") == "Just plain text."


# ---------------------------------------------------------------- header
def test_header_parsing_extracts_all_metadata():
    m = parse_sec_header(HEADER)
    assert m.accession == "0000320193-23-000106"
    assert m.cik == "0000320193"
    assert m.company == "Apple Inc."
    assert m.form_type == "10-K"
    assert m.fiscal_year == 2023
    assert m.fiscal_period == "2023Q3"


def test_filing_date_differs_from_period_end():
    """Point-in-time discipline: the filing became public over a month AFTER
    the period it describes. Treating period end as the availability date
    leaks that gap into any factor built from it."""
    m = parse_sec_header(HEADER)
    assert m.period_of_report == "2023-09-30"
    assert m.filing_date == "2023-11-03"
    assert m.filing_date > m.period_of_report


def test_header_parsing_on_missing_header_is_safe():
    m = parse_sec_header("<html><body>No header here.</body></html>")
    assert m.accession is None
    assert m.form_type == "10-K"  # documented default


# ---------------------------------------------------------------- sections
def test_split_extracts_each_item_section():
    sections = split_into_sections(html_to_text(_filing()), "10-K")
    assert "Item 1A - Risk Factors" in sections
    assert "Item 7 - MD&A" in sections
    assert "Item 8 - Financial Statements" in sections


def test_table_of_contents_is_not_mistaken_for_the_section():
    """A 10-K lists every Item heading in its TOC. Taking the FIRST match
    returns the table of contents instead of the disclosure."""
    sections = split_into_sections(html_to_text(_filing()), "10-K")
    risk = sections["Item 1A - Risk Factors"]
    assert "Detailed disclosure" in risk
    assert "....." not in risk


def test_split_falls_back_to_full_document():
    sections = split_into_sections("Plain narrative with no item headings. " * 20)
    assert list(sections) == ["Full Document"]


def test_heading_only_fragments_are_discarded():
    """Headings with no body are not returned as sections. When that leaves
    nothing, the parser falls back to the whole document rather than returning
    an empty result — losing the text entirely would be worse."""
    sections = split_into_sections("Item 1A. Risk Factors\nItem 7. Management", "10-K")
    assert list(sections) == ["Full Document"]
    assert "Item 1A" in sections["Full Document"]


def test_real_sections_survive_the_length_filter():
    sections = split_into_sections(html_to_text(_filing()), "10-K")
    assert "Full Document" not in sections
    assert all(len(v) > 200 for v in sections.values())


# ---------------------------------------------------------------- documents
def test_parse_filing_produces_one_document_per_section():
    docs = parse_filing(_filing(), FilingMetadata(ticker="AAPL"))
    assert len(docs) == 4
    assert all(d.ticker == "AAPL" for d in docs)
    assert len({d.doc_id for d in docs}) == 4


def test_documents_carry_provenance():
    docs = parse_filing(_filing(), FilingMetadata(ticker="AAPL"))
    doc = next(d for d in docs if d.section == "Item 1A - Risk Factors")
    assert doc.filing_date == "2023-11-03"
    assert doc.fiscal_period == "2023Q3"
    assert doc.accession == "0000320193-23-000106"


def test_metadata_from_filing_overrides_missing_fields():
    docs = parse_filing(_filing(header=False), FilingMetadata(ticker="AAPL", fiscal_year=2023))
    assert docs
    assert all(d.doc_id.startswith("AAPL_10K_2023") for d in docs)


@pytest.mark.parametrize(
    "form,heading,expected",
    [
        ("10-Q", "Item 2. Management's Discussion and Analysis", "Item 2 - MD&A"),
        ("10-Q", "Item 1. Financial Statements", "Item 1 - Financial Statements"),
        ("8-K", "Item 2.02 Results of Operations", "Item 2.02 - Results of Operations"),
        ("8-K", "Item 5.02 Departure of Directors", "Item 5.02 - Departure of Directors"),
    ],
)
def test_other_form_types_are_supported(form, heading, expected):
    docs = parse_filing(_section(heading), FilingMetadata(ticker="X", form_type=form))
    assert [d.section for d in docs] == [expected]


def test_all_declared_form_types_are_recognised():
    assert {"10-K", "10-Q", "8-K"} <= FORM_TYPES


# ---------------------------------------------------------------- directory
def test_load_filings_from_directory(tmp_path):
    (tmp_path / "AAPL_10K_2023.htm").write_text(_filing())
    (tmp_path / "MSFT_10Q_2024.txt").write_text(
        _section("Item 2. Management's Discussion and Analysis")
    )
    (tmp_path / "ignore.pdf").write_text("not parsed")
    docs = load_filings_from_dir(tmp_path)
    assert {d.ticker for d in docs} == {"AAPL", "MSFT"}
    assert {d.doc_type for d in docs} == {"10-K", "10-Q"}


def test_missing_directory_is_safe(tmp_path):
    assert load_filings_from_dir(tmp_path / "nope") == []


def test_one_unparseable_file_does_not_stop_ingestion(tmp_path):
    (tmp_path / "GOOD_10K_2023.txt").write_text(_filing())
    (tmp_path / "BAD_10K_2023.txt").write_bytes(b"\xff\xfe\x00bad")
    docs = load_filings_from_dir(tmp_path)
    assert any(d.ticker == "GOOD" for d in docs)


# ---------------------------------------------------------------- integration
def test_parsed_filings_are_retrievable_with_section_provenance():
    docs = parse_filing(_filing(), FilingMetadata(ticker="AAPL"))
    r = DocumentRetriever(RAGConfig(embedding_dim=4))
    r.add_documents(docs)
    hits = r.retrieve("disclosure about operations", 3)
    assert hits
    top = hits[0]
    assert top.section and top.filing_date == "2023-11-03"
    assert top.fiscal_period == "2023Q3"


def test_section_filter_works_on_real_filing_structure():
    docs = parse_filing(_filing(), FilingMetadata(ticker="AAPL"))
    r = DocumentRetriever(RAGConfig(embedding_dim=4))
    r.add_documents(docs)
    hits = r.retrieve("operations outlook", 5, section="Risk Factors")
    assert hits
    assert all("Risk Factors" in (h.section or "") for h in hits)
