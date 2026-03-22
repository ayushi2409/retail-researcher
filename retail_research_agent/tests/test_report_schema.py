"""RetailReport parsing from markdown."""

from __future__ import annotations

from schemas.report import RetailReport, parse_and_validate_retail_report


def test_parse_well_formed_markdown() -> None:
    md = """
# India Retail 2026

## Summary
Omnichannel and quick commerce continue to expand in major metros.

## Key Insights
- Modern trade share is rising
- Kirana digitization is uneven but growing

## Market Trends
Premiumization in FMCG and fashion.

## Competitor Analysis
National chains compete with regional strongholds.

## Risks
Margin pressure and inventory volatility.

## Opportunities
Private labels and loyalty data monetization.

## Sources
- https://example.com/report-one
- https://example.org/statistics
"""
    report = parse_and_validate_retail_report(md)
    assert isinstance(report, RetailReport)
    assert "India" in report.title
    assert len(report.sources) >= 2


def test_parse_fallback_minimal() -> None:
    md = "# Short\n\nSome body with https://a.com and https://b.com extra text."
    report = parse_and_validate_retail_report(md)
    assert report.title
    assert len(report.sources) >= 1
