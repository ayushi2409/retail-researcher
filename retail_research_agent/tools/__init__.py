"""Tools for search, scraping, and persistence."""

from tools.web_scraper import scrape_url, scrape_urls
from tools.web_search import SearchResult, WebSearchService

__all__ = ["WebSearchService", "SearchResult", "scrape_url", "scrape_urls"]
