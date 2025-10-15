"""DuckDuckGo search and content extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ddgs import DDGS
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

CHARS_PER_TOKEN = 4

_PDF_PIPELINE_OPTIONS = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=False,
    images_scale=1.0,
)

_DOCUMENT_CONVERTER = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_PDF_PIPELINE_OPTIONS)
    }
)


def build_search_query(
    base_query: str,
    sites: Optional[str] = None,
    inurls: Optional[str] = None,
) -> str:
    """Builds search query with site: and inurl: operators."""
    parts = [base_query]

    if sites:
        site_list = [s.strip() for s in sites.split(",") if s.strip()]
        if len(site_list) == 1:
            parts.append(f"site:{site_list[0]}")
        elif len(site_list) > 1:
            site_query = " OR ".join(f"site:{s}" for s in site_list)
            parts.append(f"({site_query})")

    if inurls:
        inurl_list = [u.strip() for u in inurls.split(",") if u.strip()]
        if len(inurl_list) == 1:
            parts.append(f"inurl:{inurl_list[0]}")
        elif len(inurl_list) > 1:
            inurl_query = " OR ".join(f"inurl:{u}" for u in inurl_list)
            parts.append(f"({inurl_query})")

    return " ".join(parts)


def strip_thinking_tokens(text: str) -> str:
    """Strips <think> blocks from reasoning models like Lucy."""

    if not text:
        return text

    cleaned = text
    while "<think>" in cleaned and "</think>" in cleaned:
        start = cleaned.find("<think>")
        end = cleaned.find("</think>") + len("</think>")
        cleaned = cleaned[:start] + cleaned[end:]
    return cleaned


def deduplicate_and_format_sources(
    search_response: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    max_tokens_per_source: int,
    fetch_full_page: bool,
) -> str:
    """Takes search results, removes duplicates by URL, and formats for the LLM."""

    if isinstance(search_response, dict):
        sources_list = search_response.get("results", [])
    elif isinstance(search_response, list):
        sources_list = []
        for entry in search_response:
            if isinstance(entry, dict) and "results" in entry:
                sources_list.extend(entry["results"])
            else:
                sources_list.extend(entry)
    else:
        raise ValueError("search_response must be a dict or list of dicts")

    by_url: dict[str, Dict[str, Any]] = {}
    for source in sources_list:
        url = source.get("url")
        if not url:
            continue
        if url not in by_url:
            by_url[url] = source

    formatted = ["Sources:\n"]
    for index, source in enumerate(by_url.values(), start=1):
        formatted.append(f"Source: {source.get('title', 'Untitled')}\n===\n")
        formatted.append(f"URL: {source.get('url', '<missing url>')}\n===\n")
        formatted.append(
            f"Most relevant content from source: {source.get('content', '')}\n===\n"
        )
        if fetch_full_page:
            char_limit = max_tokens_per_source * CHARS_PER_TOKEN
            raw_content = source.get("raw_content") or ""
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted.append(
                f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
            )
    return "".join(formatted).strip()


def format_sources(search_results: Dict[str, Any]) -> str:
    """Creates a bulleted list of sources with titles and URLs."""

    results = []
    for item in search_results.get("results", []):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        results.append(f"* {title} : {url}")
    return "\n".join(results)


def fetch_raw_content(url: str) -> Optional[str]:
    """Fetches full page content and converts it to markdown using Docling."""

    fetch_url = url
    if "www.reddit.com" in url:
        fetch_url = url.replace("www.reddit.com", "old.reddit.com")

    try:
        conversion = _DOCUMENT_CONVERTER.convert(fetch_url)
        document = getattr(conversion, "document", None)
        if not document:
            return None
        return document.export_to_markdown()
    except Exception as exc:
        print(f"Warning: Docling failed for {fetch_url}: {exc}")
        return None


def duckduckgo_search(
    query: str,
    *,
    max_results: int,
    fetch_full_page: bool,
    region: str,
    sites: Optional[str] = None,
    inurls: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """DuckDuckGo search with optional site and inurl filtering."""
    results: List[Dict[str, Any]] = []
    enhanced_query = build_search_query(query, sites, inurls)

    try:
        with DDGS() as client:
            ddgs_results = list(
                client.text(
                    enhanced_query,
                    max_results=max_results,
                    region=region or "us-en",
                    safesearch="off",
                )
            )
    except Exception as exc:
        print(f"Error in DuckDuckGo search: {exc}")
        return {"results": results}

    for item in ddgs_results:
        url = item.get("href")
        title = item.get("title")
        content = item.get("body")
        if not all([url, title, content]):
            continue
        raw_content = content
        if fetch_full_page:
            raw_content = fetch_raw_content(url) or content
        results.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "raw_content": raw_content,
            }
        )
    return {"results": results}
