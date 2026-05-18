"""Scrape brand homepages via trafilatura with polite headers + caching.

Per brand we extract: ``<title>``, ``<meta name="description">``, the
first 3 headings (h1/h2), and the first 5 paragraphs of meaningful body
text (via ``trafilatura.extract``). The four pieces are concatenated
into one document.

References:
- trafilatura: https://trafilatura.readthedocs.io/
- robots.txt respect is on by default in trafilatura.fetch_url.

CLI:
    python -m src.scrape              # scrape all brands in brands.yaml
    python -m src.scrape --only Lidl  # one brand
    python -m src.scrape --no-cache   # force re-fetch
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import trafilatura
import yaml
from bs4 import BeautifulSoup  # ships with trafilatura
from trafilatura.settings import use_config

from src import config

logger = logging.getLogger(__name__)

# Configure trafilatura with polite UA + timeout
_TRAFILATURA_CFG = use_config()
_TRAFILATURA_CFG.set("DEFAULT", "USER_AGENTS", config.USER_AGENT)
_TRAFILATURA_CFG.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(int(config.REQUEST_TIMEOUT_S)))


def _slug(s: str) -> str:
    """Conservative kebab slug for cache filenames."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:48]


def _cache_path(brand: str) -> Path:
    """Where do we cache the raw HTML for this brand?"""
    return config.RAW_HTML_DIR / f"{_slug(brand)}.html"


def fetch_html(url: str, brand: str, use_cache: bool = True) -> str | None:
    """Fetch the URL's HTML, using disk cache if present.

    Args:
        url: brand homepage URL.
        brand: brand name (for the cache filename).
        use_cache: if False, force re-fetch.

    Returns:
        Raw HTML string, or None if fetch failed.
    """
    cache = _cache_path(brand)
    if use_cache and cache.exists():
        logger.debug("[%s] cached %s", brand, cache.name)
        return cache.read_text(encoding="utf-8", errors="replace")
    logger.info("[%s] fetching %s", brand, url)
    html = trafilatura.fetch_url(url, config=_TRAFILATURA_CFG)
    if html is None:
        logger.warning("[%s] fetch returned None", brand)
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(html, encoding="utf-8")
    return html


def _take_first_n_paragraphs(extracted_text: str, n: int) -> list[str]:
    """trafilatura.extract returns text split by blank lines; take first N
    non-trivial paragraphs."""
    if not extracted_text:
        return []
    raw = [p.strip() for p in re.split(r"\n\s*\n", extracted_text)]
    keep = [p for p in raw if len(p) >= 30]
    return keep[:n]


# Substrings that mark anti-bot interstitial pages (Cloudflare, AWS WAF,
# Akamai). If we find these in what trafilatura extracted, we treat the
# scrape as failed — the real homepage text is behind a JS challenge that
# trafilatura can't solve.
ANTIBOT_MARKERS: tuple[str, ...] = (
    "you're not a bot",
    "verifying you are human",
    "verifying you're human",
    "just a moment...",
    "ddos protection by cloudflare",
    "attention required",
    "enable javascript to continue",
    "please wait while we verify",
    "checking your browser",
)


def _is_antibot_junk(text: str) -> bool:
    lowered = text[:800].lower()
    return any(m in lowered for m in ANTIBOT_MARKERS)


def extract_brand_text(brand: str, url: str, html: str) -> str:
    """Combine title + meta + first 2 brand-positioning headings.

    DELIBERATELY narrow. We previously also pulled the first 5 paragraphs
    via trafilatura, but on Czech retail/ecommerce homepages those paragraphs
    are dominated by promo carousels ("Akce týdne", "Aktuální novinky",
    placeholder text) that share generic commercial vocabulary across
    brands. The shared vocabulary inflated cosine similarity and produced
    nonsensical "nearest brand" matches in the opportunity scan (Lidl as
    the top match for travel queries, etc.).

    Title + meta + first 2 headings is the part of a homepage that
    typically carries the brand's strategic positioning ("Lidl. To se
    vyplatí.", "Stojí za to jíst lépe.") rather than this week's promo.
    """
    soup = BeautifulSoup(html, "html.parser")
    parts: list[str] = []

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    if title:
        parts.append(title)

    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        parts.append(meta["content"].strip())

    headings: list[str] = []
    for tag in soup.find_all(["h1", "h2"], limit=8):
        text = tag.get_text(" ", strip=True)
        if not text or len(text) > 200:
            continue
        # Skip generic carousel / nav labels — they're not positioning
        lowered = text.lower()
        if any(w in lowered for w in ("novinky", "aktuální", "magazín", "akce týdne",
                                      "letáky", "recepty", "kategorie", "menu",
                                      "vítejte", "nejprodávanější")):
            continue
        headings.append(text)
        if len(headings) >= 2:
            break
    parts.extend(headings)

    # Add the first 1-2 body paragraphs that look like positioning (not promo).
    # Skip paragraphs that read as promo copy (akce, sleva, leták keywords)
    # or as cookie/JS boilerplate. Cap total length so a heavy-text site
    # like Avast doesn't outweigh a tight one like Mattoni.
    body = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_recall=False,
        config=_TRAFILATURA_CFG,
    )
    promo_markers = ("sleva", "akce týdne", "leták", "akční nabíd", "cookies",
                     "javascript", "přihlas se k odběru", "newsletter")
    kept_paragraphs: list[str] = []
    for p in _take_first_n_paragraphs(body or "", n=8):
        if len(kept_paragraphs) >= 2:
            break
        lowered = p.lower()
        if any(m in lowered for m in promo_markers):
            continue
        kept_paragraphs.append(p)
    parts.extend(kept_paragraphs)

    combined = "\n\n".join(p for p in parts if p)
    if len(combined) > 600:
        combined = combined[:600].rsplit(" ", 1)[0]
    if _is_antibot_junk(combined):
        logger.warning("[%s] anti-bot interstitial detected; treating as failed scrape", brand)
        return ""
    logger.debug("[%s] extracted %d chars", brand, len(combined))
    return combined


def scrape_brand(brand_config: dict, use_cache: bool = True) -> dict:
    """One brand → one row dict, ready for the parquet."""
    brand = brand_config["brand"]
    url = brand_config["url"]
    sector = brand_config["sector"]
    html = fetch_html(url, brand, use_cache=use_cache)
    text = ""
    if html:
        try:
            text = extract_brand_text(brand, url, html)
        except Exception as exc:
            logger.warning("[%s] extract failed: %s", brand, exc)
    if len(text) < config.MIN_BRAND_TEXT_LEN:
        logger.warning("[%s] text too short (%d chars); dropping", brand, len(text))
        text = ""
    return {
        "brand": brand,
        "sector": sector,
        "url": url,
        "text": text,
        "text_len": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12] if text else "",
        "scraped_at": datetime.now(UTC).isoformat(),
    }


def scrape_all(use_cache: bool = True, only: str | None = None) -> pd.DataFrame:
    raw = yaml.safe_load(config.BRANDS_YAML.read_text(encoding="utf-8"))
    rows: list[dict] = []
    first = True
    for bc in raw["brands"]:
        if only and bc["brand"] != only:
            continue
        if not first and not (use_cache and _cache_path(bc["brand"]).exists()):
            time.sleep(config.INTER_REQUEST_SLEEP_S)
        first = False
        rows.append(scrape_brand(bc, use_cache=use_cache))
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape brand homepages.")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.setup_logging(verbose=args.verbose)
    config.ensure_dirs()
    df = scrape_all(use_cache=not args.no_cache, only=args.only)
    df.to_parquet(config.BRANDS_PARQUET, index=False)
    n_ok = (df["text_len"] >= config.MIN_BRAND_TEXT_LEN).sum()
    logger.info(
        "Wrote %d brand rows (%d non-empty) -> %s",
        len(df),
        int(n_ok),
        config.BRANDS_PARQUET,
    )


if __name__ == "__main__":
    main()
