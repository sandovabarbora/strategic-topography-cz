"""Fetch top rising Czech queries from Google Trends via pytrends.

Three sources combined for breadth:

1. ``trending_searches(pn='czech_republic')`` — current daily trends.
2. ``realtime_trending_searches(pn='CZ')`` — realtime trending stories.
3. ``related_queries`` (``rising``) for each seed in
   ``config/trend_seeds.yaml`` (``akce``, ``novinka``, ``trend``, etc.).

pytrends is unofficial and brittle. Each call is wrapped in
``tenacity`` exponential backoff. If pytrends fails completely the
fallback is ``config/trends_manual.yaml`` (hand-curated weekly).

CLI:
    python -m src.trends
    python -m src.trends --no-cache    # re-fetch even if recent cache present
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pytrends.request import TrendReq
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src import config

logger = logging.getLogger(__name__)

CACHE_TTL = timedelta(hours=6)


# --- pytrends wrappers with retry ----------------------------------------


_PYTRENDS_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)


@_PYTRENDS_RETRY
def _trending_searches(py: TrendReq) -> list[str]:
    df = py.trending_searches(pn="czech_republic")
    return df.iloc[:, 0].dropna().astype(str).tolist()


@_PYTRENDS_RETRY
def _realtime_trending_searches(py: TrendReq) -> list[str]:
    df = py.realtime_trending_searches(pn="CZ")
    if "title" in df.columns:
        return df["title"].dropna().astype(str).tolist()
    return df.iloc[:, 0].dropna().astype(str).tolist()


@_PYTRENDS_RETRY
def _related_rising_for_seed(py: TrendReq, seed: str) -> list[tuple[str, float]]:
    """Return list of (query, velocity) tuples for the 'rising' related-queries
    of one seed."""
    py.build_payload([seed], cat=0, timeframe="today 3-m", geo="CZ", gprop="")
    rq = py.related_queries()
    bag = rq.get(seed) or {}
    rising = bag.get("rising")
    if rising is None or len(rising) == 0:
        return []
    cols = list(rising.columns)
    # pytrends rising has columns 'query' and 'value' (the velocity index)
    if "query" not in cols or "value" not in cols:
        return []
    return [(str(q), float(v)) for q, v in zip(rising["query"], rising["value"], strict=True)]


# --- High-level collection ------------------------------------------------


def collect_trends() -> pd.DataFrame:
    """Run all three pytrends sources + return a unified DataFrame.

    Returns:
        Columns: ``query``, ``velocity``, ``source``, ``fetched_at``.
        Empty DataFrame if every pytrends call fails.
    """
    rows: list[dict[str, Any]] = []
    now = datetime.now(UTC).isoformat()

    py = TrendReq(hl="cs-CZ", tz=60, timeout=(10, 25), retries=2, backoff_factor=0.5)

    # Source 1: trending_searches (no velocity from this API; assign high constant)
    try:
        items = _trending_searches(py)
        for q in items:
            rows.append({"query": q, "velocity": 200.0, "source": "trending", "fetched_at": now})
        logger.info("trending_searches: %d items", len(items))
    except (RetryError, Exception) as exc:
        logger.warning("trending_searches failed: %s", exc)

    time.sleep(3)

    # Source 2: realtime_trending_searches
    try:
        items = _realtime_trending_searches(py)
        for q in items:
            rows.append({"query": q, "velocity": 150.0, "source": "realtime", "fetched_at": now})
        logger.info("realtime_trending_searches: %d items", len(items))
    except (RetryError, Exception) as exc:
        logger.warning("realtime_trending_searches failed: %s", exc)

    # Source 3: related_queries(rising) for each seed
    seeds_yaml = yaml.safe_load(config.TREND_SEEDS_YAML.read_text(encoding="utf-8"))
    seeds = seeds_yaml["seeds"]
    for seed in seeds:
        time.sleep(5)  # extra polite pause between seed payloads
        try:
            rising = _related_rising_for_seed(py, seed)
            for q, v in rising:
                rows.append({"query": q, "velocity": v, "source": f"rising:{seed}", "fetched_at": now})
            logger.info("rising[%s]: %d items", seed, len(rising))
        except (RetryError, Exception) as exc:
            logger.warning("rising[%s] failed: %s", seed, exc)

    return pd.DataFrame(rows)


def load_manual_fallback() -> pd.DataFrame:
    """Read config/trends_manual.yaml when pytrends produces nothing."""
    if not config.TRENDS_MANUAL_YAML.exists():
        logger.warning(
            "no manual fallback found at %s; the trend layer will be empty",
            config.TRENDS_MANUAL_YAML,
        )
        return pd.DataFrame(columns=["query", "velocity", "source", "fetched_at"])
    raw = yaml.safe_load(config.TRENDS_MANUAL_YAML.read_text(encoding="utf-8"))
    rows = []
    now = datetime.now(UTC).isoformat()
    for item in raw["trends"]:
        rows.append(
            {
                "query": item["query"],
                "velocity": float(item.get("velocity", 100)),
                "source": item.get("source", "manual"),
                "fetched_at": now,
            }
        )
    logger.info("loaded %d trends from manual fallback", len(rows))
    return pd.DataFrame(rows)


def dedupe_and_rank(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Lowercase-dedupe; keep max velocity per query; take top N."""
    if df.empty:
        return df
    df = df.copy()
    df["query_norm"] = df["query"].str.lower().str.strip()
    df = df.sort_values("velocity", ascending=False).drop_duplicates("query_norm")
    df = df.drop(columns="query_norm")
    return df.head(top_n).reset_index(drop=True)


def _cache_file() -> Path:
    return config.TRENDS_CACHE_DIR / "latest.parquet"


def cache_is_fresh(path: Path | None = None) -> bool:
    path = path or _cache_file()
    if not path.exists():
        return False
    age = datetime.now(UTC).timestamp() - path.stat().st_mtime
    return age < CACHE_TTL.total_seconds()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch CZ Google Trends.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.setup_logging(verbose=args.verbose)
    config.ensure_dirs()

    cache = _cache_file()
    if not args.no_cache and cache_is_fresh(path=cache):
        df = pd.read_parquet(cache)
        logger.info("trends cache hit (%s) — %d rows", cache.name, len(df))
    else:
        df = collect_trends()
        if df.empty:
            logger.warning("pytrends produced 0 rows; falling back to manual list")
            df = load_manual_fallback()
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)

    df = dedupe_and_rank(df, top_n=30)
    df.to_parquet(config.TRENDS_PARQUET, index=False)
    logger.info(
        "Wrote %d trend rows (sources: %s) -> %s",
        len(df),
        ", ".join(sorted(df["source"].unique())) if not df.empty else "—",
        config.TRENDS_PARQUET,
    )


if __name__ == "__main__":
    main()
