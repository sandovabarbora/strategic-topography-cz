"""Project configuration: paths, constants, logging, random seeds.

Episode 2 of the Brand Reflection Series. Sister of color-fingerprint-cz
and sound-fingerprint-cz; shares the same conceptual frame, different
modality (text + semantic embedding).
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

# --- Filesystem layout -----------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_HTML_DIR: Path = DATA_DIR / "raw_html"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
TEMPLATES_DIR: Path = PROJECT_ROOT / "templates"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
TRENDS_CACHE_DIR: Path = DATA_DIR / "trends_cache"

BRANDS_YAML: Path = CONFIG_DIR / "brands.yaml"
TREND_SEEDS_YAML: Path = CONFIG_DIR / "trend_seeds.yaml"
TRENDS_MANUAL_YAML: Path = CONFIG_DIR / "trends_manual.yaml"
INSIGHTS_YAML: Path = CONFIG_DIR / "insights.yaml"

BRANDS_PARQUET: Path = PROCESSED_DIR / "brands.parquet"
TRENDS_PARQUET: Path = PROCESSED_DIR / "trends.parquet"
EMBEDDINGS_PARQUET: Path = PROCESSED_DIR / "embeddings.parquet"
COORDS_PARQUET: Path = PROCESSED_DIR / "coords.parquet"
CLUSTERS_PARQUET: Path = PROCESSED_DIR / "clusters.parquet"
OPPORTUNITIES_PARQUET: Path = PROCESSED_DIR / "opportunities.parquet"
REPORT_HTML: Path = OUTPUTS_DIR / "index.html"
PIPELINE_LOG: Path = PROJECT_ROOT / "pipeline.log"

# --- Pipeline constants ----------------------------------------------------

RANDOM_STATE: int = 42

# Scraping
USER_AGENT: str = (
    "BrandReflectionSeries/1.0 (research; barbora@datasimply.eu)"
)
REQUEST_TIMEOUT_S: float = 20.0
INTER_REQUEST_SLEEP_S: float = 2.0
MIN_BRAND_TEXT_LEN: int = 50

# Embedding model
EMBED_MODEL: str = "intfloat/multilingual-e5-base"
EMBED_PREFIX_PASSAGE: str = "passage: "
EMBED_PREFIX_QUERY: str = "query: "
EMBED_DIM: int = 768

# UMAP
UMAP_N_NEIGHBORS: int = 10
UMAP_MIN_DIST: float = 0.3
UMAP_METRIC: str = "cosine"

# Clustering
KMEANS_K: int = 4
TFIDF_TOP_TOKENS_PER_CLUSTER: int = 5

# Opportunity scan
OPPORTUNITY_TOP_N_BRANDS: int = 3
OPPORTUNITY_TABLE_ROWS: int = 10


# --- Czech stopwords (small inline list — don't depend on a fragile pkg) ---

CZECH_STOPWORDS: list[str] = [
    "a", "i", "v", "na", "se", "je", "to", "že", "pro", "s", "z", "do", "od",
    "po", "k", "u", "o", "ale", "nebo", "jak", "co", "kde", "kdy", "tak",
    "už", "vám", "nám", "si", "by", "byl", "byla", "bylo", "být", "jsou",
    "jsme", "jste", "jsem", "ne", "ano", "ten", "ta", "to", "ti", "této",
    "tento", "tomto", "tomuto", "ty", "vy", "my", "on", "ona", "oni", "ho",
    "mu", "ji", "jí", "ní", "jich", "jim", "při", "před", "ze", "ke", "že",
    "až", "i", "více", "také", "už", "ještě", "jen", "jenom", "při", "díky",
    "naše", "naši", "vaše", "vaši", "svůj", "své", "svým", "této", "tuto",
    "našemu", "vašemu", "tom", "tím", "tímto", "vás", "nás", "mi", "mě",
    "co", "kdo", "kterým", "který", "která", "které", "kterou", "kterého",
    "www", "cz", "http", "https", "html",
]


# --- Logging + seed setup --------------------------------------------------


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logger: stdout INFO (DEBUG if verbose), file DEBUG."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.DEBUG if verbose else logging.INFO)
    stdout.setFormatter(fmt)
    root.addHandler(stdout)

    PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(PIPELINE_LOG, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    return root


def ensure_dirs() -> None:
    """Create raw_html / processed / outputs / figures / trends_cache dirs."""
    for d in (RAW_HTML_DIR, PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR, TRENDS_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Set seeds for python random, numpy, and PYTHONHASHSEED."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
