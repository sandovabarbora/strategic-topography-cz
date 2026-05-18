"""Embed brand text + trend queries via multilingual-e5-base.

Critical detail (Hugging Face model card for intfloat/multilingual-e5-base):
each input must be prefixed with ``"query: "`` or ``"passage: "``, even
for non-English text. Skipping the prefix is the #1 reason e5 embeddings
look "bad" — distances become unreliable.

Reference: https://huggingface.co/intfloat/multilingual-e5-base

The model is cached at ``~/.cache/huggingface/`` after first download
(~1.1 GB). All embeddings are L2-normalized so cosine similarity equals
dot product.

CLI:
    python -m src.embed
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src import config

logger = logging.getLogger(__name__)


def _load_model() -> SentenceTransformer:
    logger.info("loading %s (first run downloads ~1.1 GB)", config.EMBED_MODEL)
    return SentenceTransformer(config.EMBED_MODEL)


def embed_brands(brands: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    """Embed brand text with the 'passage: ' prefix.

    Args:
        brands: DataFrame with at minimum ``brand`` and ``text`` columns.
            Empty-text rows are dropped (already filtered upstream but
            defensive here too).
        model: a loaded SentenceTransformer.

    Returns:
        DataFrame with columns ``id`` (= brand), ``type`` = 'brand',
        ``embedding`` (list of 768 floats).
    """
    df = brands[brands["text"].str.len() > 0].copy()
    texts = [config.EMBED_PREFIX_PASSAGE + t for t in df["text"]]
    logger.info("embedding %d brand passages", len(texts))
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return pd.DataFrame(
        {
            "id": df["brand"].to_numpy(),
            "type": "brand",
            "embedding": [v.astype(np.float32).tolist() for v in vecs],
        }
    )


def embed_trends(trends: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    """Embed trend queries with the 'query: ' prefix."""
    texts = [config.EMBED_PREFIX_QUERY + q for q in trends["query"]]
    logger.info("embedding %d trend queries", len(texts))
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return pd.DataFrame(
        {
            "id": trends["query"].to_numpy(),
            "type": "trend",
            "embedding": [v.astype(np.float32).tolist() for v in vecs],
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed brand + trend texts via multilingual-e5-base.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.setup_logging(verbose=args.verbose)
    config.ensure_dirs()
    config.seed_everything()

    brands = pd.read_parquet(config.BRANDS_PARQUET)
    trends = pd.read_parquet(config.TRENDS_PARQUET)

    model = _load_model()
    brand_embeds = embed_brands(brands, model)
    trend_embeds = embed_trends(trends, model)

    combined = pd.concat([brand_embeds, trend_embeds], ignore_index=True)
    combined.to_parquet(config.EMBEDDINGS_PARQUET, index=False)

    # Quick sanity on dims + norms
    sample = np.array(combined["embedding"].iloc[0])
    norm = float(np.linalg.norm(sample))
    logger.info(
        "Wrote %d embeddings (brands=%d, trends=%d, dim=%d, sample_norm=%.4f) -> %s",
        len(combined),
        len(brand_embeds),
        len(trend_embeds),
        len(sample),
        norm,
        config.EMBEDDINGS_PARQUET,
    )


if __name__ == "__main__":
    main()
