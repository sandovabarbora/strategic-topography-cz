"""Smoke + property tests for the pipeline. Don't require external network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.reduce import _label_cluster, opportunity_scan


def test_paths_under_project_root() -> None:
    root = config.PROJECT_ROOT
    for p in (
        config.CONFIG_DIR, config.DATA_DIR, config.RAW_HTML_DIR,
        config.PROCESSED_DIR, config.OUTPUTS_DIR, config.TEMPLATES_DIR,
        config.BRANDS_YAML, config.TREND_SEEDS_YAML,
        config.BRANDS_PARQUET, config.TRENDS_PARQUET,
        config.EMBEDDINGS_PARQUET, config.COORDS_PARQUET,
        config.CLUSTERS_PARQUET, config.OPPORTUNITIES_PARQUET,
    ):
        assert root in p.parents or p == root, f"{p} escapes project root"


def test_constants_sane() -> None:
    assert config.EMBED_DIM == 768
    assert config.UMAP_N_NEIGHBORS > 1
    assert 0 < config.UMAP_MIN_DIST < 1
    assert config.KMEANS_K >= 3
    assert config.EMBED_PREFIX_PASSAGE.endswith(" ")
    assert config.EMBED_PREFIX_QUERY.endswith(" ")


def test_label_cluster_financial() -> None:
    sectors = pd.Series(["banking", "banking", "banking", "banking", "insurance"])
    assert _label_cluster(sectors) == "Financial services"


def test_label_cluster_pure_banking() -> None:
    """When the cluster is 100% banking (no insurance / other), use 'Banking'."""
    sectors = pd.Series(["banking"] * 3)
    assert _label_cluster(sectors) == "Banking"


def test_label_cluster_banking_plus_heavy_industry() -> None:
    """Banks anchoring a cluster of established institutions (energy, auto, insurance)."""
    sectors = pd.Series(["banking"] * 5 + ["energy", "automotive", "insurance"])
    assert _label_cluster(sectors) == "Banking + heavy industry"


def test_label_cluster_infrastructure() -> None:
    sectors = pd.Series(["telecom", "telecom", "telecom", "energy", "tech"])
    assert _label_cluster(sectors) == "Connectivity & infrastructure"


def test_label_cluster_food_bev() -> None:
    sectors = pd.Series(["beverage", "beverage", "qsr", "qsr"])
    assert _label_cluster(sectors) == "Food & beverage"


def test_label_cluster_mass_market_consumer() -> None:
    sectors = pd.Series(["retail", "retail", "ecommerce", "media"])
    assert _label_cluster(sectors) == "Mass-market consumer"


def test_opportunity_scan_orders_by_priority() -> None:
    """Synthetic: a high-velocity trend far from all brands should rank
    above a high-velocity trend that's near a brand."""
    rng = np.random.default_rng(0)
    # 3 brand vectors clustered around one direction
    brand_dir = np.array([1.0, 0.0, 0.0])
    brand_vecs = np.stack(
        [brand_dir + 0.05 * rng.standard_normal(3) for _ in range(3)]
    )
    brand_vecs /= np.linalg.norm(brand_vecs, axis=1, keepdims=True)
    # Two trend vectors: one near brands, one far (perpendicular)
    near = brand_dir + 0.02 * rng.standard_normal(3)
    far = np.array([0.0, 1.0, 0.0])
    trend_vecs = np.stack([near / np.linalg.norm(near), far])

    emb = pd.DataFrame(
        {
            "id": ["B1", "B2", "B3", "Tnear", "Tfar"],
            "type": ["brand", "brand", "brand", "trend", "trend"],
            "embedding": list(brand_vecs) + list(trend_vecs),
        }
    )
    trends = pd.DataFrame(
        {"query": ["Tnear", "Tfar"], "velocity": [100.0, 100.0]}
    )
    opp = opportunity_scan(emb, trends)
    # Tfar should have higher priority (lower saturation, same velocity)
    tnear_row = opp[opp["trend"] == "Tnear"].iloc[0]
    tfar_row = opp[opp["trend"] == "Tfar"].iloc[0]
    assert tfar_row["saturation_score"] < tnear_row["saturation_score"]
    assert tfar_row["priority_score"] > tnear_row["priority_score"]


def test_embeddings_parquet_dim_and_norms() -> None:
    """Integration: if the actual embeddings.parquet exists, every vector
    is dim 768 and approximately unit-normed."""
    if not config.EMBEDDINGS_PARQUET.exists():
        pytest.skip("embeddings.parquet not present; run `make embed` first")
    df = pd.read_parquet(config.EMBEDDINGS_PARQUET)
    assert (df["embedding"].apply(len) == config.EMBED_DIM).all()
    norms = df["embedding"].apply(lambda v: float(np.linalg.norm(v)))
    assert (norms.between(0.99, 1.01)).all(), "embeddings should be L2-normalized"
