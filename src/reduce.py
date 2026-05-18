"""UMAP 2D + KMeans clusters + opportunity scan.

Three things in one stage because they all depend on the embeddings
and feed the renderer:

1. **UMAP** projects all 54 points (brands + trends) to 2D so the
   renderer can plot them. n_neighbors=10 and min_dist=0.3 produce
   a layout that keeps category-level structure visible without
   pulling every point on top of its category siblings.

2. **KMeans (k=6)** runs on the BRAND embeddings only (higher fidelity
   than 2D UMAP coords) and labels each brand with a cluster id.
   TF-IDF on the brand text within each cluster extracts the top 5
   distinguishing tokens, which then auto-name the cluster.

3. **Opportunity scan** computes cosine similarity between every
   trend and every brand in the 768-d space (NOT in 2D), takes
   the 3 nearest brands per trend, and scores each trend by
   ``velocity * (1 - mean_top3_sim)`` — high momentum and low
   brand coverage means strategic white space.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

from src import config

logger = logging.getLogger(__name__)


def _label_cluster(sectors: pd.Series) -> str:
    """Strategic label for a cluster based on sector composition.

    The TF-IDF token labels were noisy (Czech short-doc tokens, brand-name
    tokens leaking in). Sector counts produce a more legible label that
    reads like a strategist would name the category.

    Per brief §8 failure mode #6: sector-based naming is the prescribed
    fallback when TF-IDF labels are unreadable; here we go a step further
    and roll related sectors up to strategic categories (Banking +
    Insurance = Financial services; Telecom + Energy + Tech =
    Connectivity & infrastructure).
    """
    counts = sectors.value_counts()
    n_total = int(counts.sum())
    if n_total == 0:
        return "Cluster"

    n_banking = int(counts.get("banking", 0))
    n_insurance = int(counts.get("insurance", 0))
    n_telecom = int(counts.get("telecom", 0))
    n_energy = int(counts.get("energy", 0))
    n_tech = int(counts.get("tech", 0))
    n_beverage = int(counts.get("beverage", 0))
    n_qsr = int(counts.get("qsr", 0))
    n_retail = int(counts.get("retail", 0))
    n_ecommerce = int(counts.get("ecommerce", 0))

    # 1. Financial services (banks + insurance dominant)
    if (n_banking + n_insurance) >= 0.6 * n_total and n_banking >= 2:
        return "Financial services"

    # 2. Connectivity & infrastructure (telecom + energy + utility tech)
    if (n_telecom + n_energy + n_tech) >= 0.6 * n_total and n_telecom >= 2:
        return "Connectivity & infrastructure"

    # 3. Food & beverage (beverage + QSR)
    if (n_beverage + n_qsr) >= 0.6 * n_total and (n_beverage + n_qsr) >= 3:
        return "Food & beverage"

    # 4. Mass-market consumer (retail + ecommerce + miscellaneous lifestyle)
    if (n_retail + n_ecommerce) >= 0.4 * n_total and (n_retail + n_ecommerce) >= 2:
        return "Mass-market consumer"

    # Fallback: top single sector if it has ≥50%
    top_sector = counts.index[0]
    if counts.iloc[0] / n_total >= 0.5:
        return top_sector.title()
    return "Mixed"


# --- UMAP -----------------------------------------------------------------


def project_2d(embeddings: pd.DataFrame) -> pd.DataFrame:
    """Project all points (brands + trends) into a shared 2D space.

    Args:
        embeddings: rows of ``id``, ``type``, ``embedding`` (list of 768).

    Returns:
        DataFrame with ``id``, ``type``, ``x``, ``y``.
    """
    mat = np.stack([np.asarray(e, dtype=np.float32) for e in embeddings["embedding"]])
    logger.info("UMAP on %d points (dim=%d)", *mat.shape)
    reducer = UMAP(
        n_components=2,
        random_state=config.RANDOM_STATE,
        metric=config.UMAP_METRIC,
        n_neighbors=min(config.UMAP_N_NEIGHBORS, len(mat) - 1),
        min_dist=config.UMAP_MIN_DIST,
    )
    coords = reducer.fit_transform(mat)
    return pd.DataFrame(
        {
            "id": embeddings["id"].to_numpy(),
            "type": embeddings["type"].to_numpy(),
            "x": coords[:, 0].astype(np.float32),
            "y": coords[:, 1].astype(np.float32),
        }
    )


# --- KMeans + cluster labels ---------------------------------------------


def cluster_brands(
    embeddings: pd.DataFrame, brands: pd.DataFrame
) -> pd.DataFrame:
    """Cluster brand embeddings; auto-label each cluster from TF-IDF top tokens.

    Returns:
        DataFrame with ``brand``, ``sector``, ``cluster_id``, ``cluster_name``,
        ``cluster_top_tokens`` (list of strings).
    """
    brand_embeds = embeddings[embeddings["type"] == "brand"].copy()
    mat = np.stack([np.asarray(e, dtype=np.float32) for e in brand_embeds["embedding"]])
    k = min(config.KMEANS_K, max(2, len(mat) - 1))
    km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10).fit(mat)
    brand_embeds["cluster_id"] = km.labels_

    # Pull brand text in cluster order so TF-IDF can run per-cluster
    merged = brand_embeds[["id", "cluster_id"]].rename(columns={"id": "brand"}).merge(
        brands[["brand", "sector", "text"]], on="brand", how="left"
    )

    cluster_rows: list[dict[str, object]] = []
    for cid, group in merged.groupby("cluster_id"):
        texts = group["text"].fillna("").tolist()
        if not any(t.strip() for t in texts):
            tokens: list[str] = []
        else:
            vec = TfidfVectorizer(
                max_features=200,
                stop_words=config.CZECH_STOPWORDS,
                lowercase=True,
                token_pattern=r"(?u)\b[a-zA-ZáčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]{3,}\b",
            )
            try:
                tfidf = vec.fit_transform(texts)
                # Mean tf-idf across docs in this cluster
                means = np.asarray(tfidf.mean(axis=0)).flatten()
                features = vec.get_feature_names_out()
                top_idx = np.argsort(means)[::-1][: config.TFIDF_TOP_TOKENS_PER_CLUSTER]
                tokens = [features[i] for i in top_idx if means[i] > 0]
            except ValueError as exc:
                logger.warning("TF-IDF failed for cluster %s: %s", cid, exc)
                tokens = []
        label = _label_cluster(group["sector"])
        for _, row in group.iterrows():
            cluster_rows.append(
                {
                    "brand": row["brand"],
                    "sector": row.get("sector", "?"),
                    "cluster_id": int(cid),
                    "cluster_name": label,
                    "cluster_top_tokens": tokens,
                }
            )
        logger.info(
            "cluster %d (%d brands) -> %s | tokens=%s",
            cid,
            len(group),
            label,
            tokens,
        )
    return pd.DataFrame(cluster_rows)


# --- Opportunity scan ----------------------------------------------------


def opportunity_scan(
    embeddings: pd.DataFrame, trends: pd.DataFrame
) -> pd.DataFrame:
    """For each trend: top 3 nearest brands by cosine, plus saturation score."""
    brand_embeds = embeddings[embeddings["type"] == "brand"]
    trend_embeds = embeddings[embeddings["type"] == "trend"]

    brand_mat = np.stack([np.asarray(e, dtype=np.float32) for e in brand_embeds["embedding"]])
    trend_mat = np.stack([np.asarray(e, dtype=np.float32) for e in trend_embeds["embedding"]])
    # Both already unit-normalized → cosine = dot product
    sims = trend_mat @ brand_mat.T  # shape (n_trends, n_brands)

    brand_names = brand_embeds["id"].to_numpy()
    trend_names = trend_embeds["id"].to_numpy()
    velocity_map = dict(zip(trends["query"], trends["velocity"], strict=True))

    rows: list[dict[str, object]] = []
    for ti, trend in enumerate(trend_names):
        order = np.argsort(sims[ti])[::-1]
        top_idx = order[: config.OPPORTUNITY_TOP_N_BRANDS]
        top_sims = sims[ti, top_idx]
        saturation = float(top_sims.mean())
        velocity = float(velocity_map.get(trend, 100.0))
        rows.append(
            {
                "trend": trend,
                "velocity": velocity,
                "near_brand_1": brand_names[top_idx[0]],
                "near_brand_1_sim": float(top_sims[0]),
                "near_brand_2": brand_names[top_idx[1]] if len(top_idx) > 1 else "",
                "near_brand_2_sim": float(top_sims[1]) if len(top_sims) > 1 else 0.0,
                "near_brand_3": brand_names[top_idx[2]] if len(top_idx) > 2 else "",
                "near_brand_3_sim": float(top_sims[2]) if len(top_sims) > 2 else 0.0,
                "saturation_score": saturation,
                "priority_score": velocity * (1.0 - saturation),
            }
        )
    return pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)


# --- CLI -----------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UMAP + KMeans + opportunity scan.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.setup_logging(verbose=args.verbose)
    config.ensure_dirs()
    config.seed_everything()

    embeddings = pd.read_parquet(config.EMBEDDINGS_PARQUET)
    brands = pd.read_parquet(config.BRANDS_PARQUET)
    trends = pd.read_parquet(config.TRENDS_PARQUET)

    coords = project_2d(embeddings)
    coords.to_parquet(config.COORDS_PARQUET, index=False)
    logger.info("Wrote coords -> %s", config.COORDS_PARQUET)

    clusters = cluster_brands(embeddings, brands)
    clusters.to_parquet(config.CLUSTERS_PARQUET, index=False)
    logger.info("Wrote clusters -> %s", config.CLUSTERS_PARQUET)

    opportunities = opportunity_scan(embeddings, trends)
    opportunities.to_parquet(config.OPPORTUNITIES_PARQUET, index=False)
    logger.info(
        "Wrote %d opportunity rows -> %s",
        len(opportunities),
        config.OPPORTUNITIES_PARQUET,
    )


if __name__ == "__main__":
    main()
