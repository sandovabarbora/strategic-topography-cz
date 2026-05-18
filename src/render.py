"""Render the strategic topography report.

One SVG map (~700x700) showing all 24 brands (circles colored by cluster)
and ~30 trends (outlined diamonds sized by velocity), plus an opportunity
table of the top 10 priority trends. Same editorial template language as
the Color and Sound projects.

CLI:
    python -m src.render
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src import config

logger = logging.getLogger(__name__)

AUTHOR_NAME = "Barbora Šandová"
REPO_URL = "https://github.com/sandovabarbora/strategic-topography-cz"

# Cluster palette (OKLCH, matches editorial neutrals + editorial-red accent).
# One distinct hue per cluster, all at similar lightness so no single
# cluster dominates the eye.
CLUSTER_COLORS: list[str] = [
    "oklch(0.55 0.14 50)",   # warm amber — mass-market consumer
    "oklch(0.45 0.13 250)",  # deep blue — financial services
    "oklch(0.48 0.12 145)",  # forest green — connectivity & infra
    "oklch(0.50 0.15 25)",   # editorial red — food & beverage
    "oklch(0.40 0.10 300)",  # plum — fallback / 5th cluster if any
    "oklch(0.50 0.08 80)",   # warm taupe — 6th
]


# --- Map SVG ---------------------------------------------------------------


def _padded_range(values, pad: float = 0.10) -> tuple[float, float]:
    lo, hi = float(min(values)), float(max(values))
    span = hi - lo or 1.0
    return lo - span * pad, hi + span * pad


def build_map_svg(
    coords: pd.DataFrame,
    clusters: pd.DataFrame,
    trends: pd.DataFrame,
    *,
    size: int = 720,
) -> str:
    """Brands as filled circles colored by cluster, trends as outlined
    diamonds sized by velocity.

    Wraps each cluster's circles + labels in a <g data-cluster-id="N">
    so JS can attach hover/click handlers and dim other clusters.
    """
    cluster_lookup = clusters.set_index("brand")
    velocity_lookup = dict(zip(trends["query"], trends["velocity"], strict=True))
    cluster_ids = sorted(clusters["cluster_id"].unique())
    cluster_names = {cid: clusters[clusters["cluster_id"] == cid]["cluster_name"].iloc[0]
                     for cid in cluster_ids}
    cluster_color = {cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cid in enumerate(cluster_ids)}

    pad = 36
    plot_lo = pad
    plot_hi = size - pad
    plot_w = plot_hi - plot_lo
    x_lo, x_hi = _padded_range(coords["x"])
    y_lo, y_hi = _padded_range(coords["y"])

    def x_to_px(x: float) -> float:
        return plot_lo + (x - x_lo) / (x_hi - x_lo) * plot_w

    def y_to_px(y: float) -> float:
        # Flip Y so up-on-screen = up-in-data
        return plot_lo + (y_hi - y) / (y_hi - y_lo) * plot_w

    parts: list[str] = [
        f'<svg class="topo-map" viewBox="0 0 {size} {size}" '
        f'xmlns="http://www.w3.org/2000/svg" aria-label="Strategic topography map">'
    ]

    # Light grid (single faint square)
    parts.append(
        f'<rect x="{plot_lo}" y="{plot_lo}" width="{plot_w}" height="{plot_w}" '
        f'fill="none" stroke="#EFEAE0" stroke-width="0.7"/>'
    )

    # Compute centroid per cluster for the cluster-name label
    centroids: dict[int, tuple[float, float]] = {}
    brand_coords = coords[coords["type"] == "brand"]
    for cid in cluster_ids:
        cluster_brands = set(clusters[clusters["cluster_id"] == cid]["brand"])
        c_pts = brand_coords[brand_coords["id"].isin(cluster_brands)]
        if c_pts.empty:
            continue
        centroids[cid] = (c_pts["x"].mean(), c_pts["y"].mean())

    # Trend diamonds first (so brand circles sit on top)
    trend_coords = coords[coords["type"] == "trend"]
    v_min, v_max = float(trends["velocity"].min()), float(trends["velocity"].max())
    for _, row in trend_coords.iterrows():
        velocity = velocity_lookup.get(row["id"], 100.0)
        # Diamond half-edge: 5 (smallest) → 11 (largest)
        v_norm = (velocity - v_min) / max(v_max - v_min, 1e-6)
        edge = 5 + v_norm * 6
        cx, cy = x_to_px(row["x"]), y_to_px(row["y"])
        points = f"{cx},{cy - edge} {cx + edge},{cy} {cx},{cy + edge} {cx - edge},{cy}"
        parts.append(
            f'<polygon points="{points}" fill="none" stroke="#181818" '
            f'stroke-width="1.1"/>'
        )
        # Label (small, gray, offset diagonally to avoid overlap with brand labels)
        parts.append(
            f'<text x="{cx + edge + 3:.1f}" y="{cy + 3:.1f}" font-size="9" '
            f'fill="#888" font-style="italic">{row["id"]}</text>'
        )

    # Brand circles, grouped by cluster so JS can dim/highlight whole groups
    for cid in cluster_ids:
        cluster_brand_ids = set(clusters[clusters["cluster_id"] == cid]["brand"])
        cluster_rows = brand_coords[brand_coords["id"].isin(cluster_brand_ids)]
        if cluster_rows.empty:
            continue
        color = cluster_color[cid]
        parts.append(
            f'<g class="cluster-group" data-cluster-id="{cid}" '
            f'tabindex="0" role="button" '
            f'aria-label="Show details for {cluster_names[cid]} neighborhood">'
        )
        for _, row in cluster_rows.iterrows():
            cx, cy = x_to_px(row["x"]), y_to_px(row["y"])
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" '
                f'fill="{color}" stroke="#181818" stroke-width="0.8"/>'
            )
            parts.append(
                f'<text x="{cx + 11:.1f}" y="{cy + 3:.1f}" font-size="11" '
                f'fill="#181818" font-weight="600">{row["id"]}</text>'
            )
        # Cluster centroid label sits inside the group too
        if cid in centroids:
            cx_data, cy_data = centroids[cid]
            cx, cy = x_to_px(cx_data), y_to_px(cy_data)
            parts.append(
                f'<text class="cluster-label" x="{cx:.1f}" y="{cy - 18:.1f}" '
                f'text-anchor="middle" font-family="Bodoni Moda, serif" '
                f'font-style="italic" font-size="14" '
                f'fill="{color}" opacity="0.85">{cluster_names[cid]}</text>'
            )
        parts.append("</g>")

    # Strategic trajectory annotation: Air Bank's recommended move from
    # its current position in Banking toward the edge of mass-market
    # (the position held by Pilulka and Avast). Renders as a dashed
    # editorial-red arc with a labeled head. Anchored in data coords
    # so it stays correct when the underlying UMAP layout shifts.
    air_bank = brand_coords[brand_coords["id"] == "Air Bank"]
    avast = brand_coords[brand_coords["id"] == "Avast"]
    pilulka = brand_coords[brand_coords["id"] == "Pilulka"]
    if not air_bank.empty and not avast.empty and not pilulka.empty:
        x0 = x_to_px(float(air_bank["x"].iloc[0]))
        y0 = y_to_px(float(air_bank["y"].iloc[0]))
        # Target: midpoint between Avast and Pilulka, pulled slightly
        # back toward Air Bank so it sits on the edge rather than inside.
        tx_data = (float(avast["x"].iloc[0]) + float(pilulka["x"].iloc[0])) / 2
        ty_data = (float(avast["y"].iloc[0]) + float(pilulka["y"].iloc[0])) / 2
        x1 = x_to_px(tx_data)
        y1 = y_to_px(ty_data)
        # Shorten so the arrow lands before the brand circles
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / length, dy / length
        x1 -= ux * 18
        y1 -= uy * 18
        x0 += ux * 12  # offset start so it doesn't overlap Air Bank circle
        y0 += uy * 12
        # Arrowhead points
        head_len = 9
        head_w = 6
        bx, by = x1 - ux * head_len, y1 - uy * head_len
        # Perpendicular for arrowhead wings
        px, py = -uy, ux
        h1x, h1y = bx + px * head_w, by + py * head_w
        h2x, h2y = bx - px * head_w, by - py * head_w
        accent = "oklch(0.46 0.16 25)"
        parts.append(
            f'<g class="trajectory" aria-label="Air Bank recommended strategic move from Banking neighborhood toward the edge of mass-market consumer">'
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
            f'stroke="{accent}" stroke-width="1.6" stroke-dasharray="5 4" opacity="0.9"/>'
            f'<polygon points="{x1:.1f},{y1:.1f} {h1x:.1f},{h1y:.1f} {h2x:.1f},{h2y:.1f}" '
            f'fill="{accent}" opacity="0.95"/>'
            f'</g>'
        )

    parts.append("</svg>")
    return "".join(parts)


def build_cluster_panel_data(clusters: pd.DataFrame) -> list[dict[str, object]]:
    """One record per cluster: id, name, color, brand list, top tokens.

    Used by the page-level JS to render the side panel when a cluster
    is clicked. Tokens come pre-computed from clusters.parquet
    (cluster_top_tokens column).
    """
    cluster_ids = sorted(clusters["cluster_id"].unique())
    rows: list[dict[str, object]] = []
    for i, cid in enumerate(cluster_ids):
        sub = clusters[clusters["cluster_id"] == cid]
        rows.append(
            {
                "id": int(cid),
                "name": sub["cluster_name"].iloc[0],
                "color": CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                "brands": sub["brand"].tolist(),
                "top_tokens": list(sub["cluster_top_tokens"].iloc[0]),
            }
        )
    return rows


# --- Cluster legend chips -------------------------------------------------


def build_legend_html(clusters: pd.DataFrame) -> list[dict[str, object]]:
    cluster_ids = sorted(clusters["cluster_id"].unique())
    rows: list[dict[str, object]] = []
    for i, cid in enumerate(cluster_ids):
        name = clusters[clusters["cluster_id"] == cid]["cluster_name"].iloc[0]
        brands = clusters[clusters["cluster_id"] == cid]["brand"].tolist()
        rows.append(
            {
                "name": name,
                "color": CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                "n_brands": len(brands),
                "brand_examples": ", ".join(brands[:4]) + ("…" if len(brands) > 4 else ""),
            }
        )
    return rows


# --- Insights -------------------------------------------------------------


def load_patterns_and_playbook() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load patterns + playbook from config/insights.yaml.

    Returns:
        (patterns, playbook) — both lists of {title, evidence/body, takeaway?}
        dicts. Empty lists if the file is absent.
    """
    if not config.INSIGHTS_YAML.exists():
        return ([], [])
    raw = yaml.safe_load(config.INSIGHTS_YAML.read_text(encoding="utf-8"))
    return list(raw.get("patterns", [])), list(raw.get("playbook", []))


def load_insights() -> list[dict[str, str]]:
    if not config.INSIGHTS_YAML.exists():
        return []
    raw = yaml.safe_load(config.INSIGHTS_YAML.read_text(encoding="utf-8"))
    return list(raw.get("insights", []))


# --- Top-level render -----------------------------------------------------


def render_report(
    *,
    out_path: Path = config.REPORT_HTML,
    templates_dir: Path = config.TEMPLATES_DIR,
) -> Path:
    coords = pd.read_parquet(config.COORDS_PARQUET)
    clusters = pd.read_parquet(config.CLUSTERS_PARQUET)
    trends = pd.read_parquet(config.TRENDS_PARQUET)
    opportunities = pd.read_parquet(config.OPPORTUNITIES_PARQUET)
    brands = pd.read_parquet(config.BRANDS_PARQUET)

    n_brands_with_text = int((brands["text_len"] > 0).sum())

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    css = env.get_template("style.css.j2").render()

    counts = {
        "n_brands_total": len(brands),
        "n_brands_used": n_brands_with_text,
        "n_brands_dropped": len(brands) - n_brands_with_text,
        "n_trends": len(trends),
        "n_clusters": int(clusters["cluster_id"].nunique()),
    }

    top_opps = opportunities.head(config.OPPORTUNITY_TABLE_ROWS).to_dict(orient="records")
    patterns, playbook = load_patterns_and_playbook()

    import json

    html = env.get_template("report.html.j2").render(
        inline_css=css,
        author=AUTHOR_NAME,
        repo_url=REPO_URL,
        counts=counts,
        map_svg=build_map_svg(coords, clusters, trends),
        legend=build_legend_html(clusters),
        cluster_panel_json=json.dumps(build_cluster_panel_data(clusters), ensure_ascii=False),
        opportunities=top_opps,
        patterns=patterns,
        playbook=playbook,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the topography report.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.setup_logging(verbose=args.verbose)
    config.ensure_dirs()
    out = render_report()
    size_kb = out.stat().st_size / 1024
    logger.info("Wrote %s (%.1f KB)", out, size_kb)


if __name__ == "__main__":
    main()
