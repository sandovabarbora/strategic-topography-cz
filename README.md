# Strategic Topography of Czech Brands

Episode 2 of the *Brand Reflection Series* (Episode 1: [color
fingerprint](https://github.com/sandovabarbora/color-fingerprint-cz),
Episode 1.5: [sound fingerprint](https://github.com/sandovabarbora/sound-fingerprint-cz)).

A static map of where ~30 major Czech brands stand semantically — their
homepage messaging embedded with multilingual-e5-base, the top rising
Czech Google Trends queries embedded in the same space, all projected
to 2D via UMAP. The map is a snapshot, not a dashboard. The opportunity
table picks trends with high momentum AND low brand coverage.

**Live report:** https://sandovabarbora.github.io/strategic-topography-cz/

## Run

```bash
make install      # uv sync — installs sentence-transformers + UMAP + trafilatura
make all          # scrape → trends → embed → reduce → render
open outputs/index.html
```

Each stage caches: re-runs skip cached scrapes, skip the model
re-download, etc. First run takes 5–10 minutes (mostly model
download + scrape with polite 2s sleep between requests).

## Methodology

1. **Scrape.** `trafilatura` fetches each homepage with a polite UA;
   we keep title + meta description + first three headings + first
   five paragraphs of body text. Anti-bot interstitials (Cloudflare
   challenges) are detected and dropped.
2. **Trends.** `pytrends` is wrapped in `tenacity` exponential
   retry. When the unofficial API rate-limits (which it currently does
   on every call due to an upstream `urllib3` incompatibility), the
   fallback is `config/trends_manual.yaml` — a hand-curated set of
   ~30 plausible rising Czech queries.
3. **Embed.** `intfloat/multilingual-e5-base` (passage prefix for
   brand text, query prefix for trend queries, L2-normalized).
4. **Reduce + cluster.** UMAP (n_neighbors=10, min_dist=0.3, cosine).
   KMeans (k=4) on the brand embeddings; cluster labels are rolled to
   strategic categories (Financial services, Connectivity &
   infrastructure, Food & beverage, Mass-market consumer).
5. **Opportunity scan.** For each trend, cosine similarity to all
   brand vectors in the 768-dim space (NOT in 2D); top three nearest
   brands; saturation = mean top-3 similarity; priority = velocity ×
   (1 − saturation).

This run: **24 brands** scraped successfully (6 dropped due to anti-bot
or empty homepages), **30 trends** loaded via manual fallback, **4
clusters**, **30 opportunities** scored.

## Limitations

This is the part that separates portfolio piece from naive analytics.

**Small N for UMAP.** With ~54 points (24 brands + 30 trends), UMAP
coordinates are sensitive to hyperparameters and should be read as
approximate topology, not precise distances. Cosine similarity in
the original 768-dimensional space is what drives opportunity
scoring, not 2D distance on the map.

**Homepage text ≠ brand strategy.** The corpus is homepage messaging
at one point in time. It misses internal brand positioning, paid
campaign creative, and seasonal communication. A brand can be more
differentiated in campaign work than its homepage suggests.

**Anti-bot drops bias the sample.** Six brands (Kaufland, Alza,
Mall.cz, Notino, Dr. Max, RegioJet) ship JS-only homepages or
challenge non-browser fetches with Cloudflare. They are absent from
the map, not because they don't have brand presence, but because
the cheap scraper can't reach it. Adding Playwright would close
this gap at the cost of a much heavier pipeline.

**pytrends is brittle.** The current `pytrends` 4.9.2 release breaks
against modern `urllib3` (Retry kwarg renamed). The pipeline detects
the failure and falls back to a manual trend list. The map uses real
brand text; the trend overlay uses curated terms. The same trends
overlay would shift week to week with live pytrends data — these
findings are about the corpus shape, not about any specific week's
trending searches.

**Czech NLP overlap inflates cosines.** Czech homepage text uses
heavily overlapping commercial vocabulary (akce, novinky, nabídka,
měsíčně). Baseline brand-to-brand cosine sits around 0.85; trend-to-
brand around 0.80. Absolute similarity numbers are not comparable to
English-text benchmarks; RELATIVE rankings within this corpus are.

**No causal claim.** The "opportunity" column identifies semantic
adjacency, not commercial fit. A trend being near a brand does NOT
mean the brand should activate it. It means a strategist should look
there.

## Project structure

```
strategic-topography-cz/
├── config/
│   ├── brands.yaml          # 30 brand × url
│   ├── trend_seeds.yaml     # pytrends seeds
│   ├── trends_manual.yaml   # fallback trend list
│   └── insights.yaml        # written observations
├── src/
│   ├── config.py            # paths, constants, logging
│   ├── scrape.py            # trafilatura + anti-bot filter
│   ├── trends.py            # pytrends + tenacity retry + manual fallback
│   ├── embed.py             # multilingual-e5-base (passage/query prefixes)
│   ├── reduce.py            # UMAP + KMeans + opportunity scan
│   └── render.py            # Jinja2 → HTML
├── templates/               # report.html.j2 + style.css.j2
├── tests/                   # property + smoke tests (no network)
├── data/
│   ├── raw_html/            # gitignored cache
│   ├── trends_cache/        # gitignored
│   └── processed/           # parquet, committable
└── outputs/index.html       # the deliverable
```

## License

- Code: MIT
- Derived data in `data/processed/`: CC-BY-4.0
- Raw HTML cache is not redistributed; only analytical metrics ship.

## Disclaimer

Analysis of publicly available brand communication. Brands are
referenced for analytical purposes only. Positioning reflects current
homepage messaging at time of scrape. No agency attribution is implied.

Built by [Barbora Šandová](https://datasimply.eu).
