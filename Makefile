.PHONY: help install scrape trends embed reduce cluster opportunity render all test lint format clean
.DEFAULT_GOAL := help

PYTHON := uv run python

help:  ## Show targets
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## uv sync
	uv sync

scrape:  ## Scrape brand homepages
	$(PYTHON) -m src.scrape

trends:  ## Fetch CZ Google Trends (pytrends)
	$(PYTHON) -m src.trends

embed:  ## Embed brands + trends via multilingual-e5-base
	$(PYTHON) -m src.embed

reduce:  ## UMAP 2D + KMeans clusters + opportunity scan
	$(PYTHON) -m src.reduce

render:  ## Render the HTML report
	$(PYTHON) -m src.render

all: scrape trends embed reduce render  ## Full pipeline

test:  ## pytest
	uv run pytest

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

clean:
	rm -rf data/processed/* outputs/index.html pipeline.log
