# `make` (default) regenerates all Munich plots + the HTML map + the docs/ copy.
# `make plots-berlin` does the same for Berlin (reads berlin_restaurants.csv).
# All recipes read existing *_restaurants.csv files and make NO API calls.

PY  := uv run python
CSV := munich_restaurants.csv

PLOTS := munich_rating_2d_hist.png munich_kde_map.png price_cuisine_grid.png munich_scan_coverage.png

BERLIN_CSV   := berlin_restaurants.csv
BERLIN_COV   := berlin_coverage.json
BERLIN_PLOTS := berlin_rating_2d_hist.png berlin_kde_map.png berlin_price_cuisine_grid.png berlin_scan_coverage.png

.PHONY: all plots plots-berlin tables tables-berlin clean scrape

all: plots

plots: $(PLOTS) docs/index.html

plots-berlin: $(BERLIN_PLOTS) berlin_map.html

munich_rating_2d_hist.png: rating_2d_hist.py $(CSV)
	$(PY) rating_2d_hist.py

munich_kde_map.png: kde_quality_map.py $(CSV)
	$(PY) kde_quality_map.py

price_cuisine_grid.png: price_cuisine_grid.py $(CSV)
	$(PY) price_cuisine_grid.py

munich_scan_coverage.png: scan_coverage.py $(CSV)
	$(PY) scan_coverage.py

munich_map.html: map_html.py $(CSV)
	$(PY) map_html.py

docs/index.html: munich_map.html
	cp munich_map.html docs/index.html

berlin_rating_2d_hist.png: rating_2d_hist.py $(BERLIN_CSV)
	$(PY) rating_2d_hist.py $(BERLIN_CSV) $@

berlin_kde_map.png: kde_quality_map.py $(BERLIN_CSV)
	$(PY) kde_quality_map.py $(BERLIN_CSV) $@

berlin_price_cuisine_grid.png: price_cuisine_grid.py $(BERLIN_CSV)
	$(PY) price_cuisine_grid.py $(BERLIN_CSV) $@

berlin_scan_coverage.png: scan_coverage.py $(BERLIN_CSV) $(BERLIN_COV)
	$(PY) scan_coverage.py $(BERLIN_CSV) $(BERLIN_COV) $@

berlin_map.html: map_html.py $(BERLIN_CSV)
	$(PY) map_html.py $(BERLIN_CSV) $@

# Print-only summaries (no file output).
tables:
	$(PY) outliers.py

tables-berlin:
	$(PY) outliers.py $(BERLIN_CSV)

# WARNING: hits the paid Google Places API. Non-default; run manually only.
scrape:
	$(PY) munich_grid_scrape.py

clean:
	rm -f $(PLOTS) $(BERLIN_PLOTS) munich_map.html berlin_map.html
