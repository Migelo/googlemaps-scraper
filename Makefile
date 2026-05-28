# `make` (default) regenerates all plots + the HTML map + the docs/ copy.
# All recipes read munich_restaurants.csv and make NO API calls (free + safe).

PY  := uv run python
CSV := munich_restaurants.csv

PLOTS := munich_rating_2d_hist.png munich_kde_map.png price_cuisine_grid.png munich_scan_coverage.png

.PHONY: all plots tables clean scrape

all: plots

plots: $(PLOTS) docs/index.html

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

# Print-only summaries (no file output); --geocode is omitted (it costs money).
tables:
	$(PY) outliers.py
	$(PY) neighborhoods.py
	$(PY) name_tokens.py

# WARNING: hits the paid Google Places API. Non-default; run manually only.
scrape:
	$(PY) munich_grid_scrape.py

clean:
	rm -f $(PLOTS) munich_map.html
