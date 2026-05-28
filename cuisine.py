#!/usr/bin/env python3
"""
Shared cuisine classification used across the analysis scripts.

Maps Google Places `types` strings to a single cuisine label, and defines the
review-count threshold every script filters on.
"""

# Minimum reviews to include. Strict greater-than everywhere: a place with
# exactly 100 reviews is dropped.
MIN_REVIEWS = 100

# Cuisine rules in priority order. The first matching specific type wins; broad
# fallbacks (Asian, Mediterranean) sit last so a vietnamese_restaurant becomes
# "Asian" only if nothing more specific matched.
CUISINE_RULES = [
    ("Italian",  ["italian_restaurant", "pizza_restaurant"]),
    ("Bavarian", ["bavarian_restaurant", "german_restaurant"]),
    ("Turkish",  ["turkish_restaurant"]),
    ("Indian",   ["indian_restaurant", "north_indian_restaurant"]),
    ("Greek",    ["greek_restaurant"]),
    ("Japanese", ["japanese_restaurant", "sushi_restaurant"]),
    ("Thai",     ["thai_restaurant"]),
    ("Chinese",  ["chinese_restaurant"]),
    ("Vietnamese", ["vietnamese_restaurant"]),
    ("Asian",    ["asian_restaurant", "asian_fusion_restaurant"]),
    ("Mediterranean", ["mediterranean_restaurant"]),
]


def classify(types_field):
    """Map a pipe-delimited Google types string to a cuisine label, or None.

    Order-dependent: the first matching CUISINE_RULES entry wins. A place
    tagged both `pizza_restaurant` and `mediterranean_restaurant` becomes
    "Italian" because Italian is listed earlier. Broad fallbacks (Asian,
    Mediterranean) intentionally sit last.
    """
    types = set(types_field.split("|")) if types_field else set()
    for label, keys in CUISINE_RULES:
        if any(k in types for k in keys):
            return label
    return None
