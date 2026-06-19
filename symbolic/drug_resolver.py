"""
Symbolic Drug Resolver
----------------------
Deterministic lookup: brand name → generic name.
This is pure symbolic — no LLM involved.
"""

import json
import os
import difflib
from typing import Optional, Tuple

FUZZY_MATCH_THRESHOLD = 0.82
MIN_FUZZY_KEY_LEN = 4

_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "drug_kb.json")

def _load_kb():
    with open(_KB_PATH, "r") as f:
        return json.load(f)

_KB = _load_kb()


def resolve_brand(name: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Given a drug name (possibly brand), return (brand, generic, kb_verified).
    kb_verified is True only if matched against the KB (exact or fuzzy above
    threshold). False means the name passed through unconfirmed.

    Returns: (brand_name, generic_name, kb_verified)
    """
    if not name:
        return None, None, False

    normalized = name.strip().lower()
    normalized = normalized.split()[0] if normalized else normalized

    b2g = _KB.get("brand_to_generic", {})

    if normalized in b2g:
        return name.strip(), b2g[normalized], True

    if len(normalized) >= MIN_FUZZY_KEY_LEN:
        best_brand, best_generic, best_ratio = None, None, 0.0
        for brand, generic in b2g.items():
            if len(brand) < MIN_FUZZY_KEY_LEN:
                continue
            ratio = difflib.SequenceMatcher(None, normalized, brand).ratio()
            if ratio > best_ratio:
                best_brand, best_generic, best_ratio = brand, generic, ratio
        if best_ratio >= FUZZY_MATCH_THRESHOLD:
            return name.strip(), best_generic, True

    return None, name.strip(), False


def normalize_form(form_raw: str) -> str:
    """Normalize drug form abbreviation."""
    if not form_raw:
        return ""
    forms = _KB.get("common_forms", {})
    key = form_raw.strip().lower().rstrip(".")
    return forms.get(key, form_raw.strip())


def normalize_route(route_raw: str) -> str:
    """Normalize route abbreviation."""
    if not route_raw:
        return ""
    routes = _KB.get("common_routes", {})
    key = route_raw.strip().lower().rstrip(".")
    return routes.get(key, route_raw.strip())
