"""
Symbolic Drug Resolver
----------------------
Deterministic lookup: brand name → generic name.
This is pure symbolic — no LLM involved.
"""

import json
import os
from typing import Optional, Tuple

_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "drug_kb.json")

def _load_kb():
    with open(_KB_PATH, "r") as f:
        return json.load(f)

_KB = _load_kb()


def resolve_brand(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a drug name (possibly brand), return (brand, generic).
    Returns (None, name) if not found in KB — treat as possibly already generic.
    
    Returns: (brand_name, generic_name)
    """
    if not name:
        return None, None

    normalized = name.strip().lower()
    # remove strength/form suffixes e.g. "Augmentin 625" → "augmentin"
    normalized = normalized.split()[0] if normalized else normalized

    b2g = _KB.get("brand_to_generic", {})

    if normalized in b2g:
        return name.strip(), b2g[normalized]

    # Partial match fallback
    for brand, generic in b2g.items():
        if brand in normalized or normalized in brand:
            return name.strip(), generic

    return None, name.strip()   # assume it might already be generic


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
