"""
Symbolic Sig Parser
-------------------
Rule-based engine to normalize prescription dosage instructions (sigs)
into structured timing objects. Handles Indian clinical shorthand.

This is the SYMBOLIC component of the neuro-symbolic pipeline.
LLM extracts raw sig strings → this engine normalizes them deterministically.
"""

import re
from typing import Optional
from models.rx_schema import TimingStructure


# ── Canonical frequency mappings ─────────────────────────────────────────────

FREQUENCY_MAP = {
    # Latin abbreviations
    "od": {"morning": 1, "afternoon": 0, "evening": 0, "night": 0, "freq": 1, "label": "OD"},
    "bd": {"morning": 1, "afternoon": 0, "evening": 0, "night": 1, "freq": 2, "label": "BD"},
    "bid": {"morning": 1, "afternoon": 0, "evening": 0, "night": 1, "freq": 2, "label": "BD"},
    "tds": {"morning": 1, "afternoon": 1, "evening": 0, "night": 1, "freq": 3, "label": "TDS"},
    "tid": {"morning": 1, "afternoon": 1, "evening": 0, "night": 1, "freq": 3, "label": "TDS"},
    "qid": {"morning": 1, "afternoon": 1, "evening": 1, "night": 1, "freq": 4, "label": "QID"},
    "qds": {"morning": 1, "afternoon": 1, "evening": 1, "night": 1, "freq": 4, "label": "QID"},
    "sos": {"morning": 0, "afternoon": 0, "evening": 0, "night": 0, "freq": 0, "label": "SOS"},
    "prn": {"morning": 0, "afternoon": 0, "evening": 0, "night": 0, "freq": 0, "label": "SOS"},
    "stat": {"morning": 1, "afternoon": 0, "evening": 0, "night": 0, "freq": 1, "label": "STAT"},

    # English plain text
    "once daily": {"morning": 1, "afternoon": 0, "evening": 0, "night": 0, "freq": 1, "label": "OD"},
    "once a day": {"morning": 1, "afternoon": 0, "evening": 0, "night": 0, "freq": 1, "label": "OD"},
    "twice daily": {"morning": 1, "afternoon": 0, "evening": 0, "night": 1, "freq": 2, "label": "BD"},
    "twice a day": {"morning": 1, "afternoon": 0, "evening": 0, "night": 1, "freq": 2, "label": "BD"},
    "three times daily": {"morning": 1, "afternoon": 1, "evening": 0, "night": 1, "freq": 3, "label": "TDS"},
    "thrice daily": {"morning": 1, "afternoon": 1, "evening": 0, "night": 1, "freq": 3, "label": "TDS"},
    "four times daily": {"morning": 1, "afternoon": 1, "evening": 1, "night": 1, "freq": 4, "label": "QID"},
    "at bedtime": {"morning": 0, "afternoon": 0, "evening": 0, "night": 1, "freq": 1, "label": "OD hs"},
    "at night": {"morning": 0, "afternoon": 0, "evening": 0, "night": 1, "freq": 1, "label": "OD hs"},
    "morning": {"morning": 1, "afternoon": 0, "evening": 0, "night": 0, "freq": 1, "label": "OD morning"},
}

# Timing modifiers
HS_PATTERNS = re.compile(r'\bhs\b|\bbedtime\b|\bat night\b|\bnight\b', re.IGNORECASE)
AC_PATTERNS = re.compile(r'\bac\b|\bbefore food\b|\bbefore meal\b|\bfasting\b|\bempty stomach\b', re.IGNORECASE)
PC_PATTERNS = re.compile(r'\bpc\b|\bafter food\b|\bafter meal\b|\bwith food\b|\bwith meal\b', re.IGNORECASE)

# Duration patterns
DURATION_PATTERNS = [
    (re.compile(r'(\d+)\s*days?', re.IGNORECASE), lambda m: int(m.group(1))),
    (re.compile(r'(\d+)\s*weeks?', re.IGNORECASE), lambda m: int(m.group(1)) * 7),
    (re.compile(r'(\d+)\s*months?', re.IGNORECASE), lambda m: int(m.group(1)) * 30),
    (re.compile(r'(\d+)\s*wks?', re.IGNORECASE), lambda m: int(m.group(1)) * 7),
    (re.compile(r'(\d+)\s*mths?', re.IGNORECASE), lambda m: int(m.group(1)) * 30),
    (re.compile(r'for\s+(\d+)', re.IGNORECASE), lambda m: int(m.group(1))),
    (re.compile(r'x\s*(\d+)', re.IGNORECASE), lambda m: int(m.group(1))),
]

# Dash pattern: "1-0-1", "1-1-1", "0-0-1", "1-0-0-1" etc.
DASH_PATTERN = re.compile(r'([\d½¼¾\.]+)\s*[-–]\s*([\d½¼¾\.]+)\s*[-–]\s*([\d½¼¾\.]+)(?:\s*[-–]\s*([\d½¼¾\.]+))?')

FRACTION_MAP = {"½": 0.5, "¼": 0.25, "¾": 0.75, "1/2": 0.5, "1/4": 0.25, "3/4": 0.75}


def _parse_dose_value(s: str) -> float:
    s = s.strip()
    if s in FRACTION_MAP:
        return FRACTION_MAP[s]
    try:
        return float(s)
    except ValueError:
        return 0.0


def _extract_food_relation(sig: str) -> Optional[str]:
    if PC_PATTERNS.search(sig):
        return "after food"
    if AC_PATTERNS.search(sig):
        return "before food"
    return None


def _extract_duration(sig: str):
    for pattern, extractor in DURATION_PATTERNS:
        m = pattern.search(sig)
        if m:
            days = extractor(m)
            raw = m.group(0)
            return days, raw
    return None, None


def parse_sig(sig_raw: str) -> TimingStructure:
    """
    Main entry point. Takes a raw sig string and returns a TimingStructure.
    
    Examples:
        "1-0-1 pc"         → morning=1, afternoon=0, night=1, after food
        "BD x 5 days"      → freq=2, duration=5
        "TDS after food"   → freq=3, food=after food
        "½ tab OD hs"      → morning=0, night=0.5 (bedtime)
        "twice daily"      → freq=2
    """
    if not sig_raw:
        return TimingStructure()

    sig = sig_raw.strip()
    food = _extract_food_relation(sig)
    duration_days, duration_label = _extract_duration(sig)

    # ── Try dash pattern first ────────────────────────────────────────────────
    dash_match = DASH_PATTERN.search(sig)
    if dash_match:
        groups = dash_match.groups()
        m = _parse_dose_value(groups[0])
        a = _parse_dose_value(groups[1])
        e = _parse_dose_value(groups[2])
        n = _parse_dose_value(groups[3]) if groups[3] else 0.0

        # if 4-part: morning-afternoon-evening-night
        if groups[3]:
            freq = sum(1 for x in [m, a, e, n] if x > 0)
            label = f"{groups[0]}-{groups[1]}-{groups[2]}-{groups[3]}"
        else:
            # check for hs modifier → move evening dose to night
            if HS_PATTERNS.search(sig):
                n = e
                e = 0.0
            freq = sum(1 for x in [m, a, e, n] if x > 0)
            label = f"{groups[0]}-{groups[1]}-{groups[2]}"

        return TimingStructure(
            morning=m, afternoon=a, evening=e, night=n,
            frequency_per_day=freq,
            timing_label=label,
            food_relation=food,
            duration_days=duration_days,
            duration_label=duration_label,
        )

    # ── Try keyword frequency map ─────────────────────────────────────────────
    sig_lower = sig.lower()

    # Multi-word patterns first (order matters)
    for phrase in sorted(FREQUENCY_MAP.keys(), key=len, reverse=True):
        if phrase in sig_lower:
            fm = FREQUENCY_MAP[phrase]
            m, a, e, n = fm["morning"], fm["afternoon"], fm["evening"], fm["night"]
            label = fm["label"]

            # Apply hs modifier
            if HS_PATTERNS.search(sig) and "hs" not in phrase:
                n = max(m, a, e, n, 1)
                m = a = e = 0
                label = label + " hs"

            return TimingStructure(
                morning=m, afternoon=a, evening=e, night=n,
                frequency_per_day=fm["freq"],
                timing_label=label,
                food_relation=food,
                duration_days=duration_days,
                duration_label=duration_label,
            )

    # ── Fallback: couldn't parse ──────────────────────────────────────────────
    return TimingStructure(
        timing_label=sig_raw,
        food_relation=food,
        duration_days=duration_days,
        duration_label=duration_label,
    )


def sig_to_human_readable(t: TimingStructure) -> str:
    """Convert a TimingStructure back to a clean human-readable string."""
    if not t.frequency_per_day and not t.timing_label:
        return "Dosage not specified"

    parts = []
    if t.timing_label:
        parts.append(t.timing_label)
    if t.food_relation:
        parts.append(t.food_relation)
    if t.duration_label:
        parts.append(f"for {t.duration_label}")
    elif t.duration_days:
        parts.append(f"for {t.duration_days} days")

    return " · ".join(parts) if parts else t.timing_label or "As directed"
