"""
Layout Agent
------------
Neural component: Groq LLM (llama-3.3-70b)
Input: raw OCR text from Google Vision
Task: segment prescription into semantic zones
Output: LayoutResult with labeled fields + bounding box hints
"""

import json
import re
import streamlit as st
from groq import Groq
from models.rx_schema import LayoutResult, LayoutSegment, ConfidenceLevel
from typing import Optional


LAYOUT_SYSTEM_PROMPT = """You are a medical prescription layout analysis agent specializing in Indian prescriptions.

You receive raw OCR text extracted from a prescription (handwritten or printed).
Your job is to identify and segment the text into semantic zones.

ZONES to identify:
1. patient_header   — patient name, age, gender, weight, date, patient ID
2. drug_list        — list of drug/medication names (brand or generic)
3. dosage_column    — dosage amounts, sig instructions (1-0-1, BD, TDS, etc.)
4. duration_column  — duration of treatment (5 days, 1 week, etc.)
5. prescriber_footer — doctor name, qualification, registration number, clinic, address, contact
6. clinical_notes   — diagnosis, chief complaints, advice, follow-up instructions
7. annotation       — any other handwritten notes, stamps, or unclear text

IMPORTANT RULES:
- Indian prescriptions often mix brand names (Augmentin, Crocin, Pan D) with generic names
- Dosage may be written as "1-0-1", "BD", "TDS", "OD hs", "½ tab", "twice daily after food"
- Instructions may be in Bengali, Hindi, or other regional languages mixed with English
- The layout may be non-standard — no fixed columns

Respond ONLY with a valid JSON object in this exact format:
{
  "segments": [
    {
      "label": "patient_header",
      "text": "extracted text belonging to this zone",
      "confidence": "high|medium|low"
    }
  ],
  "patient_header_text": "consolidated patient info text",
  "drug_list_text": "consolidated drug names only",
  "dosage_column_text": "consolidated dosage/sig text",
  "duration_column_text": "consolidated duration text (e.g., '8 Days', '5 Days', '3 Days', in same order as dosage)",
  "prescriber_footer_text": "consolidated prescriber info text",
  "clinical_notes_text": "any diagnosis or advice text",
  "annotation_text": "any remaining unclear or annotated text",
  "overall_confidence": "high|medium|low",
  "layout_notes": "any observations about the prescription layout or script"
}
"""


def run_layout_agent(
    raw_ocr_text: str,
    groq_api_key: str,
    model: str = "llama-3.3-70b-versatile",
    ocr_blocks: Optional[list] = None,
) -> LayoutResult:
    """
    Layout agent: takes raw OCR text, returns segmented LayoutResult.

    Args:
        raw_ocr_text: Full text string from OCR agent
        groq_api_key: Groq API key
        model: Groq model to use
        ocr_blocks: List of OCRBlock objects from ocr_agent

    Returns:
        LayoutResult with labeled segments
    """

    # Store blocks immediately to avoid Streamlit/serialization issues
    blocks_to_use = list(ocr_blocks) if ocr_blocks else []
    client = Groq(api_key=groq_api_key)

    user_message = f"""Analyze this prescription OCR text and segment it into zones:

--- OCR TEXT START ---
{raw_ocr_text}
--- OCR TEXT END ---

Return the structured JSON segmentation."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LAYOUT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=2000,
    )

    raw_content = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
    raw_content = re.sub(r"\s*```$", "", raw_content)

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        return LayoutResult(
            segments=[
                LayoutSegment(
                    label="unknown",
                    raw_text=raw_ocr_text,
                    confidence=ConfidenceLevel.LOW,
                )
            ],
            overall_confidence=ConfidenceLevel.LOW,
        )

    def _conf(s: str) -> ConfidenceLevel:
        return {"high": ConfidenceLevel.HIGH, "medium": ConfidenceLevel.MEDIUM}.get(
            s.lower(), ConfidenceLevel.LOW
        )

    segments = []
    for seg in data.get("segments", []):
        segments.append(
            LayoutSegment(
                label=seg.get("label", "unknown"),
                raw_text=seg.get("text", ""),
                confidence=_conf(seg.get("confidence", "medium")),
            )
        )

    def _match_segment_to_blocks(segment_text: str, ocr_blocks: list) -> Optional[dict]:
        """Find OCR blocks whose text matches segment_text and union their bboxes."""
        if not segment_text or not ocr_blocks:
            return None

        segment_lower = segment_text.lower()
        matched = []

        MIN_WORD_LEN = 4   # ignore short words like "a", "of", "mg" for word-path matching
        MIN_BLOCK_LEN = 3  # ignore blocks shorter than 3 chars entirely

        for b in ocr_blocks:
            block_text_raw = b.get("text") if isinstance(b, dict) else getattr(b, "text", None)
            if not block_text_raw or len(block_text_raw.strip()) < MIN_BLOCK_LEN:
                continue
            block_text = block_text_raw.strip().lower()

            # Path 1: full block text is a substring of the segment
            if block_text in segment_lower:
                matched.append(b)
                continue

            # Path 2: ALL meaningful words in the block appear in the segment
            # Requires at least 2 meaningful words to avoid single-word false positives
            meaningful_words = [w for w in block_text.split() if len(w) >= MIN_WORD_LEN]
            if len(meaningful_words) >= 2 and all(w in segment_lower for w in meaningful_words):
                matched.append(b)

        if not matched:
            return None

        # Union the bounding boxes of all matched blocks
        bboxes = []
        for b in matched:
            bbox = b.get("bbox") if isinstance(b, dict) else getattr(b, "bbox", None)
            if bbox and len(bbox) == 4:
                bboxes.append(bbox)

        if not bboxes:
            return None

        x0 = min(bb[0] for bb in bboxes)
        y0 = min(bb[1] for bb in bboxes)
        x1 = max(bb[2] for bb in bboxes)
        y1 = max(bb[3] for bb in bboxes)

        # Reject degenerate boxes (less than 2% of image in either dimension)
        if (x1 - x0) < 0.02 or (y1 - y0) < 0.02:
            return None

        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    # Apply bbox matching to each segment
    if blocks_to_use:
        for seg in segments:
            seg.bounding_box = _match_segment_to_blocks(seg.raw_text, blocks_to_use)

    # Techinical(debug) output — shown after bbox assignment so values are accurate
    st.write(f"📊 Layout Analysis: {len(blocks_to_use)} blocks processed → {len(segments)} segments identified")
    for seg in segments:
        st.write(f"   • {seg.label}: {'✅ spatially mapped' if seg.bounding_box else '⬜ text only'}")


    return LayoutResult(
        segments=segments,
        patient_header_text=data.get("patient_header_text"),
        drug_list_text=data.get("drug_list_text"),
        dosage_column_text=data.get("dosage_column_text"),
        duration_column_text=data.get("duration_column_text"),
        prescriber_footer_text=data.get("prescriber_footer_text"),
        annotation_text=data.get("annotation_text") or data.get("clinical_notes_text"),
        overall_confidence=_conf(data.get("overall_confidence", "medium")),
    )