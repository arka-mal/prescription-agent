"""
OCR / HTR Agent
---------------
Neural component: Google Cloud Vision API
- Handles handwritten (HTR) and printed text
- Detects mixed scripts (Latin, Bengali, Devanagari)
- Returns raw text + bounding boxes + confidence
"""

import base64
import os
import streamlit as st
from typing import Optional
from google.cloud import vision
from google.oauth2 import service_account
from models.rx_schema import OCRResult, ConfidenceLevel, OCRBlock


def _get_vision_client(credentials_path: Optional[str] = None, credentials_info: Optional[dict] = None):
    if credentials_info:
        creds = service_account.Credentials.from_service_account_info(credentials_info)
        return vision.ImageAnnotatorClient(credentials=creds)
    if credentials_path and os.path.exists(credentials_path):
        creds = service_account.Credentials.from_service_account_file(credentials_path)
        return vision.ImageAnnotatorClient(credentials=creds)
    # fallback: use GOOGLE_APPLICATION_CREDENTIALS env var
    return vision.ImageAnnotatorClient()


def _detect_scripts(full_text: str) -> list[str]:
    """Heuristic script detection from extracted text."""
    scripts = ["latin"]  # always present for drug names
    bengali_range = range(0x0980, 0x09FF)
    devanagari_range = range(0x0900, 0x097F)

    for char in full_text:
        cp = ord(char)
        if cp in bengali_range:
            if "bengali" not in scripts:
                scripts.append("bengali")
        if cp in devanagari_range:
            if "devanagari" not in scripts:
                scripts.append("devanagari")
    return scripts


def _compute_confidence(responses) -> ConfidenceLevel:
    """Map Google Vision confidence to our enum."""
    # Google Vision doesn't return per-document confidence for full text
    # We use heuristic: if blocks exist with low word confidence, flag it
    try:
        pages = responses.full_text_annotation.pages
        confidences = []
        for page in pages:
            for block in page.blocks:
                confidences.append(block.confidence)
        if not confidences:
            return ConfidenceLevel.MEDIUM
        avg = sum(confidences) / len(confidences)
        if avg >= 0.85:
            return ConfidenceLevel.HIGH
        elif avg >= 0.60:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    except Exception:
        return ConfidenceLevel.MEDIUM


def run_ocr_agent(
    image_bytes: bytes,
    credentials_path: Optional[str] = None,
    credentials_info: Optional[dict] = None,
) -> OCRResult:
    """
    Main OCR/HTR agent entry point.
    
    Args:
        image_bytes: Raw image bytes (JPEG, PNG, TIFF, PDF page)
        credentials_path: Path to GCP service account JSON (local/testing use)
        credentials_info: GCP service account dict, e.g. from st.secrets (production use)
    
    Returns:
        OCRResult with raw text, script detection, confidence
    """
    client = _get_vision_client(credentials_path, credentials_info)

    image = vision.Image(content=image_bytes)

    # Use DOCUMENT_TEXT_DETECTION — optimized for dense text, forms, handwriting
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Google Vision API error: {response.error.message}")

    full_text = response.full_text_annotation.text if response.full_text_annotation else ""
    word_count = len(full_text.split()) if full_text else 0

    scripts = _detect_scripts(full_text)
    confidence = _compute_confidence(response)

    # Identify handwritten vs printed blocks heuristically
    # Google Vision marks handwritten text with block_type == UNKNOWN or via confidence
    handwritten_regions = []
    printed_regions = []
    blocks = []

    try:
        for page in response.full_text_annotation.pages:
            page_w, page_h = page.width, page.height
            for block in page.blocks:
                block_text = ""
                for para in block.paragraphs:
                    for word in para.words:
                        word_text = "".join([s.text for s in word.symbols])
                        block_text += word_text + " "
                block_text = block_text.strip()

                if not block_text:
                    continue

                # Extract bounding box and normalize to 0-1
                verts = block.bounding_box.vertices
                xs = [v.x for v in verts]
                ys = [v.y for v in verts]
                bbox_norm = [
                    min(xs) / page_w if page_w else 0,
                    min(ys) / page_h if page_h else 0,
                    max(xs) / page_w if page_w else 1,
                    max(ys) / page_h if page_h else 1,
                ]

                blocks.append(OCRBlock(
                    text=block_text,
                    bbox=bbox_norm,
                    confidence=block.confidence,
                    is_handwritten_guess=block.confidence < 0.75,
                ))

                # Heuristic: low-confidence blocks → likely handwritten
                if block.confidence < 0.75:
                    handwritten_regions.append(block_text[:80])
                else:
                    printed_regions.append(block_text[:80])
    except Exception:
        pass

    # Flag mixed-script regions
    mixed_flags = []
    if len(scripts) > 1:
        mixed_flags.append(
            f"Mixed scripts detected: {', '.join(scripts)} — dosage instructions may be in regional language"
        )

    # Techinical(debug) output
    st.write(f"📊 OCR Analysis: {len(blocks)} text blocks extracted")
    
    return OCRResult(
        raw_text=full_text,
        detected_scripts=scripts,
        handwritten_regions=handwritten_regions[:10],
        printed_regions=printed_regions[:10],
        mixed_script_flags=mixed_flags,
        overall_confidence=confidence,
        word_count=word_count,
        blocks=blocks,
    )
