"""
UI Rendering Helpers
--------------------
Streamlit display functions for each agent's output panel.
Kept separate from app.py to keep the main file clean.
"""

import streamlit as st
import io
from PIL import Image, ImageDraw
from models.rx_schema import (
    OCRResult, LayoutResult, ConfidenceLevel
)



# ── Confidence badge helpers ──────────────────────────────────────────────────

CONF_COLOR = {
    ConfidenceLevel.HIGH: "#22c55e",
    ConfidenceLevel.MEDIUM: "#f59e0b",
    ConfidenceLevel.LOW: "#ef4444",
}

CONF_EMOJI = {
    ConfidenceLevel.HIGH: "🟢",
    ConfidenceLevel.MEDIUM: "🟡",
    ConfidenceLevel.LOW: "🔴",
}

SEGMENT_COLORS = {
    "patient_header":    "#dbeafe",
    "drug_list":         "#dcfce7",
    "dosage_column":     "#fef9c3",
    "duration_column":   "#fce7f3",
    "prescriber_footer": "#ede9fe",
    "clinical_notes":    "#ffedd5",
    "annotation":        "#f1f5f9",
    "unknown":           "#f8fafc",
}

SEGMENT_LABELS = {
    "patient_header":    "Patient Header",
    "drug_list":         "Drug List",
    "dosage_column":     "Dosage / Sig",
    "duration_column":   "Duration",
    "prescriber_footer": "Prescriber",
    "clinical_notes":    "Clinical Notes",
    "annotation":        "Annotations",
    "unknown":           "Unknown",
}


def conf_badge(level: ConfidenceLevel) -> str:
    return f"{CONF_EMOJI[level]} {level.value.capitalize()}"


# ── OCR Panel ────────────────────────────────────────────────────────────────

def render_ocr_panel(ocr: OCRResult):
    st.markdown("### 🔍 OCR / HTR Agent Output")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Word Count", ocr.word_count)
    with col2:
        st.metric("Confidence", conf_badge(ocr.overall_confidence))
    with col3:
        st.metric("Scripts Detected", len(ocr.detected_scripts))

    if ocr.detected_scripts:
        st.markdown(f"**Scripts:** {' · '.join(s.capitalize() for s in ocr.detected_scripts)}")

    if ocr.mixed_script_flags:
        for flag in ocr.mixed_script_flags:
            st.warning(f"⚠️ {flag}")

    st.markdown("**Raw Extracted Text**")
    st.text_area(
        label="raw_text",
        value=ocr.raw_text,
        height=200,
        disabled=True,
        label_visibility="collapsed",
    )

    if ocr.handwritten_regions:
        with st.expander("✍️ Likely Handwritten Regions"):
            for r in ocr.handwritten_regions:
                st.markdown(f"- `{r}`")

    if ocr.printed_regions:
        with st.expander("🖨️ Likely Printed Regions"):
            for r in ocr.printed_regions:
                st.markdown(f"- `{r}`")


# ── Layout Panel ─────────────────────────────────────────────────────────────

def render_layout_panel(layout: LayoutResult, image_bytes: bytes = None):
    st.markdown("### 🗂️ Layout Agent Output")

    st.metric("Overall Confidence", conf_badge(layout.overall_confidence))

    # Draw bounding boxes on image if available
    if image_bytes:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")
            w, h = img.size

            # Full alpha for outlines — fill will be reduced separately
            box_colors = {
                "patient_header":    (59,  130, 246, 255),
                "drug_list":         (34,  197, 94,  255),
                "dosage_column":     (234, 179, 8,   255),
                "duration_column":   (236, 72,  153, 255),
                "prescriber_footer": (139, 92,  246, 255),
                "clinical_notes":    (249, 115, 22,  255),
                "annotation":        (100, 116, 139, 255),
                "unknown":           (148, 163, 184, 255),
            }

            for seg in layout.segments:
                if not seg.bounding_box:
                    continue
                bb = seg.bounding_box
                x0 = int(bb.get("x0", 0) * w)
                y0 = int(bb.get("y0", 0) * h)
                x1 = int(bb.get("x1", 1) * w)
                y1 = int(bb.get("y1", 1) * h)

                outline_rgba = box_colors.get(seg.label, (148, 163, 184, 255))
                fill_rgba    = outline_rgba[:3] + (30,)  # nearly transparent fill

                # Draw box
                draw.rectangle([x0, y0, x1, y1], outline=outline_rgba, width=3, fill=fill_rgba)

                # Draw label as a solid pill above the box
                label_text = SEGMENT_LABELS.get(seg.label, seg.label)
                try:
                    label_x = x0 + 4
                    label_y = max(y0 - 20, 2)
                    text_w  = len(label_text) * 7
                    # Solid background behind the label text
                    draw.rectangle(
                        [label_x - 2, label_y - 2, label_x + text_w, label_y + 14],
                        fill=outline_rgba[:3] + (200,)
                    )
                    draw.text((label_x, label_y), label_text, fill=(255, 255, 255, 255))
                except Exception:
                    pass

            st.image(img, caption="Segment Annotation Overlay", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render bounding box overlay: {e}")

    st.markdown("**Semantic Segments**")

    for seg in layout.segments:
        color = SEGMENT_COLORS.get(seg.label, "#f8fafc")
        label_display = SEGMENT_LABELS.get(seg.label, seg.label)
        conf_icon = CONF_EMOJI[seg.confidence]

        st.markdown(
            f"""
            <div style="
                background:{color};
                border-radius:8px;
                padding:10px 14px;
                margin-bottom:8px;
                border-left:4px solid #94a3b8;
            ">
                <div style="font-weight:600;font-size:0.85rem;color:#374151;">
                    {label_display} {conf_icon}
                </div>
                <div style="font-size:0.9rem;color:#1f2937;margin-top:4px;white-space:pre-wrap;">
                    {seg.raw_text or "<em>empty</em>"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Consolidated fields
    with st.expander("📄 Consolidated Field Text"):
        fields = {
            "Patient Header": layout.patient_header_text,
            "Drug List":      layout.drug_list_text,
            "Dosage/Sig":     layout.dosage_column_text,
            "Prescriber":     layout.prescriber_footer_text,
            "Annotations":    layout.annotation_text,
        }
        for label, text in fields.items():
            if text:
                st.markdown(f"**{label}**")
                st.code(text, language=None)