"""
UI Rendering Helpers
--------------------
Streamlit display functions for each agent's output panel.
Kept separate from app.py to keep the main file clean.
"""

import streamlit as st
from models.rx_schema import (
    OCRResult, LayoutResult, PrescriptionResult, ConfidenceLevel
)
from symbolic.sig_parser import sig_to_human_readable


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
    "patient_header":    "👤 Patient Header",
    "drug_list":         "💊 Drug List",
    "dosage_column":     "⚖️ Dosage / Sig",
    "duration_column":   "📅 Duration",
    "prescriber_footer": "🏥 Prescriber",
    "clinical_notes":    "📋 Clinical Notes",
    "annotation":        "✏️ Annotations",
    "unknown":           "❓ Unknown",
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

def render_layout_panel(layout: LayoutResult):
    st.markdown("### 🗂️ Layout Agent Output")

    st.metric("Overall Confidence", conf_badge(layout.overall_confidence))

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
            "Drug List": layout.drug_list_text,
            "Dosage/Sig": layout.dosage_column_text,
            "Prescriber": layout.prescriber_footer_text,
            "Annotations": layout.annotation_text,
        }
        for label, text in fields.items():
            if text:
                st.markdown(f"**{label}**")
                st.code(text, language=None)


# ── Prescription Panel ────────────────────────────────────────────────────────

def render_prescription_panel(rx: PrescriptionResult):
    st.markdown("### 💊 Prescription Agent Output")

    # HITL banner
    if rx.hitl_required:
        st.error("🚨 **Human-in-the-Loop Review Required** — One or more fields require pharmacist/clinician verification.")
    else:
        st.success("✅ Prescription processed with sufficient confidence.")

    # Overall confidence
    st.metric("Overall Confidence", conf_badge(rx.overall_confidence))

    # Global flags
    if rx.global_flags:
        with st.expander("⚠️ Global Flags"):
            for flag in rx.global_flags:
                st.markdown(f"- {flag}")

    # Patient
    if rx.patient:
        st.markdown("#### 👤 Patient Information")
        p = rx.patient
        cols = st.columns(3)
        cols[0].markdown(f"**Name:** {p.name or '—'}")
        cols[1].markdown(f"**Age:** {p.age or '—'}")
        cols[2].markdown(f"**Gender:** {p.gender or '—'}")
        if p.weight:
            st.markdown(f"**Weight:** {p.weight}")
        if p.date:
            st.markdown(f"**Date:** {p.date}")
        st.caption(f"Confidence: {conf_badge(p.confidence)}")

    st.divider()

    # Medications
    st.markdown("#### 💊 Medications")
    if not rx.medications:
        st.warning("No medications extracted.")
    else:
        for i, med in enumerate(rx.medications):
            conf_color = CONF_COLOR[med.confidence]
            with st.container():
                unverified_badge = (
                    '<span style="font-size:0.7rem;font-weight:700;color:#92400e;'
                    'background:#fef3c7;border-radius:4px;padding:2px 6px;margin-left:8px;">'
                    '⚠ UNVERIFIED</span>'
                ) if not med.kb_verified else ''

                st.markdown(
                    f"""
                    <div style="
                        border:1px solid {conf_color};
                        border-radius:8px;
                        padding:12px 16px;
                        margin-bottom:12px;
                    ">
                        <div style="font-size:1rem;font-weight:700;color:#111827;">
                            {i+1}. {med.brand_name or med.generic_name or med.original_text}{unverified_badge}
                        </div>
                        {f'<div style="font-size:0.82rem;color:#6b7280;">Generic: <b>{med.generic_name}</b></div>' if med.generic_name and med.brand_name else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"**Form:** {med.form or '—'}")
                c2.markdown(f"**Strength:** {med.strength or '—'}")
                c3.markdown(f"**Route:** {med.route or '—'}")
                c4.markdown(f"**Conf:** {conf_badge(med.confidence)}")

                # Timing breakdown
                if med.timing:
                    t = med.timing
                    st.markdown(f"**Sig (raw):** `{med.sig_raw or '—'}`")
                    st.markdown(f"**Normalized:** {sig_to_human_readable(t)}")

                    if t.frequency_per_day > 0:
                        dose_cols = st.columns(4)
                        dose_cols[0].metric("Morning", t.morning)
                        dose_cols[1].metric("Afternoon", t.afternoon)
                        dose_cols[2].metric("Evening", t.evening)
                        dose_cols[3].metric("Night", t.night)

                    if t.food_relation:
                        st.caption(f"🍽️ {t.food_relation.capitalize()}")
                    if t.duration_days:
                        st.caption(f"📅 Duration: {t.duration_days} days")

                if med.flags:
                    for flag in med.flags:
                        st.warning(f"⚠️ {flag}")

    st.divider()

    # Prescriber
    if rx.prescriber:
        st.markdown("#### 🏥 Prescriber")
        pr = rx.prescriber
        cols = st.columns(2)
        cols[0].markdown(f"**Name:** {pr.name or '—'}")
        cols[1].markdown(f"**Qualification:** {pr.qualification or '—'}")
        if pr.registration_no:
            st.markdown(f"**Reg No:** {pr.registration_no}")
        if pr.clinic:
            st.markdown(f"**Clinic:** {pr.clinic}")
        if pr.contact:
            st.markdown(f"**Contact:** {pr.contact}")
        st.caption(f"Confidence: {conf_badge(pr.confidence)}")

    st.divider()

    # FHIR Output
    with st.expander("📦 FHIR-lite MedicationRequest Output (JSON)"):
        fhir_list = [r.model_dump(exclude_none=True) for r in rx.fhir_requests]
        st.json(fhir_list)

    # Processing notes
    if rx.processing_notes:
        with st.expander("📝 Processing Notes"):
            for note in rx.processing_notes:
                if note:
                    st.markdown(f"- {note}")
