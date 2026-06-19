"""
Prescription Processing Pipeline
Agentic AI + Neuro-Symbolic AI — Assignment Demo
-------------------------------------------------
Stack:
  OCR/HTR  : Google Cloud Vision (neural perception)
  Layout   : Groq LLM — llama-3.3-70b (neural segmentation)
  Rx Agent : Groq LLM + symbolic sig parser + drug KB (neuro-symbolic)
"""

import os
import json
import tempfile
import streamlit as st
from PIL import Image
import io

from utils.pipeline import run_pipeline
from utils.ui_helpers import render_ocr_panel, render_layout_panel, render_prescription_panel
from config import (
    USE_BACKEND_GROQ_KEY, USE_BACKEND_GCP_CREDS,
    GROQ_API_KEY, GCP_SERVICE_ACCOUNT_INFO, GROQ_MODEL_DEFAULT,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Prescription Processing Pipeline",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a4f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.6rem; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.9rem; }

    .pipeline-step {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        text-align: center;
    }
    .pipeline-step.active { background: #dbeafe; border-color: #3b82f6; }
    .pipeline-step.done   { background: #dcfce7; border-color: #22c55e; }
    .pipeline-step.error  { background: #fee2e2; border-color: #ef4444; }

    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
    div[data-testid="metric-container"] { background: #f8fafc; border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>💊 Prescription Processing Pipeline</h1>
    <p>Neuro-Symbolic AI · OCR/HTR Agent · Layout Agent · Prescription Agent</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: API Keys & Config ────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    if USE_BACKEND_GROQ_KEY:
        groq_key = GROQ_API_KEY
    else:
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free key at console.groq.com",
        )

    gcp_json_upload = None
    if not USE_BACKEND_GCP_CREDS:
        st.markdown("**Google Cloud Vision Credentials**")
        gcp_json_upload = st.file_uploader(
            "Upload GCP service account JSON",
            type=["json"],
            help="Download from GCP Console → IAM → Service Accounts → Keys",
        )

    if USE_BACKEND_GROQ_KEY and USE_BACKEND_GCP_CREDS:
        groq_model = GROQ_MODEL_DEFAULT
    else:
        groq_model = st.selectbox(
            "Groq Model",
            ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"],
            help="llama-3.3-70b recommended for best extraction quality",
        )

    st.divider()
    st.markdown("### 🏗️ Pipeline Architecture")
    st.markdown("""
    ```
    Image Upload
         ↓
    [OCR/HTR Agent]
    Google Vision API
    Neural perception
         ↓
    [Layout Agent]
    Groq LLM
    Field segmentation
         ↓
    [Prescription Agent]
    Groq LLM  ← Neural
    + Sig Parser ← Symbolic
    + Drug KB    ← Symbolic
    + Validator  ← Symbolic
         ↓
    Structured Output
    FHIR-lite JSON
    ```
    """)

    st.divider()
    st.markdown("### ℹ️ Neuro-Symbolic Design")
    st.markdown("""
    **Neural** components handle fuzzy perception:
    - Handwriting recognition
    - Field classification
    - Drug name extraction

    **Symbolic** components enforce structure:
    - Sig parser rule engine
    - Brand → generic KB
    - FHIR schema validator
    - HITL confidence gate
    """)

# ── Main Area ─────────────────────────────────────────────────────────────────

# Resolve GCP credentials: backend secrets take priority, else use manual upload
gcp_credentials_path = None
gcp_credentials_info = None

if USE_BACKEND_GCP_CREDS:
    gcp_credentials_info = GCP_SERVICE_ACCOUNT_INFO
elif gcp_json_upload:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="wb")
    tmp.write(gcp_json_upload.read())
    tmp.flush()
    gcp_credentials_path = tmp.name
    st.sidebar.success("✅ GCP credentials loaded")

# Upload section
st.markdown("### 📤 Upload Prescription")
uploaded_file = st.file_uploader(
    "Upload a prescription image (handwritten or printed)",
    type=["jpg", "jpeg", "png", "tiff", "bmp", "webp"],
    help="Supports handwritten, printed, or mixed prescriptions. Indian formats supported.",
)

if uploaded_file:
    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(uploaded_file, caption="Uploaded Prescription", use_container_width=True)
    with col_info:
        st.markdown("**File Details**")
        st.markdown(f"- **Name:** {uploaded_file.name}")
        st.markdown(f"- **Size:** {uploaded_file.size / 1024:.1f} KB")
        st.markdown(f"- **Type:** {uploaded_file.type}")
        st.markdown("")
        st.info("Configure your API keys in the sidebar, then click **Run Pipeline**.")

    st.divider()

    # Pipeline status display
    status_cols = st.columns(3)
    with status_cols[0]:
        ocr_status = st.empty()
        ocr_status.markdown('<div class="pipeline-step">🔍 OCR / HTR Agent</div>', unsafe_allow_html=True)
    with status_cols[1]:
        layout_status = st.empty()
        layout_status.markdown('<div class="pipeline-step">🗂️ Layout Agent</div>', unsafe_allow_html=True)
    with status_cols[2]:
        rx_status = st.empty()
        rx_status.markdown('<div class="pipeline-step">💊 Prescription Agent</div>', unsafe_allow_html=True)

    st.markdown("")
    run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    if run_btn:
        # Validate keys
        if not groq_key:
            st.error("❌ Groq API key is not configured.")
            st.stop()
        if not gcp_credentials_path and not gcp_credentials_info:
            st.error("❌ GCP credentials are not configured.")
            st.stop()

        image_bytes = uploaded_file.getvalue()

        # ── Run pipeline with live status updates ─────────────────────────────
        with st.spinner(""):
            # Stage 1 indicator
            ocr_status.markdown('<div class="pipeline-step active">🔍 OCR / HTR Agent ⏳</div>', unsafe_allow_html=True)

            from agents.ocr_agent import run_ocr_agent
            from agents.layout_agent import run_layout_agent
            from agents.prescription_agent import run_prescription_agent

            ocr_result = None
            layout_result = None
            rx_result = None
            pipeline_error = None

            # Stage 1: OCR
            try:
                ocr_result = run_ocr_agent(
                    image_bytes,
                    credentials_path=gcp_credentials_path,
                    credentials_info=gcp_credentials_info,
                )
                ocr_status.markdown('<div class="pipeline-step done">🔍 OCR / HTR Agent ✅</div>', unsafe_allow_html=True)
            except Exception as e:
                ocr_status.markdown('<div class="pipeline-step error">🔍 OCR / HTR Agent ❌</div>', unsafe_allow_html=True)
                st.error(f"OCR Agent failed: {e}")
                st.stop()

            # Stage 2: Layout
            layout_status.markdown('<div class="pipeline-step active">🗂️ Layout Agent ⏳</div>', unsafe_allow_html=True)
            try:
                layout_result = run_layout_agent(
                    raw_ocr_text=ocr_result.raw_text,
                    groq_api_key=groq_key,
                    model=groq_model,
                )
                layout_status.markdown('<div class="pipeline-step done">🗂️ Layout Agent ✅</div>', unsafe_allow_html=True)
            except Exception as e:
                layout_status.markdown('<div class="pipeline-step error">🗂️ Layout Agent ❌</div>', unsafe_allow_html=True)
                st.error(f"Layout Agent failed: {e}")
                st.stop()

            # Stage 3: Prescription
            rx_status.markdown('<div class="pipeline-step active">💊 Prescription Agent ⏳</div>', unsafe_allow_html=True)
            try:
                rx_result = run_prescription_agent(
                    layout=layout_result,
                    groq_api_key=groq_key,
                    model=groq_model,
                )
                rx_status.markdown('<div class="pipeline-step done">💊 Prescription Agent ✅</div>', unsafe_allow_html=True)
            except Exception as e:
                rx_status.markdown('<div class="pipeline-step error">💊 Prescription Agent ❌</div>', unsafe_allow_html=True)
                st.error(f"Prescription Agent failed: {e}")
                st.stop()

        st.success("✅ Pipeline complete!")
        st.divider()

        # ── Display results in tabs ───────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs([
            "🔍 OCR / HTR Agent",
            "🗂️ Layout Agent",
            "💊 Prescription Agent",
        ])

        with tab1:
            if ocr_result:
                render_ocr_panel(ocr_result)

        with tab2:
            if layout_result:
                render_layout_panel(layout_result)

        with tab3:
            if rx_result:
                render_prescription_panel(rx_result)

else:
    # Landing state
    st.markdown("""
    <div style="
        text-align:center;
        padding:3rem 1rem;
        color:#94a3b8;
        border:2px dashed #e2e8f0;
        border-radius:12px;
        margin-top:1rem;
    ">
        <div style="font-size:3rem;">📋</div>
        <div style="font-size:1.1rem;margin-top:0.5rem;">Upload a prescription image to begin</div>
        <div style="font-size:0.85rem;margin-top:0.4rem;">
            Supports handwritten · printed · mixed script (Bengali/Hindi + English)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### 🧠 How it works")

    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        st.markdown("""
        **🔍 OCR / HTR Agent**

        Google Cloud Vision extracts text from the prescription image.
        Handles handwritten and printed text, detects mixed scripts
        (Bengali, Hindi, English), and returns bounding box data
        for each text region.
        """)
    with exp2:
        st.markdown("""
        **🗂️ Layout Agent**

        A Groq LLM segments the raw text into semantic zones:
        patient header, drug list, dosage column, prescriber footer,
        and annotations. This is the neural field-detection step.
        """)
    with exp3:
        st.markdown("""
        **💊 Prescription Agent**

        Combines neural extraction (Groq LLM) with symbolic reasoning:
        a rule-based sig parser normalizes dosage instructions,
        a drug KB resolves brand → generic names, and a validator
        gates confidence and flags fields for human review.
        """)
