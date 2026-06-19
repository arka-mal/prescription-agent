"""
Pipeline Orchestrator
---------------------
Chains OCR → Layout → Prescription agents in sequence.
Returns all three intermediate results for display in Streamlit.
"""

from dataclasses import dataclass
from typing import Optional
from models.rx_schema import OCRResult, LayoutResult, PrescriptionResult
from agents.ocr_agent import run_ocr_agent
from agents.layout_agent import run_layout_agent
from agents.prescription_agent import run_prescription_agent


@dataclass
class PipelineResult:
    ocr: Optional[OCRResult] = None
    layout: Optional[LayoutResult] = None
    prescription: Optional[PrescriptionResult] = None
    error_stage: Optional[str] = None
    error_message: Optional[str] = None


def run_pipeline(
    image_bytes: bytes,
    groq_api_key: str,
    gcp_credentials_path: Optional[str] = None,
    gcp_credentials_info: Optional[dict] = None,
    groq_model: str = "llama-3.3-70b-versatile",
) -> PipelineResult:
    """
    Full pipeline: image bytes → structured prescription output.

    Args:
        image_bytes: Raw bytes of the prescription image
        groq_api_key: Groq API key for layout + prescription agents
        gcp_credentials_path: Path to GCP service account JSON (local/testing use)
        gcp_credentials_info: GCP service account dict from st.secrets (production use)
        groq_model: Groq model identifier

    Returns:
        PipelineResult with all three agent outputs
    """
    result = PipelineResult()

    # ── Stage 1: OCR / HTR ───────────────────────────────────────────────────
    try:
        result.ocr = run_ocr_agent(
            image_bytes,
            credentials_path=gcp_credentials_path,
            credentials_info=gcp_credentials_info,
        )
    except Exception as e:
        result.error_stage = "OCR Agent"
        result.error_message = str(e)
        return result

    if not result.ocr.raw_text.strip():
        result.error_stage = "OCR Agent"
        result.error_message = "No text could be extracted from the image."
        return result

    # ── Stage 2: Layout Segmentation ─────────────────────────────────────────
    try:
        result.layout = run_layout_agent(
            raw_ocr_text=result.ocr.raw_text,
            groq_api_key=groq_api_key,
            model=groq_model,
        )
    except Exception as e:
        result.error_stage = "Layout Agent"
        result.error_message = str(e)
        return result

    # ── Stage 3: Prescription Extraction + Normalization ─────────────────────
    try:
        result.prescription = run_prescription_agent(
            layout=result.layout,
            groq_api_key=groq_api_key,
            model=groq_model,
        )
    except Exception as e:
        result.error_stage = "Prescription Agent"
        result.error_message = str(e)
        return result

    return result
