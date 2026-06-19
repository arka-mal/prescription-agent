"""
Prescription Agent (Orchestrator)
----------------------------------
Neural component: Groq LLM for structured extraction
Symbolic component: sig_parser + drug_resolver + validator

Pipeline:
  LayoutResult → LLM extraction → symbolic normalization → validated PrescriptionResult
"""

import json
import re
from groq import Groq
from models.rx_schema import (
    PrescriptionResult, MedicationEntry, PatientInfo, PrescriberInfo,
    FHIRMedicationRequest, ConfidenceLevel, TimingStructure
)
from models.rx_schema import LayoutResult
from symbolic.sig_parser import parse_sig, sig_to_human_readable
from symbolic.drug_resolver import resolve_brand, normalize_form, normalize_route
from symbolic.validator import validate_and_flag, compute_overall_confidence


PRESCRIPTION_SYSTEM_PROMPT = """You are a medical prescription extraction agent specializing in Indian clinical contexts.

You receive segmented prescription text (already split into zones by a layout agent).
Your job is to extract structured information from each zone.

EXTRACTION RULES:
1. Drug names: extract as-is (brand names like "Augmentin", "Crocin", "Pan D" — do NOT convert)
2. Sigs: extract VERBATIM — "1-0-1 pc", "BD x 5 days", "TDS after food", "½ tab OD hs"
3. Patient: name, age, gender, weight, date
4. Prescriber: doctor name, qualification (MBBS/MD etc), reg no, clinic name
5. If a field is unclear or ambiguous, set confidence to "low"
6. If text contains non-English (Bengali/Hindi), try to extract meaning if possible, else note it

Respond ONLY with valid JSON in this exact format:
{
  "patient": {
    "name": "...", "age": "...", "gender": "...", "weight": "...",
    "address": "...", "date": "...", "confidence": "high|medium|low"
  },
  "prescriber": {
    "name": "...", "qualification": "...", "registration_no": "...",
    "clinic": "...", "address": "...", "contact": "...", "confidence": "high|medium|low"
  },
  "medications": [
    {
      "original_text": "...",
      "drug_name": "...",
      "form": "tab|cap|syr|...",
      "strength": "...",
      "sig_raw": "...",
      "route": "...",
      "confidence": "high|medium|low",
      "notes": "..."
    }
  ],
  "global_notes": "any overall observations"
}
"""


def _conf(s: str) -> ConfidenceLevel:
    return {"high": ConfidenceLevel.HIGH, "medium": ConfidenceLevel.MEDIUM}.get(
        (s or "").lower(), ConfidenceLevel.LOW
    )


def _build_fhir(med: MedicationEntry, patient_name: str = None) -> FHIRMedicationRequest:
    """Build a FHIR-lite MedicationRequest from a normalized MedicationEntry."""
    timing_label = ""
    if med.timing:
        timing_label = sig_to_human_readable(med.timing)

    return FHIRMedicationRequest(
        medication_code=med.generic_name,
        medication_display=med.brand_name or med.generic_name,
        subject=patient_name,
        dosage_text=timing_label or med.sig_raw,
        dose_quantity=med.strength,
        timing_code=med.timing.timing_label if med.timing else None,
        route=med.route,
        additional_instructions=", ".join(med.flags) if med.flags else None,
    )


def run_prescription_agent(
    layout: LayoutResult,
    groq_api_key: str,
    model: str = "llama-3.3-70b-versatile"
) -> PrescriptionResult:
    """
    Main prescription agent.

    Args:
        layout: LayoutResult from layout agent
        groq_api_key: Groq API key
        model: Groq model

    Returns:
        Fully validated PrescriptionResult
    """
    client = Groq(api_key=groq_api_key)

    # Build context message from layout segments
    context_parts = []
    if layout.patient_header_text:
        context_parts.append(f"PATIENT HEADER:\n{layout.patient_header_text}")
    if layout.drug_list_text:
        context_parts.append(f"DRUG LIST:\n{layout.drug_list_text}")
    if layout.dosage_column_text:
        context_parts.append(f"DOSAGE/SIG:\n{layout.dosage_column_text}")
    if layout.prescriber_footer_text:
        context_parts.append(f"PRESCRIBER:\n{layout.prescriber_footer_text}")
    if layout.annotation_text:
        context_parts.append(f"ANNOTATIONS/NOTES:\n{layout.annotation_text}")

    # Also include all raw segment text
    for seg in layout.segments:
        if seg.label not in ("patient_header", "drug_list", "dosage_column", "prescriber_footer"):
            context_parts.append(f"{seg.label.upper()}:\n{seg.raw_text}")

    user_message = "\n\n".join(context_parts) if context_parts else "No segmented text available."
    user_message += "\n\nExtract all prescription fields as JSON."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PRESCRIPTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=3000,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return PrescriptionResult(
            global_flags=["LLM extraction failed — JSON parse error"],
            hitl_required=True,
            overall_confidence=ConfidenceLevel.LOW,
            processing_notes=["Prescription agent returned malformed JSON"],
        )

    # ── Build PatientInfo ─────────────────────────────────────────────────────
    patient = None
    if data.get("patient"):
        p = data["patient"]
        patient = PatientInfo(
            name=p.get("name"),
            age=p.get("age"),
            gender=p.get("gender"),
            weight=p.get("weight"),
            address=p.get("address"),
            date=p.get("date"),
            confidence=_conf(p.get("confidence", "medium")),
        )

    # ── Build PrescriberInfo ──────────────────────────────────────────────────
    prescriber = None
    if data.get("prescriber"):
        pr = data["prescriber"]
        prescriber = PrescriberInfo(
            name=pr.get("name"),
            qualification=pr.get("qualification"),
            registration_no=pr.get("registration_no"),
            clinic=pr.get("clinic"),
            address=pr.get("address"),
            contact=pr.get("contact"),
            confidence=_conf(pr.get("confidence", "medium")),
        )

    # ── Build MedicationEntry list (neuro-symbolic) ───────────────────────────
    medications = []
    for m in data.get("medications", []):
        drug_name_raw = m.get("drug_name", "")
        sig_raw = m.get("sig_raw", "")

        # SYMBOLIC: brand → generic resolution
        brand, generic, kb_verified = resolve_brand(drug_name_raw)

        # SYMBOLIC: sig parsing → structured timing
        timing = parse_sig(sig_raw) if sig_raw else TimingStructure()

        # SYMBOLIC: normalize form and route
        form = normalize_form(m.get("form", ""))
        route = normalize_route(m.get("route", ""))

        med_flags = [m["notes"]] if m.get("notes") else []

        reported_conf = _conf(m.get("confidence", "medium"))
        if not kb_verified:
            med_flags.append(
                f"Drug name '{drug_name_raw}' not found in drug knowledge base — "
                "unverified, may be an OCR/extraction error"
            )
            if reported_conf == ConfidenceLevel.HIGH:
                reported_conf = ConfidenceLevel.MEDIUM

        med = MedicationEntry(
            original_text=m.get("original_text", drug_name_raw),
            brand_name=brand,
            generic_name=generic,
            kb_verified=kb_verified,
            form=form or m.get("form"),
            strength=m.get("strength"),
            route=route or m.get("route"),
            timing=timing,
            sig_raw=sig_raw,
            confidence=reported_conf,
            flags=med_flags,
        )
        medications.append(med)

    result = PrescriptionResult(
        patient=patient,
        prescriber=prescriber,
        medications=medications,
        processing_notes=[data.get("global_notes", "")] if data.get("global_notes") else [],
    )

    # ── SYMBOLIC: validate and flag ───────────────────────────────────────────
    result = validate_and_flag(result)

    # ── Build FHIR-lite output ────────────────────────────────────────────────
    patient_name = patient.name if patient else None
    result.fhir_requests = [_build_fhir(med, patient_name) for med in medications]

    # ── Compute overall confidence ────────────────────────────────────────────
    result.overall_confidence = compute_overall_confidence(result)

    return result
