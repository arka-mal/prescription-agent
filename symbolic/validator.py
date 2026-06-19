"""
Symbolic Validator
------------------
Deterministic rule engine for:
- Confidence gating (flag low-confidence fields for HITL)
- Basic dose sanity checks
- Missing field detection
- FHIR-lite schema validation

This is the symbolic post-processing layer after LLM extraction.
"""

from typing import List, Tuple
from models.rx_schema import (
    PrescriptionResult, MedicationEntry, ConfidenceLevel
)

# Confidence threshold below which HITL is triggered
HITL_THRESHOLD = ConfidenceLevel.LOW

# Fields that must be present for a valid prescription
REQUIRED_PATIENT_FIELDS = ["name", "age"]
REQUIRED_MED_FIELDS = ["brand_name", "generic_name"]

# Dose sanity: max tablets per dose for common forms
MAX_DOSE_SANITY = {
    "tablet": 4.0,
    "capsule": 4.0,
    "syrup": 3.0,   # in tablespoons
}


def _confidence_rank(c: ConfidenceLevel) -> int:
    return {"high": 2, "medium": 1, "low": 0}.get(c.value, 0)


def validate_and_flag(result: PrescriptionResult) -> PrescriptionResult:
    """
    Run all symbolic validation rules on the extracted prescription.
    Mutates result in-place: adds flags, sets hitl_required, adds processing notes.
    """
    flags = list(result.global_flags)
    notes = list(result.processing_notes)
    hitl = result.hitl_required

    # ── Patient field checks ──────────────────────────────────────────────────
    if result.patient:
        for field in REQUIRED_PATIENT_FIELDS:
            if not getattr(result.patient, field, None):
                flags.append(f"Missing patient field: {field}")
                hitl = True
        if result.patient.confidence == ConfidenceLevel.LOW:
            flags.append("Low confidence on patient information")
            hitl = True
    else:
        flags.append("Patient information not extracted")
        hitl = True

    # ── Medication checks ─────────────────────────────────────────────────────
    if not result.medications:
        flags.append("No medications extracted — HITL required")
        hitl = True

    for i, med in enumerate(result.medications):
        med_label = med.brand_name or med.generic_name or f"Drug #{i+1}"

        # Generic name resolution failed
        if not med.generic_name and not med.brand_name:
            med.flags.append("Drug name unclear")
            hitl = True

        # Timing not parsed
        if not med.timing or not med.timing.frequency_per_day:
            if med.timing and med.timing.timing_label:
                med.flags.append(f"Sig not fully parsed: '{med.timing.timing_label}'")
            else:
                med.flags.append("Dosage/timing not extracted")
            hitl = True

        # Dose sanity check
        if med.timing and med.form:
            form_key = med.form.lower()
            for form, max_dose in MAX_DOSE_SANITY.items():
                if form in form_key:
                    doses = [
                        med.timing.morning,
                        med.timing.afternoon,
                        med.timing.evening,
                        med.timing.night,
                    ]
                    for dose in doses:
                        if dose > max_dose:
                            med.flags.append(
                                f"Unusually high single dose ({dose} {form}) — verify"
                            )
                            hitl = True

        # Low confidence on medication
        if med.confidence == ConfidenceLevel.LOW:
            med.flags.append("Low OCR/extraction confidence")
            hitl = True

        # No duration specified (warn, don't require HITL)
        if med.timing and not med.timing.duration_days and not med.timing.duration_label:
            if med.timing.timing_label not in ("SOS", "STAT", ""):
                med.flags.append("No duration specified")

    # ── Prescriber checks ─────────────────────────────────────────────────────
    if not result.prescriber or not result.prescriber.name:
        flags.append("Prescriber information not found")

    # ── Overall confidence ────────────────────────────────────────────────────
    low_conf_meds = sum(
        1 for m in result.medications if m.confidence == ConfidenceLevel.LOW
    )
    if low_conf_meds > len(result.medications) / 2 and result.medications:
        flags.append(f"{low_conf_meds}/{len(result.medications)} medications have low confidence")
        hitl = True

    if hitl:
        notes.append("This prescription requires human-in-the-loop review before dispensing.")

    result.global_flags = flags
    result.processing_notes = notes
    result.hitl_required = hitl
    return result


def compute_overall_confidence(result: PrescriptionResult) -> ConfidenceLevel:
    """Compute an aggregate confidence score from all components."""
    scores = []

    if result.patient:
        scores.append(_confidence_rank(result.patient.confidence))
    if result.prescriber:
        scores.append(_confidence_rank(result.prescriber.confidence))
    for med in result.medications:
        scores.append(_confidence_rank(med.confidence))

    if not scores:
        return ConfidenceLevel.LOW

    avg = sum(scores) / len(scores)
    if avg >= 1.8:
        return ConfidenceLevel.HIGH
    elif avg >= 1.0:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW
