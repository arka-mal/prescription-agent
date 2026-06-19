from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TimingStructure(BaseModel):
    morning: float = 0
    afternoon: float = 0
    evening: float = 0
    night: float = 0
    frequency_per_day: int = 0
    timing_label: str = ""          # e.g. "BD", "TDS", "OD hs"
    food_relation: Optional[str] = None   # "before food", "after food", "with food"
    duration_days: Optional[int] = None
    duration_label: Optional[str] = None  # e.g. "5 days", "1 week", "continue"


class MedicationEntry(BaseModel):
    original_text: str              # raw text from prescription
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    kb_verified: bool = False 
    form: Optional[str] = None      # tablet, capsule, syrup etc.
    strength: Optional[str] = None  # e.g. "500mg", "250mg/5ml"
    route: Optional[str] = None
    timing: Optional[TimingStructure] = None
    sig_raw: Optional[str] = None   # raw sig string e.g. "1-0-1 pc"
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    flags: List[str] = Field(default_factory=list)  # warnings, ambiguities


class PatientInfo(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    weight: Optional[str] = None
    address: Optional[str] = None
    date: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM


class PrescriberInfo(BaseModel):
    name: Optional[str] = None
    qualification: Optional[str] = None
    registration_no: Optional[str] = None
    clinic: Optional[str] = None
    address: Optional[str] = None
    contact: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM


class LayoutSegment(BaseModel):
    label: str                      # e.g. "patient_header", "drug_list", "prescriber_footer"
    raw_text: str
    bounding_box: Optional[Dict[str, float]] = None  # {x, y, w, h} normalized 0-1
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM


class OCRResult(BaseModel):
    raw_text: str
    detected_scripts: List[str] = Field(default_factory=list)   # e.g. ["latin", "bengali", "devanagari"]
    handwritten_regions: List[str] = Field(default_factory=list)
    printed_regions: List[str] = Field(default_factory=list)
    mixed_script_flags: List[str] = Field(default_factory=list)
    overall_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    word_count: int = 0


class LayoutResult(BaseModel):
    segments: List[LayoutSegment] = Field(default_factory=list)
    patient_header_text: Optional[str] = None
    drug_list_text: Optional[str] = None
    dosage_column_text: Optional[str] = None
    duration_column_text: Optional[str] = None
    prescriber_footer_text: Optional[str] = None
    annotation_text: Optional[str] = None
    overall_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM


class FHIRMedicationRequest(BaseModel):
    """FHIR-lite MedicationRequest — simplified for demo"""
    resource_type: str = "MedicationRequest"
    status: str = "active"
    medication_code: Optional[str] = None      # generic name
    medication_display: Optional[str] = None   # brand name if known
    subject: Optional[str] = None              # patient name
    authored_on: Optional[str] = None          # date
    dosage_text: Optional[str] = None          # human-readable sig
    dose_quantity: Optional[str] = None
    timing_code: Optional[str] = None          # BD, TDS, OD etc.
    route: Optional[str] = None
    additional_instructions: Optional[str] = None


class PrescriptionResult(BaseModel):
    """Final structured output from prescription agent"""
    patient: Optional[PatientInfo] = None
    prescriber: Optional[PrescriberInfo] = None
    medications: List[MedicationEntry] = Field(default_factory=list)
    fhir_requests: List[FHIRMedicationRequest] = Field(default_factory=list)
    global_flags: List[str] = Field(default_factory=list)
    hitl_required: bool = False
    overall_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    processing_notes: List[str] = Field(default_factory=list)
