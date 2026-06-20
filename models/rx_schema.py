from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class OCRBlock(BaseModel):
    text: str
    bbox: List[float]            # [x0, y0, x1, y1] normalized 0-1
    confidence: float = 0.0
    is_handwritten_guess: bool = False


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"





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
    blocks: List[OCRBlock] = Field(default_factory=list)


class LayoutResult(BaseModel):
    segments: List[LayoutSegment] = Field(default_factory=list)
    patient_header_text: Optional[str] = None
    drug_list_text: Optional[str] = None
    dosage_column_text: Optional[str] = None
    duration_column_text: Optional[str] = None
    prescriber_footer_text: Optional[str] = None
    annotation_text: Optional[str] = None
    overall_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM

