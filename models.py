from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceType(str, Enum):
    LLM = "llm"
    FALLBACK = "fallback"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PatientHistory(BaseModel):
    current_medications: List[str] = Field(default_factory=list)
    known_allergies: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    age: Optional[int] = Field(default=None, ge=0, le=130)
    weight: Optional[float] = Field(default=None, gt=0, le=500)

    @field_validator("current_medications", "known_allergies", "conditions", mode="before")
    @classmethod
    def normalize_string_list(cls, v: List[str]) -> List[str]:
        return [s.strip().lower() for s in v if isinstance(s, str) and s.strip()]


class DrugSafetyRequest(BaseModel):
    proposed_medicines: List[str] = Field(
        min_length=1,
        description="List of medicines proposed for the patient (at least one required).",
    )
    patient_history: PatientHistory = Field(default_factory=PatientHistory)

    @field_validator("proposed_medicines", mode="before")
    @classmethod
    def validate_medicines(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("proposed_medicines must not be empty.")
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        if not cleaned:
            raise ValueError("proposed_medicines contains only blank strings.")
        # Deduplicate preserving order (case-insensitive)
        seen: set[str] = set()
        unique: List[str] = []
        for med in cleaned:
            key = med.lower()
            if key not in seen:
                seen.add(key)
                unique.append(med)
        return unique

    @model_validator(mode="after")
    def cross_validate(self) -> DrugSafetyRequest:
        # Warn-level check: medicines shouldn't contain numeric-only tokens
        for med in self.proposed_medicines:
            if med.replace(".", "").replace("-", "").replace(" ", "").isdigit():
                raise ValueError(
                    f"'{med}' does not look like a medicine name."
                )
        return self


# ---------------------------------------------------------------------------
# Sub-response models
# ---------------------------------------------------------------------------

class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: SeverityLevel
    mechanism: str
    clinical_recommendation: str
    source_confidence: str  # "high" | "medium" | "low"


class AllergyAlert(BaseModel):
    medicine: str
    reason: str
    severity: str  # "critical" | "high" | "medium"
    allergy_class: Optional[str] = None  # drug class that triggered the alert


class ContraindicationAlert(BaseModel):
    medicine: str
    condition: str
    reason: str
    severity: SeverityLevel


class RiskScoreBreakdown(BaseModel):
    interaction_score: float = Field(description="Points from drug-drug interactions (max 40)")
    allergy_score: float = Field(description="Points from allergy alerts (max 35)")
    contraindication_score: float = Field(description="Points from condition contraindications (max 25)")
    total: float = Field(description="Aggregated score 0–100")
    explanation: str


# ---------------------------------------------------------------------------
# Main response model
# ---------------------------------------------------------------------------

class DrugSafetyResponse(BaseModel):
    interactions: List[DrugInteraction] = Field(default_factory=list)
    allergy_alerts: List[AllergyAlert] = Field(default_factory=list)
    contraindication_alerts: List[ContraindicationAlert] = Field(default_factory=list)
    safe_to_prescribe: bool
    overall_risk_level: RiskLevel
    requires_doctor_review: bool
    source: SourceType
    cache_hit: bool
    processing_time_ms: int
    # Bonus B
    patient_risk_score: Optional[int] = Field(default=None, ge=0, le=100)
    risk_score_breakdown: Optional[RiskScoreBreakdown] = None
