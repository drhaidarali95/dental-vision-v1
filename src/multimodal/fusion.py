from dataclasses import dataclass, field
from typing import Any

@dataclass
class Evidence:
    modality: str
    source_id: str
    finding: str
    anatomy: str | None = None
    tooth: str | None = None
    confidence: float | None = None
    measurements: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

@dataclass
class PatientCase:
    patient_id: str
    history: dict[str, Any] = field(default_factory=dict)
    exam: dict[str, Any] = field(default_factory=dict)
    perio_chart: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)

@dataclass
class DiagnosticAssessment:
    abnormalities: list[dict[str, Any]] = field(default_factory=list)
    differential: list[dict[str, Any]] = field(default_factory=list)
    supported_diagnoses: list[dict[str, Any]] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    treatment_alternatives: list[dict[str, Any]] = field(default_factory=list)
    implant_plans: list[dict[str, Any]] = field(default_factory=list)
    requires_clinician_confirmation: bool = True


def fuse_case(case: PatientCase) -> dict[str, Any]:
    """Deterministic normalization layer before the clinical reasoning model.

    This does not diagnose. It produces a patient-level evidence packet with
    provenance so the reasoning layer can integrate modalities without losing
    where each observation came from.
    """
    return {
        "patient_id": case.patient_id,
        "history": case.history,
        "exam": case.exam,
        "perio_chart": case.perio_chart,
        "evidence": [e.__dict__ for e in case.evidence],
        "modalities_present": sorted({e.modality for e in case.evidence}),
        "requires_clinician_confirmation": True,
    }
