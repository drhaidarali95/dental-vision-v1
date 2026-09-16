from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ClinicalFinding:
    finding_id: str
    source_modality: str
    finding_type: str
    confidence: float
    provenance: Dict[str, Any]
    tooth: Optional[str] = None
    site: Optional[str] = None
    value: Any = None
    units: Optional[str] = None
    uncertainty: Optional[str] = None

    def validate(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.provenance:
            raise ValueError("provenance is required")


@dataclass
class PatientEvidence:
    patient_case_id: str
    modalities: List[str] = field(default_factory=list)
    findings: List[ClinicalFinding] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)

    def add(self, finding: ClinicalFinding) -> None:
        finding.validate()
        self.findings.append(finding)
        if finding.source_modality not in self.modalities:
            self.modalities.append(finding.source_modality)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_case_id": self.patient_case_id,
            "modalities": self.modalities,
            "findings": [asdict(f) for f in self.findings],
            "missing_information": self.missing_information,
            "clinician_confirmation_required": True,
        }
