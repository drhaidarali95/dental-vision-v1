from dataclasses import dataclass, field

@dataclass
class ImplantCandidate:
    site: str
    center_mm: tuple[float, float, float]
    axis_vector: tuple[float, float, float]
    diameter_mm: float
    length_mm: float
    bone_width_mm: float | None = None
    restorative_space_mm: float | None = None
    distances_mm: dict[str, float] = field(default_factory=dict)
    evidence_sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    clinician_approved: bool = False


def validate_candidate(candidate: ImplantCandidate) -> ImplantCandidate:
    """Structural safety gate only; not a substitute for surgical judgment.

    Numeric clinical thresholds are intentionally not hard-coded here. They
    belong in a separately validated, versioned clinical policy layer.
    """
    required = ("site", "center_mm", "axis_vector", "diameter_mm", "length_mm")
    for key in required:
        if getattr(candidate, key) in (None, ""):
            candidate.warnings.append(f"missing:{key}")
    if not candidate.evidence_sources:
        candidate.warnings.append("missing_provenance")
    candidate.clinician_approved = False
    return candidate
