"""Preprocessing primitives for Dental Vision multimodal inputs.

This module deliberately separates modality-specific preprocessing from clinical
reasoning. It produces normalized artifacts plus metadata/provenance; it does
not make diagnoses.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PreprocessedArtifact:
    patient_id: str
    modality: str
    source_path: str
    normalized_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


SUPPORTED = {
    "xray": {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"},
    "cbct": {".dcm", ".nii", ".nii.gz", ".nrrd", ".mha"},
    "photo": {".png", ".jpg", ".jpeg", ".heic", ".tif", ".tiff"},
    "surface_scan": {".stl", ".ply", ".obj"},
    "clinical_record": {".json", ".csv", ".txt"},
}


def validate_artifact(patient_id: str, modality: str, path: str) -> PreprocessedArtifact:
    if modality not in SUPPORTED:
        raise ValueError(f"Unsupported modality: {modality}")
    p = Path(path)
    name = p.name.lower()
    valid = any(name.endswith(ext) for ext in SUPPORTED[modality])
    if not valid:
        raise ValueError(f"Unsupported {modality} file: {p.name}")
    return PreprocessedArtifact(
        patient_id=patient_id,
        modality=modality,
        source_path=str(p),
        metadata={"filename": p.name},
    )


def preprocess_manifest(patient_id: str, items: List[Dict[str, str]]) -> List[PreprocessedArtifact]:
    """Validate a patient-level manifest before modality-specific loaders run."""
    return [validate_artifact(patient_id, item["modality"], item["path"]) for item in items]
