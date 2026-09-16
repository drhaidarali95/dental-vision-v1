"""Adapters that normalize heterogeneous dental inputs into a common patient evidence manifest."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

SUPPORTED = {
    "radiograph": {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"},
    "cbct": {".nii", ".gz", ".mha", ".mhd", ".nrrd", ".dcm"},
    "photo": {".png", ".jpg", ".jpeg", ".heic", ".tif", ".tiff"},
    "surface_scan": {".stl", ".ply", ".obj"},
    "clinical_record": {".json", ".csv", ".txt"},
}

@dataclass
class ModalityAsset:
    patient_id: str
    modality: str
    path: str
    subtype: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.modality not in SUPPORTED:
            raise ValueError(f"Unsupported modality: {self.modality}")
        p = Path(self.path)
        suffix = ".nii" if p.name.endswith(".nii.gz") else p.suffix.lower()
        if suffix not in SUPPORTED[self.modality]:
            raise ValueError(f"Unsupported {self.modality} format: {p.name}")

@dataclass
class PatientManifest:
    patient_id: str
    assets: List[ModalityAsset] = field(default_factory=list)
    clinical_context: Dict[str, Any] = field(default_factory=dict)

    def add(self, asset: ModalityAsset) -> None:
        if asset.patient_id != self.patient_id:
            raise ValueError("Patient ID mismatch")
        asset.validate()
        self.assets.append(asset)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "assets": [asdict(x) for x in self.assets],
            "clinical_context": self.clinical_context,
        }

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
