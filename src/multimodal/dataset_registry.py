from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    modality: str
    tasks: tuple[str, ...]
    license: str
    commercial_ok: bool
    source: str
    notes: Optional[str] = None

DATASETS = {
    "dentex": DatasetSpec(
        name="DENTEX", modality="xray_panoramic",
        tasks=("tooth_localization", "caries", "deep_caries", "periapical_lesion", "impacted_tooth"),
        license="VERIFY_OFFICIAL_BEFORE_COMMERCIAL_USE", commercial_ok=False,
        source="official DENTEX release",
        notes="License discrepancy across mirrors; hard-block commercial release until resolved.",
    ),
    "toothfairy2": DatasetSpec(
        name="ToothFairy2", modality="cbct",
        tasks=("multi_structure_segmentation", "teeth", "jawbone", "inferior_alveolar_canal", "maxillary_sinus", "implant", "crown", "bridge"),
        license="CC BY-SA", commercial_ok=True,
        source="https://toothfairy2.grand-challenge.org/dataset/",
        notes="Public training set; sign-up required. Preserve attribution/share-alike obligations.",
    ),
    "teeth3ds": DatasetSpec(
        name="Teeth3DS", modality="intraoral_mesh",
        tasks=("tooth_segmentation", "tooth_numbering"),
        license="CC BY-NC-ND 4.0", commercial_ok=False,
        source="https://github.com/abenhamadou/3DTeethSeg_MICCAI_Challenges",
        notes="Research/prototyping only unless separate commercial permission is obtained.",
    ),
}

def commercial_training_set():
    return {k: v for k, v in DATASETS.items() if v.commercial_ok}

def assert_commercial_ok(dataset_key: str) -> None:
    spec = DATASETS[dataset_key]
    if not spec.commercial_ok:
        raise RuntimeError(f"{spec.name} is not cleared for commercial training: {spec.license}")
