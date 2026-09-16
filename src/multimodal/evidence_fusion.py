"""Patient-level fusion contract. Models contribute evidence; this layer never invents observations."""
from collections import defaultdict
from typing import Any, Dict, Iterable, List

REQUIRED = {"source_modality", "finding", "confidence", "provenance"}

def validate_evidence(item: Dict[str, Any]) -> None:
    missing = REQUIRED - set(item)
    if missing:
        raise ValueError(f"Evidence missing fields: {sorted(missing)}")
    c = float(item["confidence"])
    if not 0.0 <= c <= 1.0:
        raise ValueError("confidence must be 0..1")

def fuse_evidence(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(items)
    for item in items:
        validate_evidence(item)

    by_tooth: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    general: List[Dict[str, Any]] = []
    for item in items:
        tooth = item.get("tooth")
        (by_tooth[str(tooth)] if tooth else general).append(item)

    modalities = sorted({i["source_modality"] for i in items})
    return {
        "modalities_present": modalities,
        "tooth_evidence": dict(by_tooth),
        "general_evidence": general,
        "evidence_count": len(items),
        "diagnosis_status": "requires_reasoning_and_clinician_review",
    }
