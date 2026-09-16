from typing import TypedDict, List, Optional


class BoundingBox(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float


class Finding(TypedDict):
    tooth: Optional[str]
    finding: str
    confidence: float
    bbox: BoundingBox


class DentalVisionResult(TypedDict):
    image_id: str
    findings: List[Finding]
    model_version: str
