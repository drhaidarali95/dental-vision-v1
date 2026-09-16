"""Geometry primitives for clinician-reviewed implant planning.
This module computes measurements; it does not autonomously approve implant placement.
"""
from dataclasses import dataclass, asdict
from math import sqrt
from typing import Dict, Tuple

Point3D = Tuple[float, float, float]

def distance_mm(a: Point3D, b: Point3D) -> float:
    return sqrt(sum((x-y)**2 for x, y in zip(a, b)))

@dataclass
class ImplantCandidate:
    site: str
    platform_center_mm: Point3D
    apex_mm: Point3D
    diameter_mm: float
    length_mm: float
    nearest_anatomy: Dict[str, float]
    restorative_axis_supported: bool
    confidence: float
    clinician_approval_required: bool = True

    def to_dict(self):
        return asdict(self)

def candidate_from_points(site: str, platform: Point3D, apex: Point3D,
                          diameter_mm: float, nearest_anatomy: Dict[str, float],
                          restorative_axis_supported: bool, confidence: float) -> ImplantCandidate:
    return ImplantCandidate(
        site=site,
        platform_center_mm=platform,
        apex_mm=apex,
        diameter_mm=diameter_mm,
        length_mm=round(distance_mm(platform, apex), 3),
        nearest_anatomy=nearest_anatomy,
        restorative_axis_supported=restorative_axis_supported,
        confidence=confidence,
    )
