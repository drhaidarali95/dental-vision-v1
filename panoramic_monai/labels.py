"""Class names shared by every ml-monai script.

The MONAI model learns the four DENTEX diagnosis classes. Harmony's detector
contract (src/lib/detector-contract.ts) only knows "cavity", "fillings",
"implant" and "impacted tooth", so the server translates on the way out:

  caries, deep_caries -> "cavity"
  impacted            -> "impacted tooth"
  periapical_lesion   -> not sent to Harmony; written to the server's own log
"""

CLASSES = ["caries", "deep_caries", "periapical_lesion", "impacted"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}

WIRE_CATEGORY = {
    "caries": "cavity",
    "deep_caries": "cavity",
    "impacted": "impacted tooth",
}
LOG_ONLY = {"periapical_lesion"}

# Spellings seen in DENTEX copies and in the dental-vision-v1 checkpoints.
ALIASES = {
    "caries": "caries",
    "cavity": "caries",
    "deep caries": "deep_caries",
    "deep_caries": "deep_caries",
    "periapical lesion": "periapical_lesion",
    "periapical_lesion": "periapical_lesion",
    "impacted": "impacted",
    "impacted tooth": "impacted",
    "impacted_tooth": "impacted",
}


def canonical(name: str) -> str | None:
    """DENTEX / baseline class name -> one of CLASSES, or None if not a diagnosis class."""
    return ALIASES.get(str(name).strip().lower().replace("-", " "))


# ---------------------------------------------------------------- anatomy
# Teeth in FDI notation: quadrant (1 upper right, 2 upper left, 3 lower left,
# 4 lower right; patient's side) then position 1 (central incisor) to 8.
TEETH = [f"{q}{t}" for q in (1, 2, 3, 4) for t in range(1, 9)]
TOOTH_INDEX = {name: i for i, name in enumerate(TEETH)}

_QUADRANT = {1: "upper right", 2: "upper left", 3: "lower left", 4: "lower right"}
_POSITION = {1: "central incisor", 2: "lateral incisor", 3: "canine", 4: "first premolar",
             5: "second premolar", 6: "first molar", 7: "second molar", 8: "third molar (wisdom tooth)"}


def tooth_name(fdi: str) -> str:
    """'36' -> 'Lower left first molar'."""
    q, t = int(fdi[0]), int(fdi[1])
    return f"{_QUADRANT[q]} {_POSITION[t]}".capitalize()


def mirror_tooth(index: int) -> int:
    """Class index after a left-right flip of the film: 16 <-> 26, 36 <-> 46."""
    q, t = TEETH[index]
    return TOOTH_INDEX[{"1": "2", "2": "1", "3": "4", "4": "3"}[q] + t]
