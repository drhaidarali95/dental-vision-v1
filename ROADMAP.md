# Model roadmap (owner-approved 2026-10-22)

One specialist model per kind of record, plus a reasoning layer that combines
their findings into suggested diagnoses for the dentist to confirm. Flat X-ray
types share models. Order: finish 1-3, then 4-6, then the reasoning layer.

| # | Model | Reads | Data (see REGISTRY.md) | Status |
|---|---|---|---|---|
| 1 | 2D X-ray findings | Panoramic, bitewing, periapical | DENTEX (panoramic); Zenodo 13772918 periapical lesions (CC BY 4.0, 0.9 GB); bitewing: no commercial-safe public set, needs own clinic data | Panoramic training |
| 2 | X-ray tooth numbering | All flat X-rays | DENTEX | Panoramic training |
| 3 | CBCT | Teeth, nerve canals, sinuses, jaws, implants | ToothFairy2 (CC BY-SA 4.0) | Data downloading |
| 4 | Clinical photos | Caries, gum health, restorations | Zenodo 14827784 caries photos (CC BY 4.0 on Zenodo, paper says NC-ND: VERIFY, 1.6 GB); Zenodo 22047269 MIO gum photos (CC BY 4.0, 2.6 GB) | Next |
| 5 | Intraoral 3D scans | Tooth numbering, missing teeth, bite, wear | No commercial-safe public set; owner's own scans (~1,000 patients, with consent) | Waiting on data |
| 6 | Perio chart + history | Risk from pockets, bleeding, history | Harmony's own clinician-entered records | Later |
| + | Reasoning layer | Combines 1-6 into suggested diagnoses | Outputs of 1-6 | Last |

Rules: commercial-safe data only, held-out test scores for every model,
nothing presented as a diagnosis without a dentist confirming it.
