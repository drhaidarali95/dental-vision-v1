# Dental Intelligence — Unified Multimodal Strategy

## Goal
Build one clinician-supervised dental intelligence system that accepts the complete patient record and produces evidence-linked findings, differential diagnoses, supported diagnoses, treatment alternatives, and implant-planning proposals.

This is one product and one clinical output, but not one monolithic neural network. Each modality uses the model family best suited to it, then all outputs are normalized into a shared patient/tooth/site evidence schema and fused by a multimodal reasoning layer.

## Inputs
- 2D radiographs: bitewing, periapical, FMX, panoramic
- 3D radiology: CBCT/DICOM
- Clinical photographs: intraoral and extraoral
- Surface scans: STL/PLY/OBJ and intraoral scans
- Periodontal charting and odontogram
- Medical/dental history, medications, symptoms, prior treatment, notes and reports
- Clinical tests and clinician-entered examination findings

## Core perception tracks
### 1. 2D dental radiology
Targets: tooth localization/numbering, restorations, caries, periapical disease, periodontal bone levels, impacted teeth, endodontic/restorative findings and image-quality/provenance checks.

### 2. CBCT / 3D radiology
Targets: teeth, jaws, inferior alveolar canals, mental foramina, maxillary sinuses, implants/restorations and other relevant anatomy. Outputs must preserve voxel-space geometry and calibrated distances.

### 3. Clinical photographs
Targets: teeth/gingiva/restorations, visible caries or defects, plaque/calculus where supported, gingival inflammation/recession and oral-mucosal abnormalities. Photography cannot replace probing, radiography or histopathology when those are required.

### 4. STL / intraoral surface scans
Targets: tooth segmentation/numbering, missing teeth, arch/occlusal geometry, restorative space, edentulous anatomy, surface change over time and registration with CBCT when available.

### 5. Structured clinical record
Targets: normalize history, symptoms, medications, periodontal chart, vitality/percussion/palpation/mobility/furcation and clinician observations into structured evidence. Preserve source, timestamp and clinician status.

## Shared evidence schema
Every observation should include at minimum:
- patient/case ID
- modality/source
- acquisition timestamp when known
- tooth/site/anatomic region
- finding type
- measurement and units when applicable
- confidence/uncertainty
- geometry or localization reference
- provenance: model, clinician, imported record, or derived measurement
- verification state
- contradiction/missing-data flags

## Fusion and clinical reasoning
Perception models do not directly write a final diagnosis or treatment. Their normalized evidence is fused with chart/history/exam evidence. The reasoning layer produces:
1. abnormalities/findings
2. supporting and conflicting evidence
3. differential diagnoses
4. most-supported diagnostic possibilities with uncertainty
5. missing information or recommended confirmatory tests
6. treatment alternatives and sequencing considerations
7. risks/benefits/contraindications/referral considerations
8. clinician review and confirmation

## Implant planning track
Implant planning is a first-class module of the same system.

Inputs: CBCT + STL/intraoral scan + restorative target + clinical/periodontal/history data.

Functions:
- segment jaw, teeth, IAN canals, mental foramina, sinuses and relevant anatomy
- register CBCT to surface scan when technically valid
- identify edentulous site and restorative envelope
- quantify ridge width/height and available bone
- calculate distances to critical anatomy and neighboring roots/teeth
- evaluate restorative/occlusal space and proposed emergence trajectory
- generate candidate implant positions, axes, diameters and lengths
- show safety distances and uncertainty
- compare candidate plans and identify constraints

The software proposes and visualizes plans; the clinician/surgeon must verify anatomy, measurements, implant system constraints and final placement.

## Training strategy
Do not train everything as one giant network. Train/validate modality-specific perception components in parallel, using shared labels/schema, then train/evaluate fusion and reasoning separately.

Initial public-data tracks to investigate and license-verify before use:
- DENTEX for panoramic radiographic detection
- ToothFairy2 for CBCT multi-structure segmentation
- open clinical-photo datasets for oral mucosa and periodontal appearance
- open dental surface-scan/STL datasets for tooth/arch segmentation and geometry
- multimodal dental datasets that pair imaging with clinical/tabular information

Never mix datasets into a training release until provenance, patient-level splits, permitted use, annotation semantics and license compatibility are documented.

## Validation gates
No component is called clinically validated merely because training completes. Each track requires held-out patient-level evaluation. Metrics depend on task: detection mAP/sensitivity/specificity, segmentation Dice/HD95, landmark/distance error in mm, calibration, failure rate and subgroup/device/site analysis. Implant planning additionally requires geometric validation against expert plans and clinically meaningful safety-distance/trajectory errors.

## Product interface
Lovable receives one normalized API response representing the whole case. UI should show:
- unified case summary
- tooth/site evidence
- source images/3D overlays
- diagnosis/differential with evidence and uncertainty
- treatment alternatives
- implant-planning workspace when applicable
- clinician edits/confirmation and complete audit trail

## Immediate next work
1. Keep existing DENTEX work as the 2D-radiology branch, not the whole product.
2. Add CBCT/ToothFairy2 branch.
3. Add clinical-photo branch.
4. Identify and license-check STL/intraoral-scan datasets.
5. Define the shared evidence schema before training branches diverge.
6. Build patient-level evaluation harnesses for every modality.
7. Only after modality benchmarks are credible, implement multimodal fusion and implant-planning evaluation.
