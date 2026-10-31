# Asset registry: every dataset, model and outside code

The one place to check before you move, license or sell any part of this work.
Update it in the same commit as any change that adds or drops data, models or
outside code. If a row says **VERIFY**, settle it before any commercial use.

Ownership: the code in this repository is the owner's own work unless a row
below says otherwise. Model files are never committed; they live in GitHub
Releases or private Kaggle datasets listed here.

## Datasets

| ID | What | Source (exact copy used) | License | Commercial use | Stored at | Used by |
|---|---|---|---|---|---|---|
| DENTEX | 705 fully labelled panoramics (caries, deep caries, periapical lesion, impacted) + tooth numbering | Zenodo record 7812323, `training_data.zip` (checked 2026-10-22) | Zenodo lists CC BY 4.0. The DENTEX GitHub says CC BY-SA 4.0 and a Hugging Face copy says CC BY-NC-SA 4.0 | Yes per Zenodo, **VERIFY**: get written confirmation from the DENTEX authors before a sale | Private Kaggle notebook output `drhaidarali/dentex-data` (one cached copy, attached by training) | `panoramic_monai`, original v1 notebooks |
| ToothFairy2 | 480 CBCT scans, 42 labelled structures (jaws, nerve canals, sinuses, every tooth, crowns, implants) | ditto.ing.unimore.it/toothfairy2, owner's account | CC BY-SA 4.0 (in the dataset's `dataset.json`) | Yes, with credit. Share-alike: **VERIFY** with a lawyer whether models trained on it must also be CC BY-SA | Private Kaggle dataset `drhaidarali/toothfairy2` | `cbct_toothfairy2` (training since 2026-10-31) |
| v1-detector-data | Data behind the original v1 Faster R-CNN model (Cavity, Fillings, Implant, Impacted Tooth; 73 test images) | **VERIFY**: not recorded in this repo | **VERIFY** | **VERIFY** | Unknown | `ml/serve.py` model |

| DenPAR | 1,000 periapical X-rays: tooth masks, CEJ/apex points, bone levels | Zenodo record 16645076 | Zenodo lists CC BY 4.0; the Scientific Data page lists CC BY-NC-ND 4.0 | Owner chose to use it (2026-10-30). **VERIFY** with the authors before a sale | Queued: private Kaggle copy | periapical model |
| Kaggle Dental Radiography | Low-resolution panoramics; Cavity, Fillings, Implant, Impacted Tooth (Roboflow export) | kaggle.com/datasets/imtkaggleteam/dental-radiography | CC BY-SA 4.0 (Kaggle metadata) | Yes, with credit; share-alike **VERIFY**. Likely the v1 model's training data: owner to confirm | Attach directly on Kaggle | panoramic (fillings, implants) |
| Roboflow Bitewing | 775 bitewing images | universe.roboflow.com/project-hjkow/bitewing-zmohp | Listed CC BY 4.0 (not yet confirmed on page) | **VERIFY** on download; needs owner's ROBOFLOW_API_KEY | Not yet downloaded | bitewing model |
| Roboflow DentalCaries | ~2,000 images: caries, cavity, crack, tooth | universe.roboflow.com/roboflow-tcwui/dentalcaries-zps2h | Listed CC BY 4.0 (not yet confirmed on page) | **VERIFY** on download; needs ROBOFLOW_API_KEY | Not yet downloaded | bitewing / 2D caries |
| Bitewing allside (Kaggle mirror) | 8,292 bitewings with FDI tooth-number boxes (13-48), train/valid/test 5,802/1,659/831 (Roboflow export) | kaggle.com/datasets/sprss96/bitewing-dataset, original universe.roboflow.com/sidharthsujith-carestack-com/bitewing_allside-qod4s | CC BY 4.0 (original Roboflow README) | Yes, with credit | Attach directly on Kaggle | bitewing model |
| Bitewing (ivinlyju) | 100 bitewing images | kaggle.com/datasets/ivinlyju/bitewing | Uploader states Apache 2.0; original source unknown: **VERIFY** provenance | **VERIFY** | Attach directly on Kaggle | bitewing model (if labelled) |
| Bitewing annotations (shubhamskg) | Bitewing images + annotations | kaggle.com/datasets/shubhamskg/bitewing-datasets | Uploader states MIT; original source unknown: **VERIFY** | **VERIFY** | Attach directly on Kaggle | bitewing model |
| MMDental | 1,000 periapical (RVG) + panoramic (OPG) X-rays, 21 classes, 12,000+ annotations | kaggle.com/datasets/suhrathtahra/dental-dataset | CC0 1.0 (public domain) per Kaggle | Yes | Kaggle copy checked 2026-10-31: 796 images only (OPG + Pa_XRays), **no annotations**. Labels must come from the authors' original release: **VERIFY** | not usable for supervised training yet |
| Dental periapical (Roboflow mirror) | Periapical X-rays (Roboflow export) | kaggle.com/datasets/nadaaglan/dental-periapical-x-rays, original universe.roboflow.com/dent-dataset | CC BY 4.0 (original Roboflow README) | Yes, with credit | Attach directly on Kaggle | periapical model |
| Periapical lesions PAI | Periapical lesion scores | huggingface.co/datasets/tirandazdylan/periapical-lesions-pai | CC BY 4.0 | Yes, with credit | Not yet copied | periapical model |
| Caries bounding boxes | Caries boxes | huggingface.co/datasets/Frallex/Caries_Bounding_Box | MIT | Yes | Not yet copied | 2D caries |

Rejected for commercial use (NonCommercial licenses): Mendeley bitewing set,
Teeth3DS, ToothFairy3, Bits2Bites, Bite2Text, IOP-Compass.

Candidates not yet used (check each before use): Zenodo 13772918 (periapical
lesions, CC BY 4.0), Zenodo 14827784 (clinical photo caries; Zenodo says CC BY
4.0 but the paper says CC BY-NC-ND: **VERIFY**), Zenodo 22047269 (gum photos,
CC BY 4.0).

## Models

| ID | Task | Trained on | Where the weights live | Status |
|---|---|---|---|---|
| v1-fasterrcnn | Detector served by `ml/serve.py` | v1-detector-data | Not in repo (`models/harmony_detector_best.pth`) | Exists; source data VERIFY |
| panoramic-findings | MONAI RetinaNet, panoramic findings | DENTEX | Private Kaggle dataset `drhaidarali/harmony-pano-models-v1`, file `pano-findings-v1.pt` (sha256 8d142f23...296) | Trained 2026-10-22. Held-out test mAP@0.5 0.54 (impacted 0.92, caries 0.47, deep caries 0.41, periapical 0.39), 106 DENTEX films; not external/clinical validation |
| panoramic-teeth | MONAI tooth numbering on panoramics | DENTEX | Same dataset, file `pano-teeth-v1.pt` (sha256 7871227c...021e) | Trained 2026-10-22. Held-out test mAP@0.5 0.96, 63 DENTEX films |
| panoramic-findings v2 | Same, trained with image-quality degradation | DENTEX | Private Kaggle dataset `drhaidarali/harmony-pano-models-v2`, file `pano-findings-v2.pt` (sha256 e2654cdc...2bd) | Trained 2026-10-30 08:28 UTC. Held-out 106 films: normal 0.544 (=v1); degraded 0.546 (v1 0.540). Current best (near-tie) |
| panoramic-teeth v2 | Same, with degradation | DENTEX | Same dataset, file `pano-teeth-v2.pt` (sha256 d6b8f89c...f37) | Held-out 0.956 (=v1) |
| periapical-findings v1 | MONAI RetinaNet, periapical: caries, badly decayed, root canal, crown, restoration | Dental periapical (Roboflow mirror) | Private Kaggle dataset `drhaidarali/harmony-periapical-models-v1`, file `periapical-findings-v1.pt` (sha256 a41f8b44...624d) | Trained 2026-10-30. Held-out 54 films mAP@0.5 0.517 (crown 0.87, restoration 0.75, caries 0.38, badly decayed 0.33, RCT 0.25) |
| bitewing-teeth v1 | MONAI RetinaNet, bitewing tooth numbering (FDI 13-48) | Bitewing allside (Kaggle mirror) | Private Kaggle dataset `drhaidarali/harmony-bitewing-models-v1`, file `bitewing-teeth-v1.pt` (sha256 d47e1ade...da87) | Trained 2026-10-30, finished ~18:20 UTC. Held-out 831 bitewings mAP@0.5 0.80 (weakest: 33 0.59, 23 0.64, 18/28 0.69) |
| pano-restorations v1 | MONAI RetinaNet, panoramic: cavity, fillings, impacted tooth, implant | Kaggle Dental Radiography | Private Kaggle dataset `drhaidarali/harmony-pano-restorations-v1`, file `pano-restorations-v1.pt` | Trained 2026-10-31, finished ~02:20 UTC. Held-out 73 films (same split as v1-fasterrcnn) mAP@0.5 0.74 (implant 0.95, impacted 0.81, fillings 0.79, cavity 0.42). v1-fasterrcnn's score on this split not recorded |
| tf2-official-checkpoints | ToothFairy2 team's CBCT models | ToothFairy2 | Private Kaggle dataset `drhaidarali/toothfairy2-checkpoints` (nnU-Net ResEnc-L 3D and VMamba 2D, checkpoint_best.pth) | Saved 2026-10-30. No license file inside: **VERIFY** commercial use with the ToothFairy2 team before relying on them |

## Outside code

| What | License | How used |
|---|---|---|
| MONAI, PyTorch, torchvision | Apache 2.0 / BSD | Libraries |
| nnU-Net | Apache 2.0 | Planned for CBCT |
| ToothFairy2-Benchmark (AImageLab) | Apache 2.0 (`benchmark_networks/LICENSE`) | Reference only, nothing copied yet |

## Credit we must give

ToothFairy2 requires citing three papers on any publication or product page
(see the ToothFairy2 site, "If you use our dataset, you must cite").
DENTEX: cite the DENTEX challenge paper.
