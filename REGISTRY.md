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
| DENTEX | 705 fully labelled panoramics (caries, deep caries, periapical lesion, impacted) + tooth numbering | Zenodo record 7812323, `training_data.zip` (checked 2026-10-22) | Zenodo lists CC BY 4.0. The DENTEX GitHub says CC BY-SA 4.0 and a Hugging Face copy says CC BY-NC-SA 4.0 | Yes per Zenodo, **VERIFY**: get written confirmation from the DENTEX authors before a sale | Downloaded fresh on each Kaggle run, not kept | `panoramic_monai`, original v1 notebooks |
| ToothFairy2 | 480 CBCT scans, 42 labelled structures (jaws, nerve canals, sinuses, every tooth, crowns, implants) | ditto.ing.unimore.it/toothfairy2, owner's account | CC BY-SA 4.0 (in the dataset's `dataset.json`) | Yes, with credit. Share-alike: **VERIFY** with a lawyer whether models trained on it must also be CC BY-SA | Private Kaggle dataset `drhaidarali/toothfairy2` (being created) | `cbct_toothfairy2` (planned) |
| v1-detector-data | Data behind the original v1 Faster R-CNN model (Cavity, Fillings, Implant, Impacted Tooth; 73 test images) | **VERIFY**: not recorded in this repo | **VERIFY** | **VERIFY** | Unknown | `ml/serve.py` model |

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
| panoramic-findings | MONAI RetinaNet, panoramic findings | DENTEX | Private Kaggle notebook output `drhaidarali/harmony-ml-monai-train`, then a draft GitHub Release | Training |
| panoramic-teeth | MONAI tooth numbering on panoramics | DENTEX | Same as above | Training |
| tf2-official-checkpoints | ToothFairy2 team's CBCT models | ToothFairy2 | Private Kaggle dataset `drhaidarali/toothfairy2` | Downloading; their own license **VERIFY** before use |

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
