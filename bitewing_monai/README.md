# bitewing_monai

Bitewing X-ray models. Shares the detector code in `../panoramic_monai` and the
COCO prep and test report in `../periapical_monai`.

- Tooth numbering (FDI 13-48) on 8,292 bitewings: Kaggle `sprss96/bitewing-dataset`,
  a mirror of Roboflow `bitewing_allside` (CC BY 4.0).
- Start: `./kaggle/push_kaggle.sh` (status / fetch the same way).

Caries on bitewings needs a labelled bitewing caries set; see `../REGISTRY.md`.
