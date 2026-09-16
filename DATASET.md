# DENTEX dataset notes

Primary sources checked before training:

- Official DENTEX GitHub repository: states DENTEX data is CC BY-SA 4.0 and repository code is MIT.
- DENTEX Grand Challenge data page: 693 quadrant-labeled panoramics, 634 quadrant+enumeration panoramics, 1005 fully annotated quadrant+enumeration+diagnosis panoramics, plus 1571 unlabeled panoramics.
- Fully annotated set: 705 train, 50 validation, 250 test; public ground truth availability differs by split.
- Diagnostic classes: caries, deep caries, periapical lesions, impacted teeth.
- Zenodo record 7812323: training_data.zip ~10.9 GB and validation_data.zip ~149.5 MB.

## License warning
The official GitHub repository states CC BY-SA 4.0, while a Hugging Face dataset card currently reports CC BY-NC-SA 4.0. Do not treat commercial-use rights as settled until the provenance/license of the exact downloaded copy is documented and, if necessary, clarified with the dataset authors.

## Expected format
DENTEX derivatives and challenge implementations commonly expose COCO-style JSON annotations. `src/dentex_dataset.py` consumes COCO `images`, `annotations`, and `categories`, with bounding boxes in `[x,y,width,height]` form.

## Clinical scope
This dataset supports radiographic evidence extraction on panoramic X-rays. It does not establish autonomous diagnosis, periodontal probing depths, BOP, symptoms, medical history, or treatment decisions.
