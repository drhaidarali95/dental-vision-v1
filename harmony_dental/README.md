# Harmony Dental AI: one model for every dental image

Owner decision (Nov 1, 2026): ONE model, one file, for all radiographs, easy to copy,
clone and plug into Harmony. Do not split the work into per-film-type models again.

`harmony-dental.pt` holds everything:
- `xray`: one 2D detector trained on ALL flat X-rays together (panoramic, periapical,
  bitewing, occlusal): findings (caries, badly decayed, root canal treatment, fillings,
  crowns, implants, impacted teeth, ...) and FDI tooth numbers.
- `cbct`: the 3D CBCT segmentation network (jaws, nerve canals, sinuses, every tooth).

`harmony_model.py` loads the one file and routes each input automatically:
a 3D volume (.mha, .nii, .nii.gz, .nrrd, DICOM folder) goes to the CBCT part; any
flat image goes to the X-ray part. One call: `HarmonyDental("harmony-dental.pt").predict(path)`.

Build: `bundle.py --xray xray.pt --cbct cbct.pt --out harmony-dental.pt`.
Training of the X-ray part: `prep_all_xray.py` merges every commercial-safe 2D set into
one datalist; `kaggle/kernel.py` trains it in one job.
