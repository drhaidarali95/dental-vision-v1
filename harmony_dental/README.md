# Harmony Dental AI: one model for every dental image

Owner decision (Nov 1, 2026): ONE model, one file, for all radiographs, easy to copy,
clone and plug into Harmony. Do not split the work into per-film-type models again.

Owner (Nov 1, 2026): ONE universal model trained on EVERYTHING a clinician uses:
CBCT and other 3D scans, every flat X-ray (periapical, bitewing, occlusal, panoramic),
clinical intraoral photos, STL / intraoral-scan meshes, and any other dental data that can
be trained on. Size and training time do not matter. It must stay one file the owner controls.

`harmony-dental.pt` holds everything:
- `xray`: one 2D detector trained on ALL flat X-rays together (panoramic, periapical,
  bitewing, occlusal): findings (caries, badly decayed, root canal treatment, fillings,
  crowns, implants, impacted teeth, ...) and FDI tooth numbers.
- `cbct`: the 3D CBCT segmentation network (jaws, nerve canals, sinuses, every tooth).
- `photo` (next): intraoral / clinical colour photos (caries, gum disease, plaque, restorations).
- `mesh` (next): STL / PLY / OBJ intraoral scans (tooth segmentation and numbering).

`harmony_model.py` loads the one file and routes each input automatically:
a 3D volume (.mha, .nii, .nii.gz, .nrrd, DICOM folder) goes to the CBCT part; a mesh
(.stl, .ply, .obj) to the mesh part; a colour photo to the photo part; a greyscale flat
image to the X-ray part. One call: `HarmonyDental("harmony-dental.pt").predict(path)`.

Build: `bundle.py --xray xray.pt --cbct cbct.pt --out harmony-dental.pt`.
Training of the X-ray part: `prep_all_xray.py` merges every commercial-safe 2D set into
one datalist; `kaggle/kernel.py` trains it in one job.
