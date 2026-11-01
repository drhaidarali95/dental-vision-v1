"""Harmony Dental AI: one model file for every dental image.

  from harmony_model import HarmonyDental
  m = HarmonyDental("harmony-dental.pt")
  m.predict("pano.jpg")        # flat X-ray -> findings + tooth numbers (boxes)
  m.predict("scan.nii.gz")     # CBCT -> labelled 3D structures

The input type is detected automatically. Output is for dentist review only.
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "panoramic_monai"))
VOLUME_EXT = (".mha", ".mhd", ".nii", ".nii.gz", ".nrrd", ".dcm")


class HarmonyDental:
    def __init__(self, path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        b = torch.load(path, map_location="cpu", weights_only=False)
        if b.get("format") != "harmony-dental":
            raise ValueError("not a harmony-dental model file")
        self.version, self.scores = b.get("version"), b.get("scores", {})
        self._xray_ckpt, self._cbct_ckpt = b["xray"], b["cbct"]
        self._xray = self._cbct = None

    @staticmethod
    def is_volume(path) -> bool:
        p = Path(path)
        return p.is_dir() or str(p).lower().endswith(VOLUME_EXT)

    def predict(self, path, min_score: float = 0.3):
        return self.read_cbct(path) if self.is_volume(path) else self.read_xray(path, min_score)

    def read_xray(self, path, min_score: float = 0.3):
        from model import build_detector, open_film, predict
        if self._xray is None:
            c = self._xray_ckpt
            d = build_detector(pretrained_backbone=False, classes=list(c["classes"]))
            d.long_side = int(c.get("long_side", 1024))
            d.network.load_state_dict(c["network"])
            self._xray = d.eval().to(self.device)
        found = [f for f in predict(self._xray, open_film(path), self.device) if f["score"] >= min_score]
        return {"type": "xray", "findings": found}

    def read_cbct(self, path):
        import SimpleITK as sitk
        from monai.inferers import sliding_window_inference
        from monai.networks.nets import SegResNet
        c = self._cbct_ckpt
        if self._cbct is None:
            net = SegResNet(spatial_dims=3, in_channels=1, out_channels=len(c["classes"]), init_filters=16,
                            blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))
            net.load_state_dict(c["state_dict"])
            self._cbct = net.eval().to(self.device)
        p = Path(path)
        if p.is_dir():
            r = sitk.ImageSeriesReader(); r.SetFileNames(r.GetGDCMSeriesFileNames(str(p))); img = r.Execute()
        else:
            img = sitk.ReadImage(str(p))
        sp = tuple(c.get("spacing", (1.0, 1.0, 1.0)))
        size = [int(round(s * o / n)) for s, o, n in zip(img.GetSize(), img.GetSpacing(), sp)]
        f = sitk.ResampleImageFilter(); f.SetOutputSpacing(sp); f.SetSize(size)
        f.SetOutputDirection(img.GetDirection()); f.SetOutputOrigin(img.GetOrigin())
        arr = sitk.GetArrayFromImage(f.Execute(img)).astype(np.float32)
        x = torch.from_numpy((np.clip(arr, -1000, 3000) + 1000) / 4000)[None, None].to(self.device)
        with torch.inference_mode():
            lab = sliding_window_inference(x, tuple(c.get("patch", (128, 128, 128))), 2, self._cbct, overlap=0.25).argmax(1)[0].cpu().numpy()
        names = c["classes"]
        vox = float(np.prod(sp))
        found = [{"structure": names[i], "volume_ml": round(float((lab == i).sum()) * vox / 1000, 2)}
                 for i in np.unique(lab) if i > 0]
        return {"type": "cbct", "spacing_mm": sp, "labels": lab, "structures": found}
