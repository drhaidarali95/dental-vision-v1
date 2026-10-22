"""dental-vision-v1's Faster R-CNN, loaded read-only as the baseline.

Accepts both checkpoint layouts that exist:
  - dental-vision-v1 train.py:   {"model", "num_classes", "categories": {label: {id, name}}}
  - ml/serve_fasterrcnn.py:      {"model", "classes": ["__background__", ...]}

Its class names are translated to ours with labels.canonical(): "Cavity" ->
caries, "Impacted Tooth" -> impacted. Classes we don't learn (fillings,
implant) are dropped. The weights are loaded strictly and never changed.
"""
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

from labels import canonical


class Baseline:
    def __init__(self, path: str, device: torch.device):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "classes" in ckpt:
            names = list(ckpt["classes"])
        else:
            cats = {int(k): v for k, v in ckpt["categories"].items()}
            names = ["__background__"] + [cats[i]["name"] for i in range(1, ckpt["num_classes"])]
        self.names = names
        self.label_to_class = {i: canonical(n) for i, n in enumerate(names) if i > 0}
        self.covers = sorted({c for c in self.label_to_class.values() if c})
        model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, len(names))
        model.load_state_dict(ckpt["model"], strict=True)
        model.roi_heads.score_thresh = 0.02
        self.model = model.eval().to(device)
        self.device = device
        self.epoch = ckpt.get("epoch")

    def predict(self, img: Image.Image) -> list[dict]:
        with torch.inference_mode():
            out = self.model([to_tensor(img).to(self.device)])[0]
        preds = []
        for box, label, score in zip(out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist()):
            name = self.label_to_class.get(int(label))
            if name:
                preds.append({"class": name, "score": float(score), "box": [float(v) for v in box]})
        return preds
