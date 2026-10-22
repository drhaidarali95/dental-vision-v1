"""MONAI RetinaNet for panoramic films, plus the shared image handling.

The backbone is torchvision's ResNet50 + FPN (ImageNet weights) wrapped in
MONAI's RetinaNet / RetinaNetDetector, which supply anchors, ATSS matching,
hard-negative sampling, losses and NMS.

Every film is resized so its long side is LONG_SIDE pixels before the model
sees it; boxes are scaled back to the original pixels afterwards.
"""
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import RetinaNet
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from labels import CLASSES

# Training size; set HARMONY_LONG_SIDE=1024 for CPU-only runs. A loaded
# checkpoint carries its own size (detector.long_side), used by predict().
LONG_SIDE = int(os.environ.get("HARMONY_LONG_SIDE", "1536"))
SIZE_DIVISIBLE = 32
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# FPN levels P3-P6 (stride 8, 16, 32, 64; torchvision always adds the max-pool
# level). Base shapes are (width, height) in model pixels at P3 and double per
# level: small caries up to whole impacted molars.
FEATURE_MAP_SCALES = (1, 2, 4, 8)
BASE_ANCHOR_SHAPES = ((24, 24), (32, 20), (20, 32), (40, 40), (56, 36), (36, 56))


def build_detector(pretrained_backbone: bool = True, classes: list[str] = CLASSES) -> RetinaNetDetector:
    """RetinaNet for `classes`: the four findings by default, or the 32 teeth."""
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None,
        trainable_layers=5,
        returned_layers=[2, 3, 4],
    )
    anchors = AnchorGeneratorWithAnchorShape(
        feature_map_scales=FEATURE_MAP_SCALES, base_anchor_shapes=BASE_ANCHOR_SHAPES
    )
    network = RetinaNet(
        spatial_dims=2,
        num_classes=len(classes),
        num_anchors=anchors.num_anchors_per_location()[0],
        feature_extractor=backbone,
        size_divisible=SIZE_DIVISIBLE,
    )
    detector = RetinaNetDetector(
        network=network, anchor_generator=anchors, spatial_dims=2,
        num_classes=len(classes), size_divisible=SIZE_DIVISIBLE,
    )
    detector.classes = list(classes)
    detector.long_side = LONG_SIDE
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_atss_matcher(num_candidates=4, center_in_gt=False)
    detector.set_hard_negative_sampler(batch_size_per_image=64, positive_fraction=0.3, pool_size=20, min_neg=16)
    detector.set_box_selector_parameters(
        score_thresh=0.02, topk_candidates_per_level=1000, nms_thresh=0.3, detections_per_img=150,
    )
    return detector


def load_detector(path: str, device: torch.device, expect: list[str] | None = CLASSES) -> tuple[RetinaNetDetector, dict]:
    """Load a checkpoint. `expect` guards against loading the wrong model
    (findings vs teeth); pass None to accept whatever classes it holds."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    classes = list(ckpt.get("classes", []))
    if expect is not None and classes != list(expect):
        raise RuntimeError(f"checkpoint classes {classes} != {list(expect)}")
    detector = build_detector(pretrained_backbone=False, classes=classes)
    detector.long_side = int(ckpt.get("long_side", 1536))
    detector.network.load_state_dict(ckpt["network"], strict=True)
    detector.eval().to(device)
    return detector, ckpt


def save_detector(detector: RetinaNetDetector, path: str, **meta) -> None:
    torch.save({"network": detector.network.state_dict(), "classes": detector.classes,
                "architecture": "monai_retinanet_resnet50_fpn", "long_side": detector.long_side, **meta}, path)


def open_film(source) -> Image.Image:
    """Path or file-like -> upright 8-bit RGB image (EXIF rotation applied)."""
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I;16B", "I"):
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(arr, (0.5, 99.5))
        arr = np.clip((arr - lo) / max(hi - lo, 1.0), 0, 1) * 255
        img = Image.fromarray(arr.astype(np.uint8))
    return img.convert("RGB")


def swap_xy(boxes: torch.Tensor) -> torch.Tensor:
    """[x1, y1, x2, y2] <-> [y1, x1, y2, x2].

    MONAI orders box coordinates by array axis, and a (C, H, W) tensor puts
    rows (y) first. Our files and the wire use x first, so boxes are swapped
    on the way into and out of the detector.
    """
    return boxes[..., [1, 0, 3, 2]]


def to_model_input(img: Image.Image, long_side: int | None = None) -> tuple[torch.Tensor, float]:
    """RGB film -> normalised (3, H, W) tensor at long_side (default LONG_SIDE), and the scale used."""
    scale = (long_side or LONG_SIDE) / max(img.size)
    w, h = round(img.width * scale), round(img.height * scale)
    gray = img.convert("L").resize((w, h), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
    return (t - MEAN) / STD, scale


def predict(detector: RetinaNetDetector, img: Image.Image, device: torch.device) -> list[dict]:
    """One film -> [{class, score, box:[x1,y1,x2,y2] in original pixels}], all scores."""
    x, scale = to_model_input(img, getattr(detector, "long_side", None))
    with torch.inference_mode():
        out = detector([x.to(device)], use_inferer=False)[0]
    boxes = swap_xy(out[detector.target_box_key].float().cpu()) / scale
    labels = out[detector.target_label_key].cpu().tolist()
    scores = out[detector.pred_score_key].float().cpu().tolist()
    return [{"class": detector.classes[int(l)], "score": float(s), "box": [float(v) for v in b]}
            for b, l, s in zip(boxes.tolist(), labels, scores)]


def panoramic_check(img: Image.Image, min_aspect: float, max_aspect: float, min_width: int) -> str | None:
    """Return a reason if this does not look like a panoramic film, else None.

    Harmony sends only the image (no film type), so this goes by shape and
    colour: panoramics are wide (about 2:1), large and greyscale. Bitewings,
    periapicals and cephs are near-square or tall; photos are in colour.
    """
    w, h = img.size
    aspect = w / max(h, 1)
    if not (min_aspect <= aspect <= max_aspect):
        return f"aspect ratio {aspect:.2f} is outside the panoramic range {min_aspect}-{max_aspect}"
    if w < min_width:
        return f"width {w}px is below the panoramic minimum {min_width}px"
    small = np.asarray(img.resize((256, max(1, round(256 / aspect)))), dtype=np.int16)
    chroma = np.abs(small[..., 0] - small[..., 1]).mean() + np.abs(small[..., 1] - small[..., 2]).mean()
    if chroma > 12:
        return "image is in colour; panoramic radiographs are greyscale"
    return None
