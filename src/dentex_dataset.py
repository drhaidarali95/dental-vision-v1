import json
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class DentexCocoDataset(Dataset):
    """COCO-style DENTEX loader with optional detection-safe augmentation."""

    def __init__(self, images_dir: str, annotation_file: str, transforms=None, augment: bool = False):
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.augment = augment
        with open(annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {x["id"]: x for x in coco["images"]}
        self.ids = sorted(self.images)
        self.annotations = {image_id: [] for image_id in self.ids}
        for ann in coco.get("annotations", []):
            if ann["image_id"] in self.annotations:
                self.annotations[ann["image_id"]].append(ann)

        categories = sorted(coco.get("categories", []), key=lambda x: x["id"])
        self.category_id_to_label = {cat["id"]: i + 1 for i, cat in enumerate(categories)}
        self.label_to_category = {i + 1: cat for i, cat in enumerate(categories)}
        self.num_classes = len(categories) + 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        info = self.images[image_id]
        image = Image.open(self.images_dir / info["file_name"]).convert("RGB")
        width, height = image.size

        boxes, labels, areas, crowds = [], [], [], []
        for ann in self.annotations[image_id]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_id_to_label[ann["category_id"]])
            areas.append(float(ann.get("area", w * h)))
            crowds.append(int(ann.get("iscrowd", 0)))

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        crowds = torch.as_tensor(crowds, dtype=torch.int64)

        # Horizontal flip is safe for disease detection because classes are not side-specific.
        if self.augment and random.random() < 0.5:
            image = F.hflip(image)
            if len(boxes):
                old = boxes.clone()
                boxes[:, 0] = width - old[:, 2]
                boxes[:, 2] = width - old[:, 0]

        # Conservative photometric augmentation to reduce memorization of acquisition appearance.
        if self.augment:
            if random.random() < 0.8:
                image = F.adjust_brightness(image, random.uniform(0.85, 1.15))
            if random.random() < 0.8:
                image = F.adjust_contrast(image, random.uniform(0.80, 1.20))
            if random.random() < 0.35:
                image = F.adjust_gamma(image, random.uniform(0.90, 1.10))

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(image_id),
            "area": areas,
            "iscrowd": crowds,
        }

        if self.transforms:
            image = self.transforms(image)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
