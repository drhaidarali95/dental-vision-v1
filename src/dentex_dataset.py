import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class DentexCocoDataset(Dataset):
    """Minimal COCO-style DENTEX object-detection loader for torchvision."""

    def __init__(self, images_dir: str, annotation_file: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms
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
        self.num_classes = len(categories) + 1  # background

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        info = self.images[image_id]
        image = Image.open(self.images_dir / info["file_name"]).convert("RGB")

        boxes, labels, areas, crowds = [], [], [], []
        for ann in self.annotations[image_id]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_id_to_label[ann["category_id"]])
            areas.append(float(ann.get("area", w * h)))
            crowds.append(int(ann.get("iscrowd", 0)))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(crowds, dtype=torch.int64),
        }

        if self.transforms:
            image = self.transforms(image)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
