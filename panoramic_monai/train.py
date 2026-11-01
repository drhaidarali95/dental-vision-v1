"""Train the MONAI RetinaNet panoramic detector (findings or teeth).

  python train.py --datalist data/datalist.json --out runs/teacher
  python train.py --datalist data/datalist.json --drafts runs/drafts.json --out runs/student
  python train.py --datalist data/teeth.json --out runs/teeth      # anatomy layer

The class list comes from the datalist ("classes"): the four findings, or the
32 FDI teeth from prepare_teeth.py.

Draft labels (--drafts) are added to the TRAINING split only; val and test stay
expert-labelled. The checkpoint with the best val mAP@0.5 is kept as best.pt,
and report_val.json records its scores. Test is not touched here.
"""
import argparse
import io
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

from labels import CLASSES, TEETH, mirror_tooth
from metrics import coco_metrics
from model import build_detector, open_film, predict, save_detector, swap_xy, to_model_input


def degrade(img: Image.Image) -> Image.Image:
    """Randomly make a film look older, scanned or lower quality, so the model
    learns to read poor films too. Box coordinates are unchanged (same size)."""
    if random.random() < 0.3:  # low-resolution film or sensor, scaled back up
        f = random.uniform(0.35, 0.8)
        small = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.BILINEAR)
        img = small.resize(img.size, Image.BILINEAR)
    if random.random() < 0.25:  # soft focus / patient movement
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.0)))
    if random.random() < 0.3:  # exposure differences between machines
        g = random.uniform(0.7, 1.4)
        img = img.point(lambda v: int(255 * (v / 255) ** g))
    if random.random() < 0.25:  # sensor / scanner noise
        a = np.asarray(img, dtype=np.float32)
        a += np.random.normal(0, random.uniform(3, 12), a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if random.random() < 0.3:  # heavy JPEG compression (emailed or exported films)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(20, 70))
        img = Image.open(io.BytesIO(buf.getvalue())).convert(img.mode)
    return img


def name_mirror(classes: list[str]):
    """For any class list of FDI tooth numbers ("16", "26", ...): the index remap
    for a left-right flip, or None when the classes aren't side-specific.
    Raises if a tooth's mirror is missing, since a flip would then mislabel it."""
    side = {"1": "2", "2": "1", "3": "4", "4": "3"}
    fdi = [c for c in classes if len(c) == 2 and c.isdigit() and c[0] in side]
    if not fdi:
        return None
    index = {c: i for i, c in enumerate(classes)}
    table = [index.get(side[c[0]] + c[1], -1) if c in fdi else i for i, c in enumerate(classes)]
    if -1 in table:
        missing = [c for c, t in zip(classes, table) if t == -1]
        raise SystemExit(f"tooth classes without a mirror tooth (flip would mislabel): {missing}")
    return lambda i: table[i]


class Films(Dataset):
    def __init__(self, items: list[dict], augment: bool, mirror=None):
        self.items = items
        self.augment = augment
        self.mirror = mirror  # class remap for a left-right flip (teeth), or None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        img = open_film(item["image"])
        boxes = torch.tensor(item["box"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(item["label"], dtype=torch.long)
        if self.augment:
            # Left-right flip. Findings don't depend on side; tooth numbers do,
            # so for teeth the labels are mirrored too (16 <-> 26).
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                boxes[:, [0, 2]] = img.width - boxes[:, [2, 0]]
                if self.mirror:
                    labels = torch.tensor([self.mirror(int(l)) for l in labels], dtype=torch.long)
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.25))
            img = degrade(img)
        x, scale = to_model_input(img)
        return x, {"box": swap_xy(boxes * scale), "label": labels}


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def evaluate(detector, items, device) -> tuple[dict, list]:
    detector.eval()
    preds = [predict(detector, open_film(it["image"]), device) for it in items]
    return coco_metrics(items, preds, detector.classes), preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datalist", required=True, type=Path)
    p.add_argument("--drafts", type=Path, help="draft-label JSON from make_drafts.py")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-items", type=int, default=0, help="smoke tests only")
    p.add_argument("--no-pretrained", action="store_true", help="smoke tests only (no download)")
    args = p.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = json.loads(args.datalist.read_text())
    classes = data["classes"]
    if not classes or not all(isinstance(c, str) for c in classes):
        raise SystemExit(f"bad class list in datalist: {classes}")
    mirror = mirror_tooth if classes == TEETH else name_mirror(classes)
    train_items, val_items = data["train"], data["val"]
    drafts = []
    if args.drafts:
        drafts = json.loads(args.drafts.read_text())["items"]
        held = {it["image"] for it in val_items + data["test"]}
        drafts = [d for d in drafts if d["image"] not in held]
    # Optional: repeat training films that contain weak classes (env HARMONY_OVERSAMPLE="name1,name2", x HARMONY_OVERSAMPLE_X)
    weak = [c.strip().lower() for c in os.environ.get("HARMONY_OVERSAMPLE", "").split(",") if c.strip()]
    if weak:
        idx = {i for i, c in enumerate(classes) if c.lower() in weak}
        times = int(os.environ.get("HARMONY_OVERSAMPLE_X", "3"))
        extra = [it for it in train_items if idx & set(it["label"])] * (times - 1)
        print(f"oversampling {len(extra)} extra films with {sorted(classes[i] for i in idx)}", flush=True)
        train_items = train_items + extra
    if args.max_items:
        train_items, val_items = train_items[: args.max_items], val_items[: args.max_items]
    print(f"train {len(train_items)} expert + {len(drafts)} draft images, val {len(val_items)}", flush=True)

    loader = DataLoader(Films(train_items + drafts, augment=True, mirror=mirror), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, collate_fn=collate, drop_last=len(train_items + drafts) > args.batch_size)
    detector = build_detector(pretrained_backbone=not args.no_pretrained, classes=classes).to(device)
    optimizer = torch.optim.AdamW(detector.network.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, args.epochs * len(loader))
    warmup = min(500, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.out.mkdir(parents=True, exist_ok=True)
    best, history, first = -1.0, [], 1
    resume = args.out / "resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        detector.network.load_state_dict(state["network"])
        optimizer.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        best, history, first = state["best"], state["history"], state["epoch"] + 1
        print(f"resuming after epoch {state['epoch']} (best val mAP@0.5 {best:.4f})", flush=True)
    deadline = time.time() + float(os.environ.get("HARMONY_HOURS", "0")) * 3600 if os.environ.get("HARMONY_HOURS") else None
    for epoch in range(first, args.epochs + 1):
        if deadline and time.time() > deadline:
            print(f"time budget reached before epoch {epoch}; stopping", flush=True)
            break
        detector.train()
        started, total = time.time(), 0.0
        for images, targets in loader:
            images = [x.to(device) for x in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, enabled=use_amp):
                losses = detector(images, targets)
                loss = losses[detector.cls_key] + losses[detector.box_reg_key]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(detector.network.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            sched.step()
            total += float(loss.detach())
        row = {"epoch": epoch, "loss": total / max(1, len(loader)), "seconds": round(time.time() - started)}

        if epoch % args.val_every == 0 or epoch == args.epochs:
            metrics, _ = evaluate(detector, val_items, device)
            row["val_mAP_50"] = metrics["mAP_50"]
            if metrics["mAP_50"] > best:
                best = metrics["mAP_50"]
                save_detector(detector, str(args.out / "best.pt"), epoch=epoch, val=metrics,
                              trained_on={"expert": len(train_items), "drafts": len(drafts)})
                (args.out / "report_val.json").write_text(json.dumps({"epoch": epoch, **metrics}, indent=2))
        save_detector(detector, str(args.out / "last.pt"), epoch=epoch)
        history.append(row)
        # Everything needed to carry on if the machine stops mid-run.
        torch.save({"network": detector.network.state_dict(), "optimizer": optimizer.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best": best, "history": history}, str(resume) + ".tmp")
        Path(str(resume) + ".tmp").replace(resume)
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
    print(f"best val mAP@0.5 = {best:.4f} -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
