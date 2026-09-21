import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.dentex_dataset import DentexCocoDataset, collate_fn


def build_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def set_backbone_trainable(model, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--freeze-backbone-epochs", type=int, default=4)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--output", default="artifacts/dental_vision_v1.pt")
    p.add_argument("--resume", default=None)
    p.add_argument("--save-every-epoch", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    base_tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    dataset = DentexCocoDataset(args.images, args.annotations, transforms=base_tf, augment=args.augment)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    model = build_model(dataset.num_classes).to(device)

    start_epoch = 0
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    resume = Path(args.resume) if args.resume else None
    if resume and resume.exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        if ckpt.get("num_classes") != dataset.num_classes:
            raise RuntimeError("Checkpoint class count does not match dataset")
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"RESUMING model weights from epoch {start_epoch}: {resume}")

    frozen = start_epoch < args.freeze_backbone_epochs
    set_backbone_trainable(model, not frozen)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - start_epoch), eta_min=args.lr * 0.05)

    for epoch in range(start_epoch, args.epochs):
        should_freeze = epoch < args.freeze_backbone_epochs
        if should_freeze != frozen:
            frozen = should_freeze
            set_backbone_trainable(model, not frozen)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr * 0.5, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch), eta_min=args.lr * 0.05)
            print(f"BACKBONE {'FROZEN' if frozen else 'UNFROZEN'} at epoch {epoch+1}")

        model.train()
        running = 0.0
        for images, targets in loader:
            images = [x.to(device) for x in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += float(loss.detach())

        scheduler.step()
        avg_loss = running / max(len(loader), 1)
        checkpoint = {
            "model": model.state_dict(), "epoch": epoch + 1, "loss": avg_loss,
            "num_classes": dataset.num_classes, "categories": dataset.label_to_category,
            "v3": {"augment": args.augment, "freeze_backbone_epochs": args.freeze_backbone_epochs}
        }
        torch.save(checkpoint, out)
        if args.save_every_epoch:
            torch.save(checkpoint, out.with_name(f"{out.stem}_epoch_{epoch+1:02d}{out.suffix}"))
        print(f"epoch={epoch+1}/{args.epochs} loss={avg_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6g} backbone_frozen={frozen}", flush=True)

    print(f"saved={out}")


if __name__ == "__main__":
    main()
