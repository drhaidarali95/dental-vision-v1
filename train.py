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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.0025)
    p.add_argument("--output", default="artifacts/dental_vision_v1.pt")
    p.add_argument("--resume", default=None, help="Checkpoint to resume from if it exists")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dataset = DentexCocoDataset(args.images, args.annotations, transforms=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    model = build_model(dataset.num_classes).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(args.epochs * 0.60)), max(2, int(args.epochs * 0.85))],
        gamma=0.1,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    resume = Path(args.resume) if args.resume else out
    start_epoch = 0

    if resume.exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        if ckpt.get("num_classes") != dataset.num_classes:
            raise RuntimeError("Checkpoint class count does not match dataset")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"RESUMING from epoch {start_epoch}/{args.epochs}: {resume}")

    if start_epoch >= args.epochs:
        print(f"Training already complete: epoch {start_epoch}/{args.epochs}")
        return

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for images, targets in loader:
            images = [x.to(device) for x in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        scheduler.step()
        avg_loss = running / max(len(loader), 1)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "loss": avg_loss,
            "num_classes": dataset.num_classes,
            "categories": dataset.label_to_category,
        }
        torch.save(checkpoint, out)
        epoch_out = out.with_name(f"{out.stem}_epoch_{epoch+1:02d}{out.suffix}")
        torch.save(checkpoint, epoch_out)
        print(f"epoch={epoch+1}/{args.epochs} loss={avg_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6g} checkpoint={out}", flush=True)

    print(f"saved={out}")


if __name__ == "__main__":
    main()
