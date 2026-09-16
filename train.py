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
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--output", default="artifacts/dental_vision_v1.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dataset = DentexCocoDataset(args.images, args.annotations, transforms=T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    model = build_model(dataset.num_classes).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(args.epochs):
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
        print(f"epoch={epoch+1}/{args.epochs} loss={running/max(len(loader),1):.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "num_classes": dataset.num_classes, "categories": dataset.label_to_category}, out)
    print(f"saved={out}")


if __name__ == "__main__":
    main()
