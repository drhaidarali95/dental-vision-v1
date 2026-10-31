"""CBCT segmentation v1 on ToothFairy2 (480 scans, 42 structures) within one
Kaggle GPU session.

Scans (~0.3 mm, ~370 MB each) are resampled once to 1.0 mm and cached, a 3D
SegResNet (MONAI) is trained on random 128^3 crops for a fixed time budget,
and the best checkpoint by validation Dice is scored on a held-out test set.

  python train_cbct.py --data /kaggle/input/toothfairy2/Dataset112_ToothFairy2 --work /kaggle/tmp/cbct --out /kaggle/working

Outputs in --out: cbct/best.pt, cbct/history.json, report_cbct_test.json.
Dice per structure is reported; nothing here is a clinical validation.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import SegResNet

SPACING = (1.0, 1.0, 1.0)
PATCH = (128, 128, 128)
SEED = 20261031


def resample(img: sitk.Image, nearest: bool) -> np.ndarray:
    size = [int(round(sz * sp / ns)) for sz, sp, ns in zip(img.GetSize(), img.GetSpacing(), SPACING)]
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(SPACING)
    f.SetSize(size)
    f.SetOutputDirection(img.GetDirection())
    f.SetOutputOrigin(img.GetOrigin())
    f.SetInterpolator(sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear)
    return sitk.GetArrayFromImage(f.Execute(img))  # z, y, x


def preprocess(data: Path, work: Path, lut: np.ndarray) -> list[Path]:
    work.mkdir(parents=True, exist_ok=True)
    out = []
    images = sorted((data / "imagesTr").glob("*_0000.mha"))
    t0 = time.time()
    for n, img_path in enumerate(images, 1):
        case = img_path.name.replace("_0000.mha", "")
        dst = work / f"{case}.npz"
        if not dst.exists():
            img = resample(sitk.ReadImage(str(img_path)), nearest=False)
            lab = resample(sitk.ReadImage(str(data / "labelsTr" / f"{case}.mha")), nearest=True)
            img = np.clip(img, -1000, 3000).astype(np.int16)
            np.savez_compressed(dst, img=img, lab=lut[np.clip(lab, 0, len(lut) - 1)].astype(np.uint8))
        out.append(dst)
        if n % 20 == 0:
            print(f"preprocessed {n}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def load(p: Path):
    z = np.load(p)
    img = (z["img"].astype(np.float32) + 1000.0) / 4000.0  # -> [0, 1]
    return img, z["lab"]


def crop(img, lab, fg_prob=0.7):
    pad = [(0, max(0, P - s)) for P, s in zip(PATCH, img.shape)]
    img, lab = np.pad(img, pad), np.pad(lab, pad)
    if random.random() < fg_prob and lab.any():
        zz, yy, xx = np.nonzero(lab)
        i = random.randrange(len(zz))
        c = (zz[i], yy[i], xx[i])
    else:
        c = tuple(random.randrange(s) for s in img.shape)
    start = [min(max(0, ci - P // 2), s - P) for ci, P, s in zip(c, PATCH, img.shape)]
    sl = tuple(slice(st, st + P) for st, P in zip(start, PATCH))
    return img[sl], lab[sl]


def dice_per_class(pred, lab, n_cls):
    out = {}
    for c in range(1, n_cls):
        p, g = pred == c, lab == c
        if g.sum() == 0 and p.sum() == 0:
            continue
        out[c] = float(2 * (p & g).sum() / (p.sum() + g.sum() + 1e-6))
    return out


def evaluate(model, cases, n_cls, dev):
    model.eval()
    per = {c: [] for c in range(1, n_cls)}
    with torch.inference_mode(), torch.autocast("cuda", enabled=dev.type == "cuda"):
        for p in cases:
            img, lab = load(p)
            x = torch.from_numpy(img)[None, None].to(dev)
            logits = sliding_window_inference(x, PATCH, 2, model, overlap=0.25)
            pred = logits.argmax(1)[0].cpu().numpy()
            for c, d in dice_per_class(pred, lab, n_cls).items():
                per[c].append(d)
    mean = {c: float(np.mean(v)) for c, v in per.items() if v}
    return (float(np.mean(list(mean.values()))) if mean else 0.0), mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hours", type=float, default=7.5, help="training time budget")
    ap.add_argument("--test", type=int, default=48)
    ap.add_argument("--val", type=int, default=24)
    a = ap.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)

    labels = json.loads((a.data / "dataset.json").read_text())["labels"]
    real = sorted((v, k) for k, v in labels.items() if not k.startswith("NA"))  # drop unused NA ids
    lut = np.zeros(max(v for v, _ in real) + 1, dtype=np.uint8)
    names = []
    for new, (old, name) in enumerate(real):
        lut[old] = new
        names.append(name)
    n_cls = len(names)
    print(f"{n_cls} classes incl. background", flush=True)

    cases = preprocess(a.data, a.work, lut)
    rng = random.Random(SEED)
    order = cases[:]
    rng.shuffle(order)
    test, val, train = order[:a.test], order[a.test:a.test + a.val], order[a.test + a.val:]
    print(f"train {len(train)}, val {len(val)}, test {len(test)}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegResNet(spatial_dims=3, in_channels=1, out_channels=n_cls, init_filters=16,
                      blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)).to(dev)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    out = a.out / "cbct"
    out.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + a.hours * 3600
    best, history, step = -1.0, [], 0
    while time.time() < deadline:
        model.train()
        t0, losses = time.time(), []
        for _ in range(250):  # one "epoch" = 250 patches of batch 2
            batch = [crop(*load(random.choice(train))) for _ in range(2)]
            x = torch.from_numpy(np.stack([b[0] for b in batch]))[:, None].to(dev)
            y = torch.from_numpy(np.stack([b[1] for b in batch]).astype(np.int64))[:, None].to(dev)
            if random.random() < 0.5:  # intensity jitter for scanner variety
                x = (x * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)).clamp(0, 1)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss))
            step += 1
            if time.time() > deadline:
                break
        dice, _ = evaluate(model, val, n_cls, dev)
        history.append({"epoch": len(history) + 1, "loss": float(np.mean(losses)), "val_dice": dice,
                        "seconds": round(time.time() - t0)})
        print(json.dumps(history[-1]), flush=True)
        if dice > best:
            best = dice
            torch.save({"state_dict": model.state_dict(), "classes": names, "spacing": SPACING,
                        "patch": PATCH, "architecture": "monai_segresnet_3d", "val_dice": dice}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history))

    ckpt = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    mean_dice, per = evaluate(model, test, n_cls, dev)
    report = {"split": "test", "scans": len(test), "spacing_mm": SPACING, "mean_dice": mean_dice,
              "per_structure_dice": {names[c]: round(d, 4) for c, d in sorted(per.items())},
              "best_val_dice": best, "epochs": len(history),
              "note": "ToothFairy2 random hold-out at 1.0 mm. Not an external or clinical validation."}
    (a.out / "report_cbct_test.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("scans", "mean_dice", "best_val_dice", "epochs")}), flush=True)


if __name__ == "__main__":
    main()
