"""Draw detections on the film: a coloured outline that follows each lesion's
shape, filled translucent, with the finding written next to it.

  python render.py film.png --model best.pt --out film_marked.png

The detector only returns boxes. Inside each box the outline is traced from
the film itself: caries and periapical lesions are darker than the tissue
around them, impacted teeth brighter, so the box is split by Otsu's
threshold and the connected dark (or bright) region nearest the centre is
kept. That makes it an *approximate* outline drawn from image intensity, not
a segmentation the model learned, and every picture says so in its footer.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

COLOURS = {  # class -> (RGB, words on the film)
    "caries": ((255, 196, 0), "Caries"),
    "deep_caries": ((255, 80, 0), "Deep caries"),
    "periapical_lesion": ((230, 0, 120), "Periapical lesion"),
    "impacted": ((0, 170, 255), "Impacted tooth"),
}
DARKER = {"caries", "deep_caries", "periapical_lesion"}
FOOTER = "AI detections for clinician review. Outlines approximate, traced from image intensity. Not a diagnosis."


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=64)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / np.maximum(w1, 1)
    return float(centres[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def lesion_mask(gray: np.ndarray, box: list[float], darker: bool) -> np.ndarray:
    """Boolean mask (same size as the box crop) roughly following the lesion."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    crop = ndimage.gaussian_filter(gray[y1:y2, x1:x2].astype(np.float32), 1.5)
    h, w = crop.shape
    ellipse = _ellipse(h, w)
    if crop.size < 16 or np.ptp(crop) < 8:
        return ellipse
    t = _otsu(crop)
    mask = crop < t if darker else crop > t
    mask = ndimage.binary_opening(mask, iterations=1)
    labels, n = ndimage.label(mask)
    if n == 0:
        return ellipse
    yy, xx = np.mgrid[0:h, 0:w]
    dist = (yy - h / 2) ** 2 + (xx - w / 2) ** 2
    # Region nearest the box centre, weighted toward larger regions.
    best = min(range(1, n + 1), key=lambda k: dist[labels == k].min() - 4 * (labels == k).sum())
    region = ndimage.binary_fill_holes(labels == best)
    # A region that fills almost nothing, or the whole box, is not a shape; use the box's ellipse.
    share = region.mean()
    return region if 0.05 < share < 0.95 else ellipse


def _ellipse(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - h / 2) / max(h / 2, 1)) ** 2 + ((xx - w / 2) / max(w / 2, 1)) ** 2 <= 1


def draw(img: Image.Image, preds: list[dict], min_score: float = 0.5) -> Image.Image:
    base = img.convert("RGB")
    gray = np.asarray(base.convert("L"))
    W, H = base.size
    fill = Image.new("RGBA", base.size, (0, 0, 0, 0))
    edges = Image.new("RGBA", base.size, (0, 0, 0, 0))
    labels_to_draw = []
    for p in sorted(preds, key=lambda d: d["score"]):
        if p["score"] < min_score or p["class"] not in COLOURS:
            continue
        x1, y1, x2, y2 = (max(0, min(v, lim)) for v, lim in zip(p["box"], (W, H, W, H)))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        rgb, words = COLOURS[p["class"]]
        mask = lesion_mask(gray, [x1, y1, x2, y2], p["class"] in DARKER)
        ox, oy = int(round(x1)), int(round(y1))
        m = Image.fromarray((mask * 255).astype(np.uint8))
        fill.paste(Image.new("RGBA", m.size, rgb + (90,)), (ox, oy), m)
        border = mask & ~ndimage.binary_erosion(mask, iterations=max(2, round(min(W, H) / 500)))
        b = Image.fromarray((border * 255).astype(np.uint8))
        edges.paste(Image.new("RGBA", b.size, rgb + (255,)), (ox, oy), b)
        labels_to_draw.append((ox, oy, rgb, p.get("text") or f"{words} {round(p['score'] * 100)}%"))

    out = Image.alpha_composite(Image.alpha_composite(base.convert("RGBA"), fill), edges)
    d = ImageDraw.Draw(out)
    size = max(14, round(min(W, H) / 40))
    font = _font(size)
    for ox, oy, rgb, text in labels_to_draw:
        tx, ty = ox, max(0, oy - size - 6)
        l, t, r, bt = d.textbbox((tx, ty), text, font=font)
        d.rectangle((l - 3, t - 2, r + 3, bt + 2), fill=(0, 0, 0, 190))
        d.text((tx, ty), text, fill=rgb + (255,), font=font)

    small = _font(max(12, round(size * 0.7)))
    l, t, r, bt = d.textbbox((0, 0), FOOTER, font=small)
    d.rectangle((0, H - (bt - t) - 10, W, H), fill=(0, 0, 0, 200))
    d.text((6, H - (bt - t) - 6), FOOTER, fill=(255, 255, 255, 255), font=small)
    x = W - 6
    for cls in reversed(list(COLOURS)):
        rgb, words = COLOURS[cls]
        l, t, r, bt = d.textbbox((0, 0), words, font=small)
        x -= (r - l) + 26
        d.rectangle((x, 6, x + 14, 20), fill=rgb + (255,))
        d.text((x + 18, 4), words, fill=(255, 255, 255, 255), font=small, stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return out.convert("RGB")


def main():
    import torch

    from model import load_detector, open_film, predict

    p = argparse.ArgumentParser()
    p.add_argument("film")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-score", type=float, default=0.5)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, _ = load_detector(args.model, device)
    img = open_film(args.film)
    draw(img, predict(detector, img, device), args.min_score).save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()


TOOTH_COLOUR = (120, 230, 160)
ANATOMY_FOOTER = "AI-detected anatomy for clinician review: tooth numbers (FDI) may be wrong. Not a diagnosis."


def draw_teeth(img: Image.Image, teeth: list[dict], footer: bool = True) -> Image.Image:
    """Number each tooth (FDI) with a thin outline of the box the model returned."""
    out = img.convert("RGBA")
    W, H = out.size
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    width = max(1, round(min(W, H) / 600))
    font = _font(max(12, round(min(W, H) / 55)))
    for t in teeth:
        x1, y1, x2, y2 = t["box"]
        d.rectangle((x1, y1, x2, y2), outline=TOOTH_COLOUR + (200,), width=width)
        label = str(t["tooth"])
        l, tp, r, b = d.textbbox((0, 0), label, font=font)
        cx = (x1 + x2) / 2 - (r - l) / 2
        # Upper teeth labelled above the crown side, lower teeth below.
        ty = y1 + 4 if t["tooth"][0] in "12" else y2 - (b - tp) - 6
        d.rectangle((cx - 3, ty - 1, cx + (r - l) + 3, ty + (b - tp) + 4), fill=(0, 0, 0, 170))
        d.text((cx, ty), label, fill=TOOTH_COLOUR + (255,), font=font)
    out = Image.alpha_composite(out, layer)
    if footer:
        d = ImageDraw.Draw(out)
        small = _font(max(12, round(min(W, H) / 57)))
        l, t, r, b = d.textbbox((0, 0), ANATOMY_FOOTER, font=small)
        d.rectangle((0, H - (b - t) - 10, W, H), fill=(0, 0, 0, 200))
        d.text((6, H - (b - t) - 6), ANATOMY_FOOTER, fill=(255, 255, 255, 255), font=small)
    return out.convert("RGB")
