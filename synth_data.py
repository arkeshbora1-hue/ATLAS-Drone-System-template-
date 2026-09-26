"""Synthetic drop-zone dataset generator.

Composites a landing-pad style marker (ring + "H", the pattern used on the
ATLAS drop-zone mat) onto aerial-looking backgrounds with random scale,
rotation, perspective squash, blur, brightness, shadows and distractors.
Used to bootstrap training and to smoke-test the pipeline end-to-end; the
final model should be fine-tuned on real frames captured from the vehicle's
own camera (see training/README.md).

    python training/synth_data.py --out data/synth --n 4000
    python training/synth_data.py --out data/synth --n 4000 --backgrounds data/aerial_bg/
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_labels  # noqa: E402

W, H = 320, 240


def marker(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size
    d.ellipse([0, 0, s - 1, s - 1], fill=(250, 250, 250, 255))
    d.ellipse([s * 0.08, s * 0.08, s * 0.92, s * 0.92], fill=(220, 30, 40, 255))
    d.ellipse([s * 0.18, s * 0.18, s * 0.82, s * 0.82], fill=(250, 250, 250, 255))
    bw, x0, x1, y0, y1 = s * 0.09, s * 0.33, s * 0.67, s * 0.30, s * 0.70
    d.rectangle([x0 - bw / 2, y0, x0 + bw / 2, y1], fill=(20, 20, 20, 255))
    d.rectangle([x1 - bw / 2, y0, x1 + bw / 2, y1], fill=(20, 20, 20, 255))
    d.rectangle([x0, s * 0.5 - bw / 2, x1, s * 0.5 + bw / 2], fill=(20, 20, 20, 255))
    return img


def procedural_background(rng: random.Random) -> Image.Image:
    """Grass/soil/concrete-like texture from layered noise + random patches."""
    base = np.array(rng.choice([(70, 110, 50), (95, 120, 60), (120, 110, 90), (140, 140, 135), (85, 95, 70)]),
                    dtype=np.float32)
    low = np.array(Image.fromarray((np.random.rand(12, 16) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC),
                   dtype=np.float32)[..., None] / 255.0
    high = np.random.randn(H, W, 1).astype(np.float32)
    img = base * (0.75 + 0.5 * low) + high * rng.uniform(4, 14)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(rng.randint(0, 6)):  # paths, roofs, patches
        c = tuple(int(v) for v in np.clip(base + rng.uniform(-50, 60), 0, 255))
        x, y = rng.randint(0, W), rng.randint(0, H)
        if rng.random() < 0.5:
            d.rectangle([x, y, x + rng.randint(10, 120), y + rng.randint(5, 60)], fill=c)
        else:
            d.line([x, y, x + rng.randint(-200, 200), y + rng.randint(-200, 200)], fill=c, width=rng.randint(3, 14))
    return im


def distractor(d: ImageDraw.ImageDraw, rng: random.Random) -> None:
    """Hard negatives: white circles, red shapes, letters that are not the marker."""
    x, y, r = rng.randint(0, W), rng.randint(0, H), rng.randint(6, 30)
    kind = rng.random()
    if kind < 0.4:
        d.ellipse([x - r, y - r, x + r, y + r], outline=(240, 240, 240), width=rng.randint(2, 5))
    elif kind < 0.7:
        d.rectangle([x - r, y - r, x + r, y + r], fill=(200, 40, 40))
    else:
        d.text((x, y), rng.choice("AHXT"), fill=(240, 240, 240))


def make_sample(rng: random.Random, bgs: list[Path], positive: bool):
    if bgs:
        bg = Image.open(rng.choice(bgs)).convert("RGB")
        s = rng.uniform(0.5, 1.0)
        cw, ch = int(bg.width * s), int(bg.height * s)
        x, y = rng.randint(0, bg.width - cw), rng.randint(0, bg.height - ch)
        bg = bg.crop((x, y, x + cw, y + ch)).resize((W, H))
    else:
        bg = procedural_background(rng)
    d = ImageDraw.Draw(bg)
    for _ in range(rng.randint(0, 3)):
        distractor(d, rng)
    box = [0.0, 0.0, 0.0, 0.0]
    if positive:
        size = rng.randint(18, 150)                 # ~25 m down to ~3 m AGL
        m = marker(size).rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=True)
        squash = rng.uniform(0.8, 1.0)             # off-nadir tilt
        m = m.resize((m.width, max(4, int(m.height * squash))))
        # allow partial visibility at image edges
        cx = rng.uniform(0.05, 0.95) * W
        cy = rng.uniform(0.05, 0.95) * H
        bg.paste(m, (int(cx - m.width / 2), int(cy - m.height / 2)), m)
        bw, bh = size * 0.92, size * 0.92 * squash
        x0, x1 = max(0, cx - bw / 2), min(W, cx + bw / 2)
        y0, y1 = max(0, cy - bh / 2), min(H, cy + bh / 2)
        box = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
    if rng.random() < 0.4:                           # shadow band
        sh = Image.new("L", (W, H), 0)
        ImageDraw.Draw(sh).polygon([(rng.randint(0, W), 0), (rng.randint(0, W), H), (W, H), (W, 0)], fill=90)
        bg = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), bg, sh.filter(ImageFilter.GaussianBlur(8)))
    bg = ImageEnhance.Brightness(bg).enhance(rng.uniform(0.6, 1.4))
    bg = ImageEnhance.Contrast(bg).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.3:                           # motion / focus blur
        bg = bg.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.8)))
    return bg, positive, box


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--pos-frac", type=float, default=0.6)
    ap.add_argument("--backgrounds", default=None, help="folder of aerial background photos (optional)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    bgs = sorted(Path(a.backgrounds).glob("*.jpg")) if a.backgrounds else []
    rows = []
    for i in range(a.n):
        img, pos, box = make_sample(rng, bgs, rng.random() < a.pos_frac)
        name = f"synth_{i:05d}.jpg"
        img.save(out / "images" / name, quality=90)
        rows.append({"filename": name, "present": pos, "box": box})
    write_labels(out, rows)
    print(f"wrote {a.n} images to {out} ({sum(r['present'] for r in rows)} positive)")


if __name__ == "__main__":
    main()
