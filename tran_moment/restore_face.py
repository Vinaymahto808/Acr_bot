"""Face restoration (GFPGAN 1.4, ONNX on CPU) before a portrait becomes the puppet.

The ACR face comes from a 500x500 photo: even after EDSR x2 the lips and teeth stayed soft on the card
(2026-11-13 feedback "lip blur"). Super-resolution only enlarges what is there; GFPGAN redraws the face's
fine detail (lip edges, teeth, eyelashes) from a face prior. The face is aligned to the FFHQ 512 template with
5 MediaPipe points, restored, and pasted back through a feathered mask, so hair, ears and clothes stay the
photo's own. --weight blends restored and original (1.0 = fully restored).

    python tran_moment/restore_face.py --photo prepared/face_v2_source_2x.png --bg 252,253,255 --out face_v3
      -> prepared/face_v3_flat.png (RGB, for prepare_portrait.py) + prepared/face_v3_source.png (RGBA)

Model: tran_moment/motion/models/gfpgan_1.4.onnx (github.com/facefusion/facefusion-assets, 340 MB; git-ignored).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_portrait as pp  # noqa: E402

MODEL = HERE / "motion" / "models" / "gfpgan_1.4.onnx"
# FFHQ 512 alignment template: eye centres, nose tip, mouth corners
TEMPLATE = np.array([[192.98, 239.95], [318.90, 240.19], [256.63, 314.02], [201.26, 371.41], [313.09, 371.15]], np.float32)


def five(p: dict) -> np.ndarray:
    mid = lambda a, b: ((p[a][0] + p[b][0]) / 2, (p[a][1] + p[b][1]) / 2)  # noqa: E731
    return np.array([mid("eye_l_outer", "eye_l_inner"), mid("eye_r_inner", "eye_r_outer"), p["nose_tip"],
                     p["mouth_left"], p["mouth_right"]], np.float32)


def restore(rgb: np.ndarray, p: dict, weight: float) -> np.ndarray:
    import onnxruntime as ort
    M, _ = cv2.estimateAffinePartial2D(five(p), TEMPLATE, method=cv2.LMEDS)
    crop = cv2.warpAffine(rgb, M, (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    x = ((crop.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    out = np.clip((y.transpose(1, 2, 0) * 0.5 + 0.5) * 255.0, 0, 255).astype(np.float32)
    out = weight * out + (1 - weight) * crop.astype(np.float32)
    # feathered mask in crop space: the face, fading well before the crop's edges
    mask = np.zeros((512, 512), np.float32)
    cv2.ellipse(mask, (256, 272), (185, 205), 0, 0, 360, 1.0, -1)
    mask[:40], mask[-40:], mask[:, :40], mask[:, -40:] = 0, 0, 0, 0   # never reach the crop's edge (a seam line)
    mask = cv2.GaussianBlur(mask, (0, 0), 14)
    Mi = cv2.invertAffineTransform(M)
    H, W = rgb.shape[:2]
    back = cv2.warpAffine(out, Mi, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    m = cv2.warpAffine(mask, Mi, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    return np.clip(back * m + rgb.astype(np.float32) * (1 - m), 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path, required=True, help="RGBA (or RGB) portrait")
    ap.add_argument("--bg", default="252,253,255", help="colour a transparent background is flattened onto")
    ap.add_argument("--out", default="face_v3", help="output stem in prepared/")
    ap.add_argument("--weight", type=float, default=0.85)
    a = ap.parse_args()
    photo = a.photo if a.photo.is_absolute() else (HERE / a.photo if (HERE / a.photo).exists() else a.photo)
    src = ImageOps.exif_transpose(Image.open(photo)).convert("RGBA")
    bg = tuple(int(v) for v in a.bg.split(","))
    flat = Image.new("RGBA", src.size, bg + (255,)); flat.alpha_composite(src)
    rgb = np.asarray(flat.convert("RGB"))
    t = time.time()
    fixed = Image.fromarray(restore(rgb, pp.landmarks(flat.convert("RGB")), a.weight))
    flat_out, rgba_out = HERE / "prepared" / f"{a.out}_flat.png", HERE / "prepared" / f"{a.out}_source.png"
    fixed.save(flat_out)
    # edge pixels of the flattened photo carry the --bg colour mixed in; un-mix it (C = (F - (1-a)*bg) / a), or
    # the hair outline shows a light fringe on the card's black background (2026-11-13)
    a8 = np.asarray(src.getchannel("A"), np.float32)[..., None] / 255.0
    f = np.asarray(fixed, np.float32)
    true = np.where(a8 > 0.04, (f - (1 - a8) * np.array(bg, np.float32)) / np.maximum(a8, 0.04), f)
    rgba = Image.fromarray(np.clip(true, 0, 255).astype(np.uint8)); rgba.putalpha(src.getchannel("A")); rgba.save(rgba_out)
    print(f"GFPGAN restore ({a.weight:.2f}) in {time.time() - t:.0f}s -> {flat_out.name}, {rgba_out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
