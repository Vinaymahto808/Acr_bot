"""2x AI super-resolution of a portrait before it becomes the puppet (OpenCV dnn_superres + EDSR, CPU).

A small photo (the ACR face is 500x500) is enlarged ~1.7x to fill the 640x852 frame and again by the phone
screen, which read as blur (2026-11-13 feedback). EDSR adds real detail where plain resizing only smears.
Colour is super-resolved on the photo flattened onto --bg; the transparency (alpha) is enlarged with Lanczos.

    python tran_moment/upscale_photo.py --photo prepared/face_v2_source.jpg --bg 252,253,255
      -> prepared/face_v2_source_2x.png (RGBA) + prepared/face_v2_flat_2x.png (RGB, for prepare_portrait.py)

Model: tran_moment/motion/models/EDSR_x2.pb (github.com/Saafke/EDSR_Tensorflow, 38 MB; git-ignored).
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
MODEL = HERE / "motion" / "models" / "EDSR_x2.pb"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path, required=True)
    ap.add_argument("--bg", default="252,253,255", help="colour a transparent background is flattened onto")
    a = ap.parse_args()
    photo = a.photo if a.photo.is_absolute() else (HERE / a.photo if (HERE / a.photo).exists() else a.photo)
    src = ImageOps.exif_transpose(Image.open(photo)).convert("RGBA")
    bg = tuple(int(v) for v in a.bg.split(","))
    flat = Image.new("RGBA", src.size, bg + (255,)); flat.alpha_composite(src)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(MODEL)); sr.setModel("edsr", 2)
    t = time.time()
    up = sr.upsample(cv2.cvtColor(np.asarray(flat.convert("RGB")), cv2.COLOR_RGB2BGR))
    big = Image.fromarray(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
    alpha = src.getchannel("A").resize(big.size, Image.LANCZOS)
    stem = photo.stem.replace("_source", "")
    flat_out, rgba_out = photo.with_name(f"{stem}_flat_2x.png"), photo.with_name(f"{stem}_source_2x.png")
    big.save(flat_out)
    rgba = big.copy(); rgba.putalpha(alpha); rgba.save(rgba_out)
    print(f"EDSR x2: {src.size} -> {big.size} in {time.time() - t:.0f}s -> {flat_out.name}, {rgba_out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
