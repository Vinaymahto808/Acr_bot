"""Level a transparent portrait (rotate about the eye midpoint until the eyes are horizontal) and flatten it onto
the card's soft studio background. CPU only (MediaPipe FaceLandmarker for the eyes).

    uv run --no-project --python 3.12 --with mediapipe --with numpy --with pillow --with opencv-python \
        python tran_moment/level_portrait.py --photo tran_moment/path_a/face1.png
    -> prepared/face_v4_source.png  levelled RGBA (same size as the photo)
       prepared/face_v4_flat.png    the same on the soft background (what prepare_portrait.py is run on)

Then: prepare_portrait.py --photo prepared/face_v4_flat.png, path_b/generate_mouths.py, generate_reactions.py,
generate_grid_v2.py, fix_lip_seam.py, finish_portrait.py, make_bubble.py, avatar_v2/build_reference_pack.py.

Why a soft background and not black: the card's stage is black (#000) and her hair and the black ACR polo vanished into
it; an opaque base also means a mouth patch can never carry a different background colour over the jaw line (the old
portrait was flattened on white and showed a white wedge beside the chin while she spoke). The rotation is rigid (the
face is not warped) and done on premultiplied colour, so the hair edges keep no dark or light fringe.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_portrait as pp  # noqa: E402

BG_TOP, BG_BOTTOM = np.array([238, 240, 243], np.float32), np.array([207, 212, 219], np.float32)


def soft_background(w: int, h: int) -> np.ndarray:
    """light neutral studio sweep: a vertical gradient with a gentle glow behind the head"""
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    bg = BG_TOP * (1 - y) + BG_BOTTOM * y
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    glow = np.exp(-(((xx - w / 2) / (0.45 * w)) ** 2 + ((yy - 0.3 * h) / (0.4 * h)) ** 2))[..., None]
    return np.clip(bg + 8.0 * glow, 0, 255)


def flatten(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    bg = soft_background(rgba.shape[1], rgba.shape[0])
    return np.clip(rgba[..., :3].astype(np.float32) * a + bg * (1 - a), 0, 255).astype(np.uint8)


def roll_deg(p: dict) -> float:
    (x0, y0), (x1, y1) = p["eye_l_outer"], p["eye_r_outer"]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def rotate_rgba(rgba: np.ndarray, centre: tuple[float, float], deg: float) -> np.ndarray:
    M = cv2.getRotationMatrix2D(centre, deg, 1.0)       # positive = counter-clockwise on screen: lifts the right side
    f = rgba.astype(np.float32) / 255.0
    pre = np.dstack([f[..., :3] * f[..., 3:4], f[..., 3:4]])
    h, w = rgba.shape[:2]
    out = cv2.warpAffine(pre, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    out = np.clip(out, 0, 1)
    a = out[..., 3:4]
    rgb = np.where(a > 1e-4, out[..., :3] / np.maximum(a, 1e-4), 0)
    return np.clip(np.dstack([rgb, a]) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "prepared")
    a = ap.parse_args()
    src = np.asarray(ImageOps.exif_transpose(Image.open(a.photo)).convert("RGBA"))
    p = pp.landmarks(Image.fromarray(flatten(src)))
    before = roll_deg(p)
    cx = (p["eye_l_outer"][0] + p["eye_r_outer"][0]) / 2
    cy = (p["eye_l_outer"][1] + p["eye_r_outer"][1]) / 2
    lev = rotate_rgba(src, (cx, cy), before)
    for _ in range(2):   # measure again; a second small step absorbs landmark noise
        r = roll_deg(pp.landmarks(Image.fromarray(flatten(lev))))
        if abs(r) < 0.15:
            break
        before_total = before + r
        lev = rotate_rgba(src, (cx, cy), before_total)
        before = before_total
    q = pp.landmarks(Image.fromarray(flatten(lev)))
    mouth = math.degrees(math.atan2(q["mouth_right"][1] - q["mouth_left"][1], q["mouth_right"][0] - q["mouth_left"][0]))
    a.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(lev, "RGBA").save(a.out / "face_v4_source.png")
    Image.fromarray(flatten(lev)).save(a.out / "face_v4_flat.png")
    print(f"rotated {before:+.2f} deg about the eye midpoint ({cx:.0f},{cy:.0f}); eyes now {roll_deg(q):+.2f} deg, "
          f"mouth corners {mouth:+.2f} deg -> {a.out / 'face_v4_source.png'}, face_v4_flat.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
