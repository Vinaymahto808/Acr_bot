"""Cut ~9 aligned, photo-accurate mouth-shape PNGs from prepared/face.jpg (Path B, CPU).

This is the realistic pass: instead of resizing the mouth crop into a blob, it
*wraps the real pixels* around the landmark mouth — the lower lip + chin drop
(so the photo's own teeth / inner-mouth band stretches open, exactly like the
live widget canvas), the lips spread or round, and the whole edit is feathered
back into the untouched skin.  No model, no GPU, numpy + PIL only.

A designer can still replace the files in mouths/ with hand-painted shapes;
keep the same filenames and canvas size (full portrait, transparent outside the
mouth region).

    uv run python tran_moment/path_b/generate_mouths.py
    # without uv:  python tran_moment/path_b/generate_mouths.py   (numpy + pillow)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PREP = HERE.parent / "prepared"
OUT = HERE / "mouths"

# Each shape: how far the jaw drops (open, -0.1..1.0) and how the lips move
# sideways (width, -1 = round O/U .. +1 = wide E/I).  "open" is the same 0..1
# the live widget feeds its canvas jaw, so the two paths agree frame-to-frame.
SHAPES = {
    "rest":   {"open": 0.00, "width": 0.00, "file": "X.png"},
    "closed": {"open": -1.00, "width": 0.00, "file": "closed.png"},  # lips pressed together over the teeth (MBP)
    "A":      {"open": 0.90, "width": 0.05, "file": "A.png"},        # a / आ / schwa
    "E":      {"open": 0.50, "width": 0.85, "file": "E.png"},        # e / ai
    "I":      {"open": 0.38, "width": 0.92, "file": "I.png"},        # i / ee
    "O":      {"open": 0.66, "width": -0.85, "file": "O.png"},       # o
    "U":      {"open": 0.46, "width": -0.95, "file": "U.png"},       # u / oo
    "FV":     {"open": 0.18, "width": 0.22, "file": "FV.png"},       # f / v (teeth on lip)
    "L":      {"open": 0.55, "width": 0.28, "file": "L.png"},        # l / t / d
}

# Rhubarb A..H,X → our photo shapes (A is MBP = lips together, X is rest)
RHUBARB = {
    "A": "closed.png",
    "B": "E.png",
    "C": "A.png",
    "D": "I.png",
    "E": "O.png",    # round open (o, au)
    "F": "U.png",
    "G": "FV.png",   # teeth on lower lip (f, v)
    "H": "L.png",
    "X": "X.png",
}


def _smooth(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _close(rgb: np.ndarray, pts: dict, c: float) -> np.ndarray:
    """Lips brought together over the teeth (c = 0 photo .. 1 closed).

    The portrait has parted lips, so without this the mouth never closed: M/B/P and the pauses between words
    still showed teeth (2026-11-13 feedback "lips / jaw movement properly"). The upper lip slides down and the
    lower lip up by half the gap each (plus a pixel, into the lip colour), the band between them (the teeth)
    disappears, and a faint shadow marks where the lips meet. Fades out towards the corners and to nose/chin.
    """
    from scipy.ndimage import map_coordinates

    H, W = rgb.shape[:2]
    lx, rx = float(pts["mouth_left"][0]), float(pts["mouth_right"][0])
    up, lo = float(pts["lip_upper_inner"][1]), float(pts["lip_lower_inner"][1])
    lip_top, lip_bottom = float(pts["lip_top"][1]), float(pts["lip_bottom"][1])
    nose, chin = float(pts["nose_bottom"][1]), float(pts["chin"][1])
    cx, mw, mid = (lx + rx) / 2.0, max(8.0, rx - lx), (up + lo) / 2.0
    D = c * ((lo - up) / 2.0 + 1.2)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.clip(np.abs(xx - cx) / (mw / 2.0) / 1.05, 0, 1)
    fall = np.where(np.abs(xx - cx) <= mw / 2.0 * 1.05, np.cos(d ** 3 * np.pi / 2.0), 0.0).astype(np.float32)
    wu = np.where(yy <= mid, np.where(yy >= lip_top, 1.0, _smooth((yy - nose) / max(4.0, lip_top - nose))), 0.0)
    wl = np.where(yy > mid, np.where(yy <= lip_bottom, 1.0, _smooth((chin - yy) / max(4.0, chin - lip_bottom))), 0.0)
    sy = (yy - D * fall * wu + D * fall * wl).astype(np.float32)
    out = np.empty_like(rgb, dtype=np.float32)
    for ch in range(3):
        out[..., ch] = map_coordinates(rgb[..., ch].astype(np.float32), np.array([sy, xx]), order=3, mode="nearest")
    seam = 1.0 - 0.20 * c * fall * np.exp(-((yy - mid) / 1.3) ** 2)
    return out * seam[..., None]


def _warp(rgb: np.ndarray, pts: dict, open_k: float, width_k: float) -> np.ndarray:
    """Return a copy of ``rgb`` with the mouth warped open/wide/round (open_k < 0: lips closing, see _close)."""
    from scipy.ndimage import map_coordinates

    if open_k < 0:
        closed = _close(rgb, pts, min(1.0, -open_k))
        return _warp(np.clip(closed, 0, 255).astype(np.uint8), pts, 0.0, width_k) if abs(width_k) > 0.01 else closed

    H, W = rgb.shape[:2]
    lx = float(pts["mouth_left"][0])
    rx = float(pts["mouth_right"][0])
    up = float(pts["lip_upper_inner"][1])
    lo = float(pts["lip_lower_inner"][1])
    lip_top = float(pts["lip_top"][1])
    lip_bottom = float(pts["lip_bottom"][1])
    nose = float(pts["nose_bottom"][1])
    chin = float(pts["chin"][1])
    cx = (lx + rx) / 2.0
    mw = max(8.0, rx - lx)
    lipY = (up + lo) / 2.0
    span = max(12.0, chin - lipY)

    teeth = lo - up >= 5                # parted lips in the photo: the teeth already show
    # jaw drop: ~16% of the mouth width at full opening (a natural gape, not a hole); 10% when the teeth already
    # show, so stretching them stays believable (2026-11-13 feedback)
    gap = open_k * mw * (0.13 if teeth else 0.16)   # 0.13 (was 0.10): the jaw visibly drops (2026-11-13)
    upper = open_k * mw * 0.05          # the upper lip lifts a little too

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # horizontal falloff: 1 at the mouth centre, 0 past the corners
    d = np.abs(xx - cx) / (mw / 2.0)
    fall = np.where(d <= 1.15, np.cos(np.clip(d / 1.15, 0, 1) * np.pi / 2.0), 0.0) ** 1.3
    fall = fall.astype(np.float32)

    # vertical weight below the lip line: full over the lower lip + chin, easing out on the neck
    ty = (yy - lipY) / span
    wl = np.where(ty < 0, 0.0, np.where(ty <= 0.85, 1.0, _smooth((1.15 - ty) / 0.30)))
    # vertical weight above the lip line: strongest at the lip, easing to the nose
    tu = (lipY - yy) / max(8.0, lipY - nose)
    wu = np.where(yy > lipY, 0.0, _smooth(tu) * 0.85)

    dy = (gap * fall * wl - upper * fall * wu).astype(np.float32)

    # sideways spread/round, damped away from the mouth
    sx = 1.0 + width_k * 0.12
    vert = np.exp(-((yy - lipY) / (0.95 * max(6.0, lip_bottom - lip_top))) ** 2)
    hw = (vert * fall).astype(np.float32)
    xs = (xx + ((xx - cx) / sx - (xx - cx)) * hw).astype(np.float32)

    sy = yy - dy
    if teeth:
        # Teeth showing: the band between the inner lips is SCALED to the opened height (up .. lo+g), so the lower
        # lip follows the jaw and the teeth grow smoothly. Repeating rows here gave streaks and a doubled seam
        # (the "black line / buffer" in the middle, 2026-11-13 feedback). Continuous with the jaw drop at lo+g.
        g = gap * fall
        band = (yy >= up) & (yy <= lo + g) & (fall > 0.02)
        sy = np.where(band, up + (yy - up) * (lo - up) / np.maximum(lo - up + g, 1e-3), sy)
    src = np.array([sy, xs], dtype=np.float32)
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        out[..., c] = map_coordinates(rgb[..., c].astype(np.float32), src, order=3, mode="nearest")   # bicubic: bilinear softened every speaking shape (2026-11-13)

    # a soft shadow inside the opening reads as depth (never a flat dark hole)
    if open_k > 0.05 and not teeth:
        cav_top = up - upper * fall
        cav_bot = lo + gap * fall
        inside = (yy > cav_top) & (yy < cav_bot)
        cavw = np.clip(1.0 - (np.abs(xx - cx) / (mw * 0.5)) ** 2, 0.0, 1.0)
        shade = 1.0 - 0.08 * cavw * inside   # was 0.20: on teeth that read as a dark band
        out *= shade[..., None]

    return out


def _clean_teeth_seam(rgb: np.ndarray, pts: dict) -> np.ndarray:
    """Fill the dark line between the upper and lower teeth with the teeth's own colour.

    A portrait with parted lips has that seam on the lip line, and _warp opens the mouth there: the seam
    was pushed down and the upper teeth repeated above it, which read as a black bar through the teeth
    while she spoke (2026-11-13 feedback). Only the speaking shapes are warped from this cleaned copy;
    the resting mouth keeps the photo as it is. A face with closed lips (no teeth showing) is unchanged.
    """
    import cv2

    lx, rx = float(pts["mouth_left"][0]), float(pts["mouth_right"][0])
    up, lo = float(pts["lip_upper_inner"][1]), float(pts["lip_lower_inner"][1])
    mw = max(8.0, rx - lx)
    if lo - up < 5:
        return rgb                                   # lips closed: no teeth band
    x0, x1 = int(lx - 0.04 * mw), int(rx + 0.04 * mw)   # through the corners: their dark dots too
    y0, y1 = int(up) - 1, int(np.ceil(lo)) + 2
    box = rgb[y0:y1, x0:x1].astype(np.float32)
    lum = box @ np.array([0.299, 0.587, 0.114], np.float32)
    if (lum > 140).mean() < 0.25:
        return rgb                                   # no bright teeth in the opening
    bright = lum > np.percentile(lum, 50)
    teeth = float(np.median(lum[bright]))   # mid level of the bright half (teeth + light lip): the grey threshold
    t_hi = float(np.percentile(lum, 88))    # the teeth themselves: the "clearly dark" threshold (lip-edge shadow)
    teeth_rgb = np.median(box[bright & (lum > 0.85 * teeth)], axis=0)
    redness = box[..., 0] - box[..., 1]
    yy = np.arange(y0, y1, dtype=np.float32)[:, None] + np.zeros_like(lum)
    mid = (up + lo) / 2.0
    inner = np.abs(yy - mid) <= (lo - up) * 0.75
    # dark OR greyish-not-lip: the gap under the teeth, its grey rim and the corner shadows. Lips are red
    # (R-G high) and stay. An inpaint averaged these with their neighbours and left a grey band and single
    # black dots (2026-11-13 feedback after the GFPGAN rebuild), so the pixels are filled directly instead.
    xx = np.arange(x0, x1, dtype=np.float32)[None, :] + np.zeros_like(lum)
    # the lower lip's inner edge under the middle teeth: a reddish-dark line the open shapes stretch into a tick
    lip_edge = (lum < 0.5 * t_hi) & (np.abs(xx - (lx + rx) / 2) < 0.3 * mw) & (yy >= lo - 4)
    seam = inner & ((lum < 0.55 * teeth) | ((lum < 0.85 * teeth) & (redness < 46)) | lip_edge)
    if not seam.any():
        return rgb
    seam = cv2.dilate(seam.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & inner & ~((lum > 0.85 * teeth))
    # each column: the masked run is interpolated from the clean pixel above it (teeth or upper lip) to the clean
    # pixel below (lower lip), so the gap becomes a soft teeth-to-lip gradient, then smoothed sideways a little
    fill = box.copy()
    hh = box.shape[0]
    for x in range(box.shape[1]):
        col = seam[:, x]
        y = 0
        while y < hh:
            if not col[y]:
                y += 1; continue
            s = y
            while y < hh and col[y]:
                y += 1
            top = box[s - 1, x] if s > 0 else (box[y, x] if y < hh else teeth_rgb)
            bot = box[y, x] if y < hh else top
            t = (np.arange(s, y, dtype=np.float32) - s + 1) / (y - s + 1)
            fill[s:y, x] = top * (1 - t[:, None]) + bot * t[:, None]
    fill = cv2.GaussianBlur(fill, (0, 0), sigmaX=1.6, sigmaY=0.6)
    a = cv2.GaussianBlur(seam.astype(np.float32), (0, 0), 0.8)[..., None]
    a = np.maximum(a, seam[..., None].astype(np.float32))
    out = rgb.copy()
    out[y0:y1, x0:x1] = np.clip(box * (1 - a) + fill * a, 0, 255).astype(np.uint8)
    return out


def sharpen(im: Image.Image) -> Image.Image:
    """The same light unsharp mask on every face layer (base, mouth shapes, smile, eyes, brows), so her face is
    crisp on a phone and the sharpness never changes when the mouth shape does (2026-11-13 blur feedback)."""
    from PIL import ImageFilter  # noqa: PLC0415
    rgb = im.convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.3, percent=65, threshold=2))
    if im.mode == "RGBA":
        rgb.putalpha(im.getchannel("A"))
    return rgb


def _alpha(rgb_wh: tuple[int, int], pts: dict) -> np.ndarray:
    """Feathered alpha that is opaque over the mouth/jaw edit and 0 outside it."""
    H, W = rgb_wh
    lx = float(pts["mouth_left"][0])
    rx = float(pts["mouth_right"][0])
    lip_top = float(pts["lip_top"][1])
    chin = float(pts["chin"][1])
    cx = (lx + rx) / 2.0
    mw = max(8.0, rx - lx)
    x0, x1 = cx - mw * 1.35, cx + mw * 1.35
    y0, y1 = lip_top - mw * 0.55, chin + mw * 0.30
    fx = mw * 0.35
    fy = mw * 0.25

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ax = _smooth((xx - x0) / fx) * _smooth((x1 - xx) / fx)
    ay = _smooth((yy - y0) / fy) * _smooth((y1 - yy) / fy)
    return np.clip(ax * ay * 255.0, 0, 255).astype(np.uint8)


# Continuous lips (2026-11-13 "train proper lip movement"): the same warp sampled on a dense (open x width) grid,
# packed into one JPEG atlas of mouth-box crops. The card blends the 4 nearest cells every frame, so the jaw rises
# and falls smoothly with the voice instead of jumping between 9 fixed shapes. All 9 shapes above are points of
# this grid (same warp), so the in-between frames are the same geometry, never a cross-fade of two mouths.
GRID_OPENS = [-1.0, -0.5, 0.0, 0.16, 0.34, 0.52, 0.72, 0.92]   # < 0: lips closing over the teeth
GRID_WIDTHS = [-0.95, -0.45, 0.0, 0.45, 0.92]


def write_grid(clean: np.ndarray, pts: dict, box: list[int]) -> dict:
    x0, y0, x1, y1 = box
    H, W = clean.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    bw, bh = x1 - x0, y1 - y0
    atlas = Image.new("RGB", (bw * len(GRID_OPENS), bh * len(GRID_WIDTHS)))
    for r, wk in enumerate(GRID_WIDTHS):
        for c, ok in enumerate(GRID_OPENS):
            warped = np.clip(_warp(clean, pts, ok, wk), 0, 255).astype(np.uint8)
            cell = sharpen(Image.fromarray(warped[y0:y1, x0:x1]))
            atlas.paste(cell, (c * bw, r * bh))
    atlas.save(OUT / "grid.jpg", quality=93, subsampling=0)
    return {"file": "grid.jpg", "box": [x0, y0, bw, bh], "opens": GRID_OPENS, "widths": GRID_WIDTHS}


def main() -> int:
    jpg = PREP / "face.jpg"
    meta = PREP / "face.json"
    if not jpg.exists() or not meta.exists():
        print("run check_and_prepare.py first", file=sys.stderr)
        return 1
    data = json.loads(meta.read_text(encoding="utf-8"))
    pts = data["points"]
    img = Image.open(jpg).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    H, W = rgb.shape[:2]
    alpha = _alpha((H, W), pts)

    OUT.mkdir(parents=True, exist_ok=True)
    sharpen(Image.fromarray(_clean_teeth_seam(rgb, pts))).save(OUT / "base.png")

    # every shape, the resting mouth too: the photo's dark gap between the lips read as a black line while she
    # talked (2026-11-13 feedback). base.png (the face under the shapes) is cleaned the same way below.
    clean = _clean_teeth_seam(rgb, pts)
    for name, spec in SHAPES.items():
        warped = _warp(clean, pts, spec["open"], spec["width"])
        rgba = np.dstack([np.clip(warped, 0, 255).astype(np.uint8), alpha])
        dest = OUT / spec["file"]
        sharpen(Image.fromarray(rgba, "RGBA")).save(dest)
        print("wrote", dest.name)

    index = {
        "canvas": {"width": W, "height": H},
        "mouth_alpha_box": [
            int((pts["mouth_left"][0] + pts["mouth_right"][0]) / 2 - (pts["mouth_right"][0] - pts["mouth_left"][0]) * 1.35),
            int(pts["lip_top"][1] - (pts["mouth_right"][0] - pts["mouth_left"][0]) * 0.55),
            int((pts["mouth_left"][0] + pts["mouth_right"][0]) / 2 + (pts["mouth_right"][0] - pts["mouth_left"][0]) * 1.35),
            int(pts["chin"][1] + (pts["mouth_right"][0] - pts["mouth_left"][0]) * 0.30),
        ],
        "shapes": {k: v["file"] for k, v in SHAPES.items()},
        "rhubarb": RHUBARB,
        "open": {k: v["open"] for k, v in SHAPES.items()},
        "width": {k: v["width"] for k, v in SHAPES.items()},
        "note": "Photo-accurate warp of prepared/face.jpg. Replace PNGs with designer shapes; keep filenames and canvas size.",
    }
    index["grid"] = write_grid(clean, pts, index["mouth_alpha_box"])
    print("wrote grid.jpg", index["grid"]["box"], f"{len(GRID_OPENS)}x{len(GRID_WIDTHS)} cells")
    (HERE / "mouths.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("wrote", HERE / "mouths.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
