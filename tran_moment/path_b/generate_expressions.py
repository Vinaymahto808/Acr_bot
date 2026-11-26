"""Expression morph sequences for the card's puppet, warped from base.png itself (CPU: numpy + pillow + scipy).

    uv run --no-project --python 3.13 --with numpy --with pillow --with scipy python tran_moment/path_b/generate_expressions.py

Writes path_b/expressions/<name>/fNN.webp (lossless, cropped to the layer box) + path_b/expressions.json, which the
card loads as PUP.expr and plays frame by frame (motion, never an alpha cross-fade of two faces):

  (round 2, 2026-11-25: all three ~45 % stronger, they read as too subtle at card size)
  smile_soft     gentle closed-mouth smile: corners up/out ~5 px, a touch of cheek     (listening after a warm line,
                                                                                         empathetic rest at ~40 %)
  smile_warm     a fuller closed smile with cheek lift: corners ~6 px, cheeks ~3 px     (greeting, booking confirmed)
  brows_concern  inner brow ends up and a little in (worried / sympathetic, NOT a frown) (accident, complaint, angry)
  brows_up       whole brows up ~3 px                                                  (questions, emphasis)

Rules (owner, 2026-11-25): every pixel is the photo's own (a geometric warp of base.png, bicubic), nothing painted.
Each displacement field is exactly 0 within FEATHER px of its box border, so the feathered edge only ever blends
identical pixels (no seam). The brow fields are 0 from y = BROW_FLOOR down, where the blink layers
(reactions/eyes_*.png, rows >= 197) start: a blink during a concerned face can never bring an unmoved brow back.
The smile fields are 0 above the eye layers (y < 252); at rest the lip grid is off, so nothing else overlaps.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
PREP = HERE.parent / "prepared"
OUT = HERE / "expressions"
FEATHER = 8
BROW_FLOOR = 198.0
FOLD = 1.0           # round 4: nasolabial deepen, px per px of cheek lift (smile_warm: ~4.6 px toward the fold)     # first row the blink layers change (measured: eyes_half/low/closed differ from base from y 197)
EYE_LAYER_BOTTOM = 252.0


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def window(shape, box, f=FEATHER):
    """1 inside the box, 0 on its outer `f + 2` px: the field (and so the frame) equals base.png at the border."""
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    m = f + 2
    return (smooth((xx - x0 - m) / f) * smooth((x1 - 1 - m - xx) / f)
            * smooth((yy - y0 - m) / f) * smooth((y1 - 1 - m - yy) / f)), yy, xx


def alpha_mask(box, f=FEATHER):
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    return (smooth((xx - x0) / f) * smooth((x1 - 1 - xx) / f) * smooth((yy - y0) / f) * smooth((y1 - 1 - yy) / f))


def warp_crop(rgba: np.ndarray, box, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """out(y, x) = src(y - dy, x - dx) inside the box (positive dy moves content down); bicubic on the photo."""
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    coords = np.array([yy - dy, xx - dx])
    out = np.stack([map_coordinates(rgba[..., c], coords, order=3, mode="nearest") for c in range(4)], -1)
    exact = (np.abs(dx) < 1e-4) & (np.abs(dy) < 1e-4)                 # untouched pixels stay bit-identical
    out[exact] = rgba[y0:y1, x0:x1][exact]
    return out


def smile_field(pts, box, up, out_, cheek):
    w, yy, xx = window(None, box)
    lx, ly = pts["mouth_left"]; rx, ry = pts["mouth_right"]
    mw = rx - lx
    dx = np.zeros_like(yy); dy = np.zeros_like(yy)
    for cx, cy, s in ((lx, ly, -1.0), (rx, ry, 1.0)):
        r = np.sqrt((xx - cx) ** 2 + ((yy - cy) * 1.15) ** 2) / (mw * 0.46)
        k = np.clip(1 - r, 0, 1) ** 1.5
        dy -= up * k                       # corner up
        dx += s * out_ * k                 # and out
        # the lips between the corners follow a little (the whole line curves, not just two points)
        lip = np.exp(-((yy - cy) / 9.0) ** 2) * np.clip(1 - np.abs(xx - cx) / (mw * 0.55), 0, 1) ** 2
        dy -= 0.25 * up * lip
    for cx, cy in ((lx - mw * 0.24, ly - mw * 0.42), (rx + mw * 0.24, ry - mw * 0.42)):
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (mw * 0.52)
        dy -= cheek * np.clip(1 - r, 0, 1) ** 1.6                       # cheek apples rise
    # round 4 (owner: "the portion left / right of the nose does not react"): the cheek apples bunch a little OUT as they
    # rise, and the nasolabial fold (nose wing -> mouth corner) deepens: the cheek skin right beside the fold is pushed
    # toward it (0 on the fold line itself, 0 on the nose, 0 a little out in the cheek), so the fold's own shading gets
    # narrower and darker - a warp of her pixels, nothing painted. The nose stays fixed.
    cxm = 0.5 * (lx + rx); nb = pts["nose_bottom"][1]
    for (cx, cy), s in (((lx - mw * 0.24, ly - mw * 0.42), -1.0), ((rx + mw * 0.24, ry - mw * 0.42), 1.0)):
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (mw * 0.52)
        dx += s * 0.45 * cheek * np.clip(1 - r, 0, 1) ** 1.6
    for (ax_, ay_), (mx_, my_), s in (((cxm - 27.0, nb - 3.0), (lx - 2.0, ly), -1.0), ((cxm + 27.0, nb - 3.0), (rx + 2.0, ry), 1.0)):
        vx, vy = mx_ - ax_, my_ - ay_; L2 = vx * vx + vy * vy
        tt = np.clip(((xx - ax_) * vx + (yy - ay_) * vy) / L2, 0, 1)
        px_, py_ = ax_ + tt * vx, ay_ + tt * vy
        dist = (xx - px_) * s                                              # > 0 on the cheek side of the fold
        bump = np.clip(dist / 7.0, 0, 1) * np.clip(1 - (dist - 7.0) / 16.0, 0, 1)
        along = smooth((yy - (nb - 2.0)) / 8.0) * smooth((my_ + 18.0 - yy) / 12.0)
        dx -= s * FOLD * cheek * bump * along                                # toward the fold
        dy -= 0.5 * FOLD * cheek * bump * along                              # and up with the cheek
    keep_nose = 1 - (1 - smooth((np.abs(xx - cxm) - 28.0) / 6.0)) * (1 - smooth((yy - (nb + 1.0)) / 6.0))   # the nose itself never moves
    dx *= keep_nose; dy *= keep_nose
    dy *= smooth((yy - EYE_LAYER_BOTTOM) / 14.0)                           # nothing under the eye layers moves
    dx *= smooth((yy - EYE_LAYER_BOTTOM) / 14.0)
    return dx * w, dy * w


def brow_field(pts, box, mode, amt_up, amt_in):
    w, yy, xx = window(None, box)
    dx = np.zeros_like(yy); dy = np.zeros_like(yy)
    by = 0.5 * (pts["brow_l"][1] + pts["brow_r"][1]) + 8.0              # landmark is the top edge; brow body ~ +8 px
    # rows: full move on the brow body, eased to 0 at BROW_FLOOR (blink layers below) and up into the forehead
    vert = smooth((BROW_FLOOR - yy) / 9.0) * smooth((yy - (by - 46.0)) / 26.0)
    inner_l, inner_r = pts["eye_l_inner"][0] + 10.0, pts["eye_r_inner"][0] - 10.0   # inner brow heads ~x 300 / 339
    outer_l, outer_r = pts["eye_l_outer"][0] - 14.0, pts["eye_r_outer"][0] + 14.0
    # the bindi between the brows stays exactly where it is (a warp through it smeared it into a streak)
    mid = 0.5 * (inner_l + inner_r)
    keep = smooth((np.abs(xx - mid) - 7.0) / 9.0)
    if mode == "concern":
        for ix, s in ((inner_l, 1.0), (inner_r, -1.0)):
            near = np.clip(1 - np.abs(xx - ix) / 40.0, 0, 1) ** 1.3        # inner third of the brow
            dy -= amt_up * near * vert * keep                             # inner ends up (worry / sympathy)
            dx += s * amt_in * near * vert * keep                         # and drawn a little toward each other
    else:
        span = smooth((xx - outer_l) / 18.0) * smooth((outer_r - xx) / 18.0)
        dy -= amt_up * span * vert * keep
    return dx * w, dy * w


def main() -> int:
    pts = json.loads((PREP / "face.json").read_text(encoding="utf-8"))["points"]
    base = np.asarray(Image.open(HERE / "mouths" / "base.png").convert("RGBA"), np.float32)
    lx, rx = pts["mouth_left"][0], pts["mouth_right"][0]
    mw = rx - lx
    smile_box = [int(lx - 0.9 * mw), int(EYE_LAYER_BOTTOM - 4), int(rx + 0.9 * mw), int(pts["mouth_left"][1] + 0.75 * mw)]
    brow_box = [int(pts["eye_l_outer"][0] - 26), int(min(pts["brow_l"][1], pts["brow_r"][1]) - 50),
                int(pts["eye_r_outer"][0] + 26), int(BROW_FLOOR + FEATHER)]
    specs = {
        # name: (region, box, frames, field(amount 0..1) -> (dx, dy))
        "smile_soft": ("face", smile_box, 8, lambda t: smile_field(pts, smile_box, 5.8 * t, 3.7 * t, 2.4 * t)),
        "smile_warm": ("face", smile_box, 10, lambda t: smile_field(pts, smile_box, 9.2 * t, 5.6 * t, 4.6 * t)),
        "brows_concern": ("brows", brow_box, 7, lambda t: brow_field(pts, brow_box, "concern", 7.2 * t, 1.4 * t)),
        "brows_up": ("brows", brow_box, 6, lambda t: brow_field(pts, brow_box, "up", 3.2 * t, 0.0)),
    }
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    layers, h = {}, hashlib.sha1()
    for name, (region, box, n, field) in specs.items():
        (OUT / name).mkdir()
        a = alpha_mask(box)
        frames = []
        for i in range(1, n + 1):
            t = float(smooth(np.float32(i / n)))          # ease in / out: the sequence is played at a constant rate
            dx, dy = field(t)
            fr = warp_crop(base, box, dx, dy)
            fr[..., 3] = fr[..., 3] * a
            im = Image.fromarray(np.clip(np.round(fr), 0, 255).astype(np.uint8), "RGBA")
            p = OUT / name / f"f{i:02d}.webp"
            im.save(p, lossless=True, quality=100, method=4)
            h.update(p.read_bytes())
            frames.append(f"{name}/f{i:02d}.webp")
        mx = float(np.abs(np.stack(field(1.0))).max())
        layers[name] = {"region": region, "box": box[:2], "size": [box[2] - box[0], box[3] - box[1]], "frames": frames,
                        "max_px": round(mx, 2)}
        print(f"{name:14s} {region:6s} box {box} frames {n} max displacement {mx:.2f} px")
    manifest = {
        "version": h.hexdigest()[:12],
        "layers": layers,
        # resting face per conversational state (embed.html setReaction): listening stays neutral-attentive
        "moods": {"greeting": "smile_warm", "backchannel": "smile_soft", "listening": None},
        "note": "generate_expressions.py: warps of mouths/base.png (no painted pixels); fields are 0 at the box border",
    }
    (HERE / "expressions.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("wrote", OUT, "and expressions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
