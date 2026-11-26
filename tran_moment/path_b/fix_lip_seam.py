"""Remove the smeared teeth line between closed / half-closed lips in the puppet's mouth assets (CPU, OpenCV).

The portrait's lips are parted, so generate_mouths.py closes them by sliding the lips together. That squeezes the
teeth into a thin white smear along the middle of the lips instead of removing them; the card shows it at every
pause between words (lips relax to open -0.45) and on every M/B/P. Reported 2026-11-24: "lips ke middle me blur".

This pass, for every grid cell whose openness is below 0 (and for closed.png): finds the whitish (teeth) pixels in a
band around the inner lip line, fills them with lip colour from above and below (inpaint), and draws a soft lip seam.
The open cells (teeth meant to show) are untouched. Run it after generate_mouths.py:

    python tran_moment/path_b/fix_lip_seam.py            # rewrites mouths/grid.jpg + mouths/closed.png in place
    python tran_moment/path_b/fix_lip_seam.py --check    # only report the teeth pixels left in the closed cells
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MOUTHS = HERE / "mouths"
FACE = HERE.parent / "prepared" / "face.json"


def teeth_mask(bgr: np.ndarray, band: tuple[int, int, int, int]) -> np.ndarray:
    """Whitish, low-saturation pixels inside the band (x0, y0, x1, y1): what is left of the teeth."""
    x0, y0, x1, y1 = band
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    lip_s = np.median(s[y0:y1, x0:x1])
    m = np.zeros(bgr.shape[:2], np.uint8)
    sub = (s[y0:y1, x0:x1] < lip_s * 0.62) & (v[y0:y1, x0:x1] > 150)
    m[y0:y1, x0:x1] = sub.astype(np.uint8) * 255
    # the smear is a long thin horizontal run; the lower lip's shine is a small blob: keep only wide components
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_WIDTH] >= 10 and stats[i, cv2.CC_STAT_WIDTH] >= 2.5 * stats[i, cv2.CC_STAT_HEIGHT]:
            keep[lab == i] = 255
    return cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)


def fix(bgr: np.ndarray, pts: dict, dx: int, dy: int) -> tuple[np.ndarray, int]:
    """Fill the teeth smear and draw the seam. pts are portrait coords, (dx, dy) the image's offset in the portrait."""
    lx, rx = pts["mouth_left"][0] - dx, pts["mouth_right"][0] - dx
    up, lo = pts["lip_upper_inner"][1] - dy, pts["lip_lower_inner"][1] - dy
    mid = (up + lo) / 2.0
    band = (int(lx + 3), int(mid - 6), int(rx - 3), int(mid + 5))   # tight: the lips meet here; their shine is outside
    mask = teeth_mask(bgr, band)
    n = int((mask > 0).sum())
    if not n:
        return bgr, 0
    out = cv2.inpaint(bgr, mask, 4, cv2.INPAINT_TELEA)
    # seam: where the lips meet, a soft darker line, strongest in the middle, fading to the corners
    H, W = out.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # one smooth curve (quadratic through the smear's column centres): a per-column line came out stepped
    cols = np.where(mask.any(axis=0))[0]
    centres = np.array([np.where(mask[:, c] > 0)[0].mean() for c in cols], np.float32)
    if len(cols) >= 6:
        k = np.polyfit(cols.astype(np.float32), centres, 2)
        seam_y = np.polyval(k, np.arange(W, dtype=np.float32)).astype(np.float32)
    else:
        seam_y = np.full(W, centres.mean() if len(cols) else mid, np.float32)
    half = (rx - lx) / 2.0
    across = np.clip(1.0 - np.abs(xx - (lx + rx) / 2.0) / max(half, 1.0), 0, 1) ** 0.6
    near = np.exp(-((yy - seam_y[None, :]) ** 2) / (2 * 0.9 ** 2))
    shade = 1.0 - 0.32 * near * across * ((xx > lx) & (xx < rx))
    out = np.clip(out.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)
    return out, n


def main(argv: list[str]) -> int:
    check = "--check" in argv
    meta = json.loads((HERE / "mouths.json").read_text(encoding="utf-8"))
    pts = json.loads(FACE.read_text(encoding="utf-8"))["points"]
    bx, by, bw, bh = meta["grid"]["box"]
    opens, widths = meta["grid"]["opens"], meta["grid"]["widths"]
    atlas = cv2.imread(str(MOUTHS / meta["grid"]["file"]), cv2.IMREAD_COLOR)
    total = 0
    for r in range(len(widths)):
        for c, o in enumerate(opens):
            if o >= 0:
                continue
            cell = atlas[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            fixed, n = fix(cell.copy(), pts, bx, by)
            total += n
            if not check:
                atlas[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw] = fixed
    # closed.png: the 9-shape fallback (full portrait canvas, RGBA)
    closed = cv2.imread(str(MOUTHS / "closed.png"), cv2.IMREAD_UNCHANGED)
    fixed, n_closed = fix(closed[..., :3].copy(), pts, 0, 0)
    print(f"teeth pixels in closed cells: {total}  closed.png: {n_closed}")
    if check:
        return 0
    if total == 0 and n_closed == 0:   # already clean: do not re-encode the JPEG atlas (each save loses a little)
        print("nothing to fix; files left as they are")
        return 0
    cv2.imwrite(str(MOUTHS / meta["grid"]["file"]), atlas, [cv2.IMWRITE_JPEG_QUALITY, 93])
    closed[..., :3] = fixed
    cv2.imwrite(str(MOUTHS / "closed.png"), closed)
    after = sum(fix(atlas[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw].copy(), pts, bx, by)[1]
                for r in range(len(widths)) for c, o in enumerate(opens) if o < 0)
    print(f"after: teeth pixels left in closed cells: {after}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
