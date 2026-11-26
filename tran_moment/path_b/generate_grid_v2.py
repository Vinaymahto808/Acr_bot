"""Mouth atlas v2 for the card's continuous lips (CPU: numpy + scipy + pillow + opencv).

Why v2 (2026-11-24 owner feedback "blur between the lips" / "teeth do not move like a real person"):
  * v1 opened the mouth by STRETCHING the photo's teeth band to the new height (tall rubbery teeth) and closed it
    by squeezing the teeth into a smear; the card then blended the 4 nearest cells every frame, so a closed cell
    (no teeth) and an open cell (teeth) were double-exposed -> a grey ghost line between the lips.
  * v2 models the anatomy instead: the upper teeth are rigid (attached to the skull); the lower lip and chin drop
    with the jaw and the gap that opens between upper teeth and lower lip is a dark oral cavity (with the lower
    incisor tips peeking above the lower lip on wide openings). Closing slides the lips OVER the teeth (occlusion),
    the teeth are never squeezed. Every cell is therefore a clean image of one geometry.
  * The grid is dense (18 openness x 7 width steps, <= 2 px of jaw between neighbours) so the card can show the
    NEAREST cell with no cross-fade at all.
  * All displacement fades to zero inside the cell well before its feathered edge, so the soft mask only ever
    blends identical pixels with base.png (no doubled chin / lip contour at the patch border).

    uv run --no-project --python 3.12 --with numpy --with pillow --with scipy --with opencv-python \
        python tran_moment/path_b/generate_grid_v2.py          # writes mouths/grid2.jpg + the "grid" entry of mouths.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, median_filter, gaussian_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_mouths as gm  # noqa: E402  (clean-teeth pass + sharpen, shared with base.png)

OPENS = [-1.0, -0.8, -0.6, -0.45, -0.3, -0.15, 0.0, 0.07, 0.14, 0.22, 0.3, 0.38, 0.47, 0.56, 0.66, 0.76, 0.88, 1.0]
WIDTHS = [-0.95, -0.62, -0.3, 0.0, 0.3, 0.62, 0.92]
JAW = 0.15          # jaw drop at open=1, as a fraction of the mouth width (~16 px on this portrait)
FEATHER = 10        # px of the card's soft mask at the cell border (written to mouths.json)
TUCK_W = 1.6        # sentinel "width" of the F/V row (outside -1..1: the card selects it explicitly, never by easing)


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def teeth_profile(clean: np.ndarray, pts: dict):
    """Per column: top and bottom of the visible upper-teeth band (y), and the lips' meeting line at the corners."""
    lx, rx = pts["mouth_left"][0], pts["mouth_right"][0]
    up, lo = pts["lip_upper_inner"][1], pts["lip_lower_inner"][1]
    mid = (up + lo) / 2.0
    W = clean.shape[1]
    top = np.full(W, np.nan, np.float32); bot = np.full(W, np.nan, np.float32)
    y0, y1 = int(up - 6), int(lo + 6)
    for x in range(int(lx) + 2, int(rx) - 1):
        col = clean[y0:y1, x].astype(np.int32)
        red = col[:, 0] - col[:, 1]
        teeth = (red < 40) & (col.sum(axis=1) > 450)
        idx = np.where(teeth)[0]
        if len(idx) >= 3:
            top[x], bot[x] = y0 + idx[0], y0 + idx[-1] + 1.0
    xs = np.arange(W, dtype=np.float32)
    good = ~np.isnan(top)
    gx = xs[good]
    # corners: the band closes to the lips' meeting line at the mouth corners
    cx_pts = np.concatenate([[lx], gx, [rx]])
    t_pts = np.concatenate([[pts["mouth_left"][1]], top[good], [pts["mouth_right"][1]]])
    b_pts = np.concatenate([[pts["mouth_left"][1]], bot[good], [pts["mouth_right"][1]]])
    t_pts = median_filter(t_pts, 5, mode="nearest"); b_pts = median_filter(b_pts, 5, mode="nearest")
    T = np.interp(xs, cx_pts, t_pts); B = np.interp(xs, cx_pts, b_pts)
    T = gaussian_filter1d(T, 2.5); B = gaussian_filter1d(B, 2.5)
    inside = (xs > lx) & (xs < rx)
    T = np.where(inside, T, mid); B = np.where(inside, B, mid)
    return T.astype(np.float32), B.astype(np.float32), float(gx.min()), float(gx.max())


def make_cell(clean, pts, prof, box, o, k):
    """One cell, antialiased: the occlusion edges (lip over teeth, cavity) are hard per-pixel decisions, so where
    four quarter-pixel-offset renders disagree (an edge) their mean is used; elsewhere the exact render (no blur)."""
    c0 = _cell(clean, pts, prof, box, o, k, 0.0, 0.0)
    subs = [_cell(clean, pts, prof, box, o, k, dx, dy) for dx in (-0.25, 0.25) for dy in (-0.25, 0.25)]
    mean = np.mean(subs, axis=0)
    spread = np.max(np.abs(np.array(subs) - c0[None]), axis=(0, 3))
    w = np.clip((spread - 12.0) / 20.0, 0, 1)[..., None]
    return np.clip(c0 * (1 - w) + mean * w, 0, 255).astype(np.uint8)


def _cell(clean, pts, prof, box, o, k, ox, oy):
    x0, y0, x1, y1 = box
    T, B, tx0, tx1 = prof
    lx, rx = pts["mouth_left"][0], pts["mouth_right"][0]
    cx, mw = (lx + rx) / 2.0, rx - lx
    up, lo = pts["lip_upper_inner"][1], pts["lip_lower_inner"][1]
    lipY = (up + lo) / 2.0
    nose, chin = pts["nose_bottom"][1], pts["chin"][1]
    lip_top, lip_bot = pts["lip_top"][1], pts["lip_bottom"][1]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    yy += oy; xx += ox
    # displacement window: 1 in the core, 0 within FEATHER+2 px of the cell border
    m = FEATHER + 2
    win = (smooth((xx - x0 - m) / FEATHER) * smooth((x1 - 1 - m - xx) / FEATHER)
           * smooth((yy - y0 - m) / FEATHER) * smooth((y1 - 1 - m - yy) / FEATHER))
    d = np.abs(xx - cx) / (mw / 2.0)
    fall = (np.where(d <= 1.15, np.cos(np.clip(d / 1.15, 0, 1) * np.pi / 2.0), 0.0) ** 1.3).astype(np.float32)
    # 1) lips spread (E/I) or round (O/U): horizontal scale around the mouth centre, damped away from the lips
    sx = 1.0 + k * 0.12
    vert = np.exp(-((yy - lipY) / (0.95 * (lip_bot - lip_top))) ** 2)
    hw = vert * fall * win
    xs = xx + ((xx - cx) / sx - (xx - cx)) * hw
    Tx = np.interp(xs, np.arange(len(T)), T); Bx = np.interp(xs, np.arange(len(B)), B)
    sy = yy.copy()
    cav = np.zeros_like(yy)           # coverage of the oral cavity (0..1)
    cav_t = np.zeros_like(yy)         # depth inside the cavity, 0 at the upper teeth .. 1 at the lower lip
    J = np.zeros_like(yy)
    if o > 0:
        # 2) jaw: everything below the teeth band (lower lip, chin) drops; upper lip + upper teeth stay
        # lens-shaped opening: deepest in the middle, closing smoothly into the mouth corners
        fx = np.clip(1 - ((xs - cx) / (0.5 * mw * 0.97)) ** 2, 0, 1) ** 0.75 * win
        J = o * JAW * mw * fx
        # the lower lip follows the jaw fully; the chin moves less (0.45) so the jaw line, which the patch cannot
        # move all the way to the ears, never kinks at the patch edge (seen 2026-11-24 at 16 px)
        mid_w = 1 - 0.55 * smooth((yy - (lip_bot + 4)) / max(4.0, chin - lip_bot - 4))
        wl = np.where(yy < lip_bot + 4, 1.0, np.where(yy < chin, mid_w, 0.45 * smooth((chin + 28 - yy) / 28.0)))
        below = yy >= Bx + J
        # skip the grey teeth-to-lip transition rows at the lower lip's inner edge (they drew a grey rim)
        rim = 1.8 * np.clip(J / 2, 0, 1) * np.exp(-((yy - (Bx + J)) / 7.0) ** 2)
        sy = np.where(below, yy - J * wl + rim, sy)
        # the cavity: rows [B, B+J) with antialiased coverage at both edges
        top_cov = np.clip(yy + 0.5 - Bx, 0, 1)
        bot_cov = np.clip(Bx + J - (yy - 0.5), 0, 1)
        cav = np.where(J > 0.05, np.minimum(top_cov, bot_cov), 0.0)
        cav_t = np.clip((yy - Bx) / np.maximum(J, 1e-3), 0, 1)
        # cavity pixels sample the lip edge below (bottom half) or the teeth (top half) for the antialiased blend
        sy = np.where((yy >= Bx) & ~below, np.where(cav_t > 0.5, Bx + 0.3, Bx - 0.7), sy)
    elif o < 0:
        # 3) closing: the lips slide over the (rigid) teeth from above and below; at -1 they meet on the mid line
        c = -o
        gap = np.maximum(Bx - Tx, 0)
        Dm = c * (gap / 2.0 + 1.0)
        midl = (Tx + Bx) / 2.0
        wu = smooth((yy - nose) / max(4.0, up - nose))            # upper lip moves, nose stays
        wl2 = np.where(yy < chin - 30, 1.0, smooth((chin - yy) / 30.0))
        Du, Dl = Dm * win, Dm * win
        upper_out = yy < Tx + Du
        lower_out = yy >= Bx - Dl
        if c >= 0.999:
            upper_out = yy < midl; lower_out = ~upper_out
        else:
            both = upper_out & lower_out
            upper_out = upper_out & ~(both & (yy >= midl)); lower_out = lower_out & ~(both & (yy < midl))
        rim_u = 1.6 * smooth(c / 0.3) * np.exp(-((yy - (Tx + Du)) / 6.0) ** 2)   # and the upper lip's grey inner rim
        sy = np.where(upper_out, yy - Du * np.where(yy >= Tx, 1.0, wu) - rim_u, sy)
        rim_c = 1.8 * smooth(c / 0.3) * np.exp(-((yy - (Bx - Dl)) / 7.0) ** 2)   # skip the grey teeth/lip rim rows
        sy = np.where(lower_out, yy + Dl * np.where(yy <= Bx, 1.0, wl2) + rim_c, sy)
        teeth_rows = ~upper_out & ~lower_out
        remain = np.maximum(gap - 2 * Dm, 0)
        shade_close = np.where(teeth_rows, 1.0 - 0.55 * (1 - smooth(remain / 10.0)) * smooth(gap / 5.0), 1.0)
    src = np.array([sy, xs])
    out = np.stack([map_coordinates(clean[..., ch].astype(np.float32), src, order=3, mode="nearest") for ch in range(3)], -1)
    if o > 0:
        # cavity colour: dark under the upper teeth, a warmer tongue tone deeper, lower incisor tips on wide openings
        top_c = np.array([70, 30, 34], np.float32); deep_c = np.array([104, 46, 52], np.float32)
        tdeep = smooth((cav_t - 0.3) / 0.7)[..., None] * np.clip((J[..., None] - 4) / 10, 0, 1)
        thin = np.clip((J[..., None] - 1.5) / 5.0, 0, 1)         # a hairline gap is a soft shadow, not a black line
        top_c = np.array([128, 66, 66], np.float32) * (1 - thin) + top_c * thin
        col = top_c * (1 - tdeep) + deep_c * tdeep
        side = (0.70 + 0.30 * np.clip(1 - np.abs(xs - cx) / (mw * 0.42), 0, 1) ** 0.7)[..., None]
        col = col * side
        tc = (np.clip(np.abs(xs - cx) / (mw * 0.5), 0, 1) ** 2 * 0.6)[..., None]   # corners: lip shadow, not a black wedge
        col = col * (1 - tc) + np.array([122, 62, 64], np.float32) * tc
        tip_h = np.clip((J - 9) * 0.45, 0, 3.2)[..., None]   # px of lower teeth visible above the lower lip
        across = smooth((1 - np.abs(xs - cx) / (mw * 0.2)) / 0.4)[..., None]
        tip_a = np.clip((yy[..., None] - (Bx + J)[..., None] + tip_h) / 1.2, 0, 1) * across * (tip_h > 0.2)
        tooth = np.array([214, 204, 198], np.float32)
        col = col * (1 - tip_a * 0.85) + tooth * tip_a * 0.85
        # a soft shadow under the upper teeth edge (1 px) so they read as in front of the cavity
        a = cav[..., None]
        out = out * (1 - a) + col * a
        edge = np.exp(-((yy - Bx) / 0.9) ** 2) * (J > 0.5) * np.clip(J / 3, 0, 1)
        out = out * (1 - 0.18 * edge[..., None] * (cav_t < 0.5)[..., None])
    if o < 0:
        dk = (1 - shade_close)[..., None]            # teeth in the lips' shadow: darker and warmer, not grey
        out = out * (1 - dk * np.array([0.85, 1.0, 1.0], np.float32))
    if o < -0.6:
        # lips pressed together: a soft seam line where they meet, strongest in the middle
        c = -o
        midl = (Tx + Bx) / 2.0
        across = np.clip(1 - np.abs(xs - cx) / (mw / 2), 0, 1) ** 0.6
        seam = np.exp(-((yy - midl) / 0.9) ** 2) * across * smooth((c - 0.6) / 0.4) * win
        out = out * (1 - 0.30 * seam[..., None])
    return np.clip(out, 0, 255)


def main_closed_lips(pts: dict, rgb: np.ndarray) -> int:
    """Portrait with CLOSED lips (no teeth band to keep rigid): the cells come from the avatar v2 renderer
    (tran_moment/avatar_v2/build_reference_pack.py), which adds procedural upper teeth hanging from the upper lip,
    the cavity, tongue and lower lip contact shadow, centred on the lips' own axis. Same atlas layout and mouths.json
    "grid" entry as below, so the card does not change."""
    sys.path.insert(0, str(HERE.parent / "avatar_v2"))
    import build_reference_pack as rp  # noqa: PLC0415
    F = rp.Face(rgb, pts)
    x0, y0, x1, y1 = rp.mouth_box(F)   # the box from the pack's geometry (unchanged: the card and av2Between rely on it)
    # round 2 (2026-11-25, owner: "ah / E too wide, a black slot"): speech is mostly small-to-medium openings. The lip
    # grid's peak aperture is 65 % of the pack's (JAW 0.29 -> 0.19) and the opening stops short of the corners.
    rp.JAW = 0.19
    rp.OPEN_SPAN = 0.86
    # round 3 (2026-11-25, owner: "upper lip reactions, left / right"): the upper lip is no longer frozen. It lifts with the
    # opening (~3 px at the loudest "ah", ~1.5 px on E, less toward the corners; the philtrum follows, 0 at the nose) and
    # the upper teeth hang from the lifted edge, so more of them shows; E / I push the corners out further and up a
    # little; a rounded upper lip is a touch fuller. F / V get their own row (TUCK_W): the lower lip under the teeth.
    rp.UPLIFT = 0.045            # round 4: 0.036 was still subtle at phone size
    rp.CHEEK = 1.0
    rp.PHILTRUM = 0.6
    # round 4: the patch reaches up beside the nose (to just under the eye layers, which end at y ~246) and out over
    # the cheeks, so the skin there can move with the mouth; every displacement is still 0 at the patch border
    x0, x1 = int(F.cxm - 1.45 * F.mw), int(F.cxm + 1.45 * F.mw)
    y0 = int(pts["nose_tip"][1] - 41)
    rp.SPREAD_PUSH = 0.14
    rp.CORNER_UP = 0.014
    rp.ROUND_UPPER_FULL = 0.09
    bw, bh = x1 - x0, y1 - y0
    widths = [0.0] if "--mid" in sys.argv else WIDTHS + [TUCK_W]
    atlas = Image.new("RGB", (bw * len(OPENS), bh * len(widths)))
    tuck_cell = None
    for r, k in enumerate(widths):
        if k == TUCK_W:   # F / V: one pose (lower lip tucked under the upper incisors), the same in every column
            if tuck_cell is None:
                tuck_cell = rp.finish_cell(*rp.render_aa(F, {**rp.DEFAULTS, "a": -0.05, "k": 0.1, "tuck": 1.0}, [x0, y0, x1, y1]))
            for c in range(len(OPENS)):
                atlas.paste(tuck_cell, (c * bw, r * bh))
            continue
        rnd = max(0.0, -k)
        for c, o in enumerate(OPENS):
            # lower incisor tips on the widest openings, as the pack's aa (lt 0 below o 0.7, 0.6 at o 1); none through
            # rounded lips (the pucker covers them)
            lt = max(0.0, min(0.6, (o - 0.7) * 2.0)) * (1.0 - rnd)   # only the widest openings (a sliver on mid ones read as a grey line)
            # rounded lips ("oo", 2026-11-25): a smaller opening than an "ah" of the same loudness (the jaw drops less)
            a = o * (1.0 - 0.45 * rnd) if o > 0 else o
            # rounding grows with the opening: a closed or barely parted mouth is never a pout (round 2: pursed lips
            # before the first word / between words, from a round width carried over a closure)
            kk = k if k >= 0 else k * (0.25 + 0.75 * float(rp.smooth(np.float32((o - 0.05) / 0.35))))
            # the deepest closures are M/B/P: lips pressed and rolled in a little (the portrait itself is already closed,
            # so without the press "mm" looked exactly like the resting face)
            press = float(rp.smooth(np.float32((-o - 0.6) / 0.4))) if o < -0.6 else 0.0
            p = {**rp.DEFAULTS, "a": a, "k": kk, "tongue": "low" if o > 0.2 else None, "tg": 0.9, "lt": lt, "press": press}
            atlas.paste(rp.finish_cell(*rp.render_aa(F, p, [x0, y0, x1, y1])), (c * bw, r * bh))
    dest = gm.OUT / "grid2.jpg"
    atlas.save(dest, quality=95, subsampling=0)
    idx_path = HERE / "mouths.json"
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    index["grid"] = {"file": "grid2.jpg", "box": [x0, y0, bw, bh], "opens": OPENS, "widths": widths, "tuck_w": TUCK_W,
                     "feather": rp.FEATHER, "snap": True,
                     "note": "generate_grid_v2.py (closed-lips portrait: avatar_v2 renderer, procedural upper teeth); nearest cell, no cross-fade"}
    idx_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("wrote", dest, atlas.size, "box", [x0, y0, bw, bh], "(closed lips: avatar_v2 renderer)")
    return 0


def main() -> int:
    pts = json.loads((gm.PREP / "face.json").read_text(encoding="utf-8"))["points"]
    rgb = np.asarray(Image.open(gm.PREP / "face.jpg").convert("RGB"), dtype=np.uint8)
    if pts["lip_lower_inner"][1] - pts["lip_upper_inner"][1] < 5 and "--preview" not in sys.argv:
        return main_closed_lips(pts, rgb)
    clean = gm._clean_teeth_seam(rgb, pts)
    prof = teeth_profile(clean, pts)
    lx, rx = pts["mouth_left"][0], pts["mouth_right"][0]
    cx, mw = (lx + rx) / 2.0, rx - lx
    box = [int(cx - mw * 0.83), int(pts["nose_bottom"][1] - 14), int(cx + mw * 0.85), int(pts["chin"][1] + 52)]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    widths = [0.0] if "--mid" in sys.argv else WIDTHS      # --mid: one row only (fast preview)
    atlas = Image.new("RGB", (bw * len(OPENS), bh * len(widths)))
    for r, k in enumerate(widths):
        for c, o in enumerate(OPENS):
            cell = gm.sharpen(Image.fromarray(make_cell(clean, pts, prof, box, o, k)))
            atlas.paste(cell, (c * bw, r * bh))
    preview = "--preview" in sys.argv          # --preview DIR: write the atlas there, leave mouths.json alone
    dest = Path(sys.argv[sys.argv.index("--preview") + 1]) / "grid2.jpg" if preview else gm.OUT / "grid2.jpg"
    atlas.save(dest, quality=95, subsampling=0)
    if preview:
        print("preview", dest, atlas.size, "box", [x0, y0, bw, bh])
        return 0
    idx_path = HERE / "mouths.json"
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    index["grid"] = {"file": "grid2.jpg", "box": [x0, y0, bw, bh], "opens": OPENS, "widths": WIDTHS,
                     "feather": FEATHER, "snap": True,
                     "note": "generate_grid_v2.py: rigid upper teeth + jaw cavity; draw the nearest cell, no cross-fade"}
    idx_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("wrote", dest, atlas.size, "box", [x0, y0, bw, bh])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
