"""Avatar v2 reference pack: one aligned mouth image per viseme (and intensity) for ONE portrait, plus a manifest.

    uv run --no-project --python 3.13 --with numpy --with pillow --with scipy --with opencv-python \
        python tran_moment/avatar_v2/build_reference_pack.py                       # current Neha portrait
    ... build_reference_pack.py --photo tran_moment/face_new.png                    # any front-facing portrait
        (needs --with mediapipe too: landmarks via tran_moment/prepare_portrait.py into avatar_v2/portrait/)

Writes tran_moment/avatar_v2/pack/:
    atlas.jpg     every frame as one cell of the mouth box (what the card draws)
    pack.json     manifest: box, feather, cell grid, viseme -> frames per intensity, the render parameters
    frames/*.png  the full-intensity frame of every viseme (the human-readable reference pack)
    sheet.png     labelled contact sheet for review
    base.png + face.json   only for a portrait other than the default (the card then draws this base)

"Reference pack" (idea from corpunum/UnumAvatar: per-viseme reference mouth images of one identity; that repo has
no licence, so no code or image of it is used). Here the references are RENDERED from the portrait itself, so
identity, lighting and grain match the face pixel for pixel. What v2 changes against path_b/generate_grid_v2.py:
  * symmetric about the LIP MIDLINE (philtrum), not the midpoint of the corners: the portrait is turned a little,
    so the corners are 43 / 54 px from the midline; v1 centred the opening 5 px off the lips' axis -> crooked look.
    Every displacement is a function of u = signed distance from the midline, normalised by that side's half-width.
  * a modelled inside: graded cavity (dark under the upper teeth, warmer deeper), a lit tongue (low / tip-up /
    between the teeth), lower incisors with per-tooth edges, notches between the upper incisal edges, a contact
    shadow on the lower lip; a light, deterministic grain so painted areas carry the photo's texture.
  * viseme-specific lips: pressed + thinned for M/B/P, lower lip tucked under the teeth for F/V, rounded and fuller
    for O/U/CH, spread for E/I/S, teeth together for S/CH.
  * edge-only antialiasing (3x3 subsamples where the layer or the warp is discontinuous), everything else is the
    exact cubic sample: no blur in the middle of the lips.
  * upper teeth are rigid (attached to the skull); a portrait with CLOSED lips gets procedural upper teeth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, median_filter

HERE = Path(__file__).resolve().parent
TM = HERE.parent
sys.path.insert(0, str(TM / "path_b"))
import generate_mouths as gm  # noqa: E402  (teeth-seam clean + the shared sharpen, same as base.png)

LEVELS = (0.4, 0.7, 1.0)      # intensities per viseme (the card picks one from cue weight x loudness)
JAW = 0.29                    # jaw drop at a=1 as a fraction of the mouth width (~27 px on this portrait; 0.22 read as "barely opens", 2026-11-25)
UPLIFT = 0.0                  # upper-lip lift at a=1 (fraction of the mouth width), plus 0.015 x rounding. 0: owner
                              # feedback 2026-11-25 - the upper lip must stay almost still; speech moves the lower lip + jaw
FEATHER = 10                  # px of soft mask at the cell border (every displacement is 0 there)
# rounded lips (O / U / CH, k < 0), 2026-11-25 owner reference "oo = rounded, small opening": corners pulled in harder,
# the opening an ellipse in the middle ~half of the mouth, the upper lip's middle arching up a little with it
# (was 0.24 / 0.30 / 0.015: a wide slot with the lower lip sagging under it)
ROUND_PULL, ROUND_NARROW, ROUND_LIFT = 0.30, 0.46, 0.035
OPEN_SPAN = 1.0               # opening reaches this far toward the corners (|u|); the lip grid uses 0.86
CHIN_FOLLOW = 0.5             # share of the jaw drop the chin follows (was 0.32)
PRESS_THIN = 0.24
# round 3 (2026-11-25, owner: "upper lip reactions, left / right"): the grid build sets UPLIFT > 0 (the upper lip lifts on
# open vowels, the philtrum follows, 0 at the nose); spread vowels push the corners out a little more and up; a rounded
# upper lip is a touch fuller (pushed forward). Defaults keep the pack as it was.
SPREAD_PUSH = 0.10
CORNER_UP = 0.0
ROUND_UPPER_FULL = 0.04
CHEEK = 0.0                   # round 4: cheek coupling beside the nose (px scale); the lip grid sets 1.0
PHILTRUM = 1.0                # round 4: <1 makes the skin above the upper lip follow its lift more (grid: 0.6)             # lips thinner at full press (M/B/P); was 0.14
SS = 3                        # subsamples per axis on discontinuities
DEFAULTS = dict(a=0.0, k=0.0, press=0.0, tuck=0.0, tongue=None, tg=0.0, lt=0.0, bite=0.0)
# full-intensity parameters. a: aperture (-1 lips pressed .. 0 portrait .. 1 jaw fully dropped); k: -1 round .. 1 spread;
# press: lips thinned; tuck: lower lip under the upper teeth; tongue/tg: kind/amount; lt: lower teeth; bite: teeth together
VISEMES = {
    "sil": dict(a=-0.8),
    "PP":  dict(a=-1.0, press=1.0),
    "FF":  dict(a=-0.05, tuck=1.0, k=0.1),
    "TH":  dict(a=0.26, k=0.15, tongue="between", tg=1.0, lt=0.3),
    "DD":  dict(a=0.30, k=0.25, tongue="tip", tg=1.0, lt=0.5),
    "kk":  dict(a=0.40, k=0.12, tongue="low", tg=1.25, lt=0.45),
    "CH":  dict(a=0.20, k=-0.6, bite=1.0),
    "SS":  dict(a=0.17, k=0.55, bite=1.0),
    "nn":  dict(a=0.26, k=0.2, tongue="tip", tg=0.8, lt=0.35),
    "RR":  dict(a=0.30, k=-0.45, tongue="low", tg=0.8, lt=0.2),
    "aa":  dict(a=1.0, k=0.05, tongue="low", tg=1.0, lt=0.6),
    "E":   dict(a=0.58, k=0.7, tongue="low", tg=1.0),
    "I":   dict(a=0.38, k=0.92, tongue="low", tg=1.1),
    "O":   dict(a=0.68, k=-0.9, tongue="low", tg=0.8, lt=0.15),
    "U":   dict(a=0.36, k=-1.0, tongue="low", tg=0.6),
}
SINGLE = ("sil", "PP", "FF")
REST_A = -0.3     # weak-syllable base for a parted-lips portrait; 0 (the portrait itself) when the lips are closed  # one intensity: a closure is a closure

# round 2 (2026-11-25): "near-black slot" -> a warm dark red-brown inside, as in the owner's reference
CAV_TOP = np.array([78, 32, 34], np.float32)      # under the upper teeth
CAV_DEEP = np.array([116, 50, 54], np.float32)    # deeper, toward the floor of the mouth
LIP_SHADOW = np.array([126, 64, 66], np.float32)  # a hairline gap reads as lip shadow, not black
MOUTH_GAP = np.array([88, 40, 44], np.float32)    # teeth seen through nearly closed lips: in shadow, warm, never grey
TONGUE = np.array([176, 86, 92], np.float32)
CAV_BACK = np.array([118, 46, 54], np.float32)    # back of the mouth in the middle, a little lit
INNER_LIP = np.array([150, 58, 68], np.float32)   # wet inner surface of the lower lip, in the mouth's shade
TONGUE_LIT = np.array([212, 132, 130], np.float32)
TOOTH = np.array([216, 207, 199], np.float32)
LOWER_TOOTH = np.array([224, 209, 196], np.float32)
UPPER_TOOTH = np.array([238, 231, 221], np.float32)   # synthetic upper incisors (closed-lips portrait): lit, warm white   # warmer: lit by the mouth, not the sky


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def params_at(vis: str, level: float) -> dict:
    """Parameters of `vis` at an intensity: interpolated from the speech rest (sil) to the full shape."""
    rest = {**DEFAULTS, "a": REST_A}        # weak syllable base: lips almost together, teeth just showing
    full = {**DEFAULTS, **VISEMES[vis]}
    if vis in SINGLE or level >= 0.999:
        return full
    p = dict(full)
    for key in ("a", "k", "press", "tuck", "tg", "lt", "bite"):
        p[key] = rest[key] + (full[key] - rest[key]) * level
    return p


class Face:
    """Landmarks + the photo's upper-teeth band, in the coordinates the renderer needs."""

    def __init__(self, rgb: np.ndarray, pts: dict):
        self.pts = pts
        P = lambda k: pts[k]  # noqa: E731
        self.lx, self.rx = P("mouth_left")[0], P("mouth_right")[0]
        self.up, self.lo = P("lip_upper_inner")[1], P("lip_lower_inner")[1]
        self.lip_top, self.lip_bot = P("lip_top")[1], P("lip_bottom")[1]
        self.nose, self.chin = P("nose_bottom")[1], P("chin")[1]
        # the lips' own axis (philtrum .. chin dimple), not the midpoint of the corners
        self.cxm = float(np.median([P("lip_top")[0], P("lip_upper_inner")[0], P("lip_lower_inner")[0], P("lip_bottom")[0]]))
        self.hl, self.hr = max(8.0, self.cxm - self.lx), max(8.0, self.rx - self.cxm)
        self.mw = self.rx - self.lx
        self.lipY = (self.up + self.lo) / 2.0
        self.clean = gm._clean_teeth_seam(rgb, pts).astype(np.float32)
        self.has_teeth = self.lo - self.up >= 5
        self.T, self.B = self._teeth_band()
        self.gaps = self._tooth_gaps()
        self.donor = load_donor() if not self.has_teeth else None

    def u(self, x):
        """signed, side-normalised distance from the lip midline: -1 left corner, 0 midline, +1 right corner"""
        return np.where(x < self.cxm, (x - self.cxm) / self.hl, (x - self.cxm) / self.hr)

    def _teeth_band(self):
        W = self.clean.shape[1]
        xs = np.arange(W, dtype=np.float32)
        ml, mr = self.pts["mouth_left"][1], self.pts["mouth_right"][1]
        if not self.has_teeth:
            # closed lips: the meeting line. Model: a shallow arc from corner to corner through the inner-lip midpoint;
            # then traced in the photo (the darkest row within +-4 px of the arc, sub-pixel) so the boundary sits ON
            # the dark line. With the arc alone it was 2-3 px low halfway to the corners: a sliver of the lower lip
            # stayed on the upper lip and showed as a bright pink seam above the teeth (2026-11-25).
            uu = np.clip(np.abs(self.u(xs)), 0, 1)
            side = np.where(xs < self.cxm, ml, mr)
            line = (self.lipY + (side - self.lipY) * uu ** 2).astype(np.float32)
            lum = gaussian_filter1d(self.clean @ np.array([0.299, 0.587, 0.114], np.float32), 0.6, axis=0)
            tx, ty = [], []
            for x in range(int(self.lx) + 3, int(self.rx) - 2):
                c = int(round(line[x]))
                col = lum[c - 4:c + 5, x]
                i = int(np.argmin(col))
                if 0 < i < len(col) - 1:
                    d = col[i - 1] - 2 * col[i] + col[i + 1]
                    off = 0.5 * (col[i - 1] - col[i + 1]) / d if d > 1e-3 else 0.0
                    tx.append(x); ty.append(c - 4 + i + float(np.clip(off, -0.5, 0.5)))
            if len(tx) > 10:
                ty = gaussian_filter1d(median_filter(np.array(ty), 5, mode="nearest"), 1.5, mode="nearest")
                traced = np.interp(xs, np.concatenate([[self.lx], tx, [self.rx]]), np.concatenate([[ml], ty, [mr]]))
                inside = (xs > self.lx) & (xs < self.rx)
                line = np.where(inside, traced, line).astype(np.float32)
            return line, line
        top = np.full(W, np.nan, np.float32)
        bot = np.full(W, np.nan, np.float32)
        y0, y1 = int(self.up - 6), int(self.lo + 6)
        for x in range(int(self.lx) + 2, int(self.rx) - 1):
            col = self.clean[y0:y1, x].astype(np.int32)
            teeth = ((col[:, 0] - col[:, 1]) < 40) & (col.sum(axis=1) > 450)
            idx = np.where(teeth)[0]
            if len(idx) >= 3:
                top[x], bot[x] = y0 + idx[0], y0 + idx[-1] + 1.0
        good = ~np.isnan(top)
        if good.sum() < 6:
            self.has_teeth = False
            return self._teeth_band()
        gx = xs[good]
        cx_pts = np.concatenate([[self.lx], gx, [self.rx]])
        t_pts = median_filter(np.concatenate([[ml], top[good], [mr]]), 5, mode="nearest")
        b_pts = median_filter(np.concatenate([[ml], bot[good], [mr]]), 5, mode="nearest")
        T = gaussian_filter1d(np.interp(xs, cx_pts, t_pts), 2.5)
        B = gaussian_filter1d(np.interp(xs, cx_pts, b_pts), 2.5)
        inside = (xs > self.lx) & (xs < self.rx)
        return np.where(inside, T, self.lipY).astype(np.float32), np.where(inside, B, self.lipY).astype(np.float32)

    def _tooth_gaps(self) -> list[float]:
        """x of the gaps between upper teeth (darker columns near the incisal edge); a layout when not visible"""
        if self.has_teeth:
            xs = np.arange(int(self.lx + 0.12 * self.mw), int(self.rx - 0.12 * self.mw))
            prof = []
            for x in xs:
                t, b = self.T[x], self.B[x]
                ys = np.arange(int(t + 0.35 * (b - t)), int(b))
                if len(ys) == 0:
                    prof.append(np.nan)
                    continue
                prof.append(float(np.mean(self.clean[ys, x] @ np.array([0.299, 0.587, 0.114], np.float32))))
            prof = np.array(prof)
            if np.isfinite(prof).sum() > 10:
                prof = np.where(np.isfinite(prof), prof, np.nanmean(prof))
                sm = gaussian_filter1d(prof, 0.8)
                base = gaussian_filter1d(prof, 4.0)
                gaps = [float(xs[i]) for i in range(1, len(sm) - 1)
                        if sm[i] < sm[i - 1] and sm[i] <= sm[i + 1] and base[i] - sm[i] > 6.0]
                if len(gaps) >= 2:
                    return gaps
        # layout in u: central incisors 0..0.2, laterals ..0.36, canines ..0.5
        return [self.cxm] + [self.cxm + s * h * b for s, h in ((-1, self.hl), (1, self.hr)) for b in (0.2, 0.36, 0.5)]


def teeth_texture(F: Face, xs, yrel, height, lower: bool):
    """Procedural tooth row: colour (H,W,3) for pixels `yrel` px below the row's top edge (lower: above its root).
    Per-tooth rounded edges, dark interproximal lines, side teeth narrower and in shadow."""
    us = np.abs(F.u(xs))
    bounds = (0.13, 0.26, 0.40, 0.55, 0.7) if lower else (0.2, 0.36, 0.5, 0.64, 0.78)
    edges = np.array((0.0,) + bounds)
    # position inside the tooth 0..1 and the tooth index
    idx = np.clip(np.searchsorted(edges, us) - 1, 0, len(edges) - 2)
    lo_e, hi_e = edges[idx], edges[idx + 1]
    pos = np.clip((us - lo_e) / np.maximum(hi_e - lo_e, 1e-3), 0, 1)
    px_w = (hi_e - lo_e) * np.where(xs < F.cxm, F.hl, F.hr)
    # interproximal line: darker within ~0.7 px of each tooth boundary
    dist_px = np.minimum(pos, 1 - pos) * px_w
    sep = np.exp(-(dist_px / (0.9 if lower else 0.7)) ** 2) * (us > 0.02)
    sep = np.maximum(sep, np.exp(-(np.abs(xs - F.cxm) / 0.6) ** 2))       # the central gap
    # rounded incisal edge: the tooth corners are a little shorter
    corner = 0.9 * (2 * pos - 1) ** 4
    # side teeth turn away from the light and sit deeper in the mouth's shadow
    shade = 1.0 - (0.22 if lower else 0.55) * np.clip(us / 0.75, 0, 1) ** (1.4 if lower else 1.1)                    # side teeth turn away from the light
    vert = np.clip(yrel / np.maximum(height, 1e-3), 0, 1)
    if lower:   # lower incisors: lit tip, darker toward the lower lip (in its shadow)
        tone = 1.0 - 0.18 * vert
    else:       # upper (synthetic): darker at the gum line, light edge
        tone = 0.80 + 0.20 * smooth(vert / 0.35)   # the upper lip's shadow on the top px
    col = (LOWER_TOOTH if lower else UPPER_TOOTH) * (shade * tone)[..., None] * (1 - (0.30 if lower else 0.22) * sep * (1 - 0.5 * np.clip(us / 0.5, 0, 1)))[..., None]
    return col, corner


DONOR = HERE / "teeth_donor.png"     # real upper-teeth band (rows from the upper lip's edge down), columns = u -1..1
DONOR_META = HERE / "teeth_donor.json"


def make_donor(rgb: np.ndarray, pts: dict, cols: int = 241) -> None:
    """Cut the visible upper-teeth band of a portrait WITH parted lips into a normalised strip: column j is
    u = -1..1 across the mouth (side-normalised, midline at the centre), row i is i px below the upper lip's edge.
    Rows past the teeth's incisal edge are transparent. A closed-lips portrait borrows these real teeth."""
    F = Face(rgb, pts)
    if not F.has_teeth:
        raise SystemExit("the donor portrait shows no teeth")
    us = np.linspace(-1, 1, cols, dtype=np.float32)
    xs = np.where(us < 0, F.cxm + us * F.hl, F.cxm + us * F.hr)
    T = np.interp(xs, np.arange(len(F.T)), F.T)
    B = np.interp(xs, np.arange(len(F.B)), F.B)
    H = int(np.ceil((B - T).max())) + 2
    yy = T[None, :] + np.arange(H, dtype=np.float32)[:, None]
    xx = np.broadcast_to(xs[None, :], yy.shape)
    strip = np.stack([map_coordinates(F.clean[..., ch], np.array([yy, xx]), order=3, mode="nearest") for ch in range(3)], -1)
    # alpha: teeth down to 1 px above B (the band's last rows are the cleaned teeth-to-lip gradient)
    a = np.clip((B - 1.2)[None, :] - yy + 0.5, 0, 1) * (np.abs(us) < 0.97)[None, :]
    Image.fromarray(np.dstack([np.clip(strip, 0, 255), a * 255]).astype(np.uint8), "RGBA").save(DONOR)
    DONOR_META.write_text(json.dumps({"mouth_width_px": round(F.mw, 2), "cols": cols, "rows": H,
                                      "note": "upper teeth of the previous Neha portrait (AI-rendered), cut by make_donor()"}),
                          encoding="utf-8")


def load_donor():
    if not DONOR.exists():
        return None
    im = np.asarray(Image.open(DONOR).convert("RGBA"), np.float32)
    meta = json.loads(DONOR_META.read_text(encoding="utf-8"))
    return {"rgb": im[..., :3], "a": im[..., 3] / 255.0, "mw": float(meta["mouth_width_px"])}


def _upper_teeth(F: Face, cav, xs, yt, us, Bx, J, r: float = 0.0):
    """Upper teeth for a closed-lips portrait: they hang from the upper lip's edge, rigid (they do not move with the
    jaw); the real donor band when available, else the procedural row. Returns (mask, colour)."""
    yrel = yt - Bx
    if F.donor is not None:
        D = F.donor
        s = D["mw"] / F.mw                                  # donor px per px of this face
        col = (np.clip(us, -1, 1) + 1) / 2 * (D["a"].shape[1] - 1)
        row = yrel * s + 0.6                                # the first donor row is in the lip's shadow
        a = map_coordinates(D["a"], np.array([row, col]), order=1, mode="constant", cval=0.0)
        rgb = np.stack([map_coordinates(D["rgb"][..., c], np.array([row, col]), order=1, mode="nearest") for c in range(3)], -1)
        mask = cav & (a > 0.5) & (yrel >= 0)
        if r > 0:   # puckered lips (O/U/CH) cover most of the teeth: a round dark opening, the incisal tips only
            mask &= yrel < (1.0 - 0.65 * r) * (D["a"].sum(axis=0).max() / s)
        # match this face's light: a touch warmer and darker toward the corners (the mouth's own shadow)
        # round 2: real teeth are not one flat white: darker toward the corners and up at the gum line (the upper lip's
        # shadow), a faint highlight a little below it, and a touch warmer / dimmer overall inside the mouth
        dy_ = np.maximum(yrel, 0.0) * s
        shade = (1.0 - 0.32 * np.clip(np.abs(us) / 0.8, 0, 1) ** 1.4
                 - 0.16 * np.exp(-(dy_ / 1.6) ** 2)
                 + 0.05 * np.exp(-((dy_ - 3.0) / 1.5) ** 2)) * 0.93
        return mask, rgb * shade[..., None] * np.array([1.0, 0.97, 0.94], np.float32)
    th = np.minimum(J * 0.66, 0.10 * F.mw) * np.clip(1 - (us / 0.85) ** 2, 0, 1) ** 0.4
    tc, corner = teeth_texture(F, xs, yrel, th, lower=False)
    return cav & (yrel < th - corner), tc


def render(F: Face, P: dict, box, ox=0.0, oy=0.0):
    """One frame of the mouth box. Returns (rgb float, layer id, source y) for the subsample offset (ox, oy)."""
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    yy += oy
    xx += ox
    m = FEATHER + 2
    win = (smooth((xx - x0 - m) / FEATHER) * smooth((x1 - 1 - m - xx) / FEATHER)
           * smooth((yy - y0 - m) / FEATHER) * smooth((y1 - 1 - m - yy) / FEATHER))
    a, k = P["a"], P["k"]
    lip_h = F.lip_bot - F.lip_top
    d = np.abs(F.u(xx))
    fall = (np.where(d <= 1.15, np.cos(np.clip(d / 1.15, 0, 1) * np.pi / 2.0), 0.0) ** 1.3).astype(np.float32)
    # 1) lips: spread / round (horizontal scale about the midline) + fuller when round, thinner when pressed.
    # The scale must reach the CORNERS (that is where a rounded or spread mouth moves most): full weight out to
    # |u| 0.9, easing to 0 in the cheek by |u| 1.9. The old cosine fall-off left the corners at ~13 % of the move,
    # so O / U were barely narrower than the rest (2026-11-25).
    fall_x = (1.0 - smooth((d - 0.9) / 1.0)).astype(np.float32)
    if k < 0:
        # rounding stays on the lips and the skin right around them (2026-11-25 round 2: the cheek outline and the
        # jaw contour were pulled in, "oo" pinched the whole lower face): zero by |u| 1.6, inside the face contour (|u| ~2.4 at the lips)
        fall_x = (1.0 - smooth((d - 0.8) / 0.8)).astype(np.float32)   # 0 by |u| 1.6 (a 0.45 ramp folded the corner shadow into ghost spots)
    # rows: the whole lip band narrows / spreads as one, INCLUDING the lower lip after the jaw has carried it down
    # (a Gaussian about the resting lip line left the dropped lower lip at ~40 % of the move: a narrow top over a wide
    # bottom, the "pouch" of the old O/U cells); then it eases out into the cheek / chin skin
    reach = np.where(yy < F.lipY, 0.45 * lip_h, 0.55 * lip_h + max(a, 0.0) * JAW * F.mw)
    vert = np.exp(-(np.maximum(np.abs(yy - F.lipY) - reach, 0.0) / ((0.35 if k < 0 else 0.7) * lip_h)) ** 2)
    # round 4: the nose never moves. With the lip grid's patch reaching up beside the nose, the lips' sideways pull and
    # fullness reached the nostrils; this guard is 0 inside the nose (above its base, within ~30 px of the midline)
    nose_guard = 1.0 - (1.0 - smooth((yy - (F.nose - 2.0)) / 7.0)) * (1.0 - smooth((np.abs(xx - F.cxm) - 30.0) / 10.0))
    hw = vert * fall_x * win * nose_guard
    # round lips pull the corners in more than spread pushes out (0.24 -> 0.34, 2026-11-25: "oo" still read as a wide
    # slot with a sagging lower lip; a real "oo" is a small rounded opening with the corners drawn toward the middle)
    sx = 1.0 + k * (SPREAD_PUSH if k > 0 else ROUND_PULL)
    xs = xx + ((xx - F.cxm) / sx - (xx - F.cxm)) * hw
    r = max(0.0, -k)
    vlip = np.exp(-((yy - F.lipY) / (1.25 * lip_h)) ** 2) * fall * win * nose_guard
    # puckered lips push forward and evert: more red shows above and below the line (fuller), thinner when pressed
    # (the upper lip only a touch: its top line and cupid's bow must stay where the photo has them)
    # (lower lip 0.30 -> 0.14, 2026-11-25: with the jaw drop on top it swelled into a pouch)
    full = np.where(yy < F.lipY, ROUND_UPPER_FULL, 0.07) * r     # round 2: lower 0.14 read puffy; round 3: the upper lip pushes forward too
    # pressed lips (M/B/P) roll in: both lips ~20 % thinner at full press (was 14 %: a closure looked like the rest face)
    yt = F.lipY + (yy - F.lipY) * (1.0 + PRESS_THIN * P["press"] * vlip) / (1.0 + full * vlip)
    if k > 0 and CORNER_UP > 0:
        # round 3 (2026-11-25, owner: "left / right of the lips"): spread vowels (E / I) lift the corners a little
        # as they pull out; the lift sits on the corners and fades into the cheek and toward the middle
        cu = k * CORNER_UP * F.mw * np.exp(-((d - 1.0) / 0.38) ** 2) * np.exp(-((yy - F.lipY) / (0.9 * lip_h)) ** 2) * win
        yt = yt + cu
    if CHEEK > 0:
        # round 4 (2026-11-25, owner: "the portion left / right of the nose does not react"): the cheeks beside the
        # nose move with the mouth as one skin. Weight: 0 within 30 px of the nose midline (the nose never moves), 0
        # before the face outline, 0 above the lower cheek under the eyes; strongest beside the nostrils down to the
        # mouth corners. E / I: cheeks up and out with the corners; O / U: in and a little down toward the lips;
        # open vowels: the lower cheek stretches down with the jaw. Content moves by (cx, cy); we sample at -(cx, cy).
        adx = np.abs(xx - F.cxm)
        wx = smooth((adx - 30.0) / 14.0) * (1.0 - smooth((adx - 92.0) / 24.0))
        wy = smooth((yy - (F.nose - 40.0)) / 22.0) * (1.0 - smooth((yy - (F.lipY + 8.0)) / 30.0))
        wc = (wx * wy * win).astype(np.float32)
        low = smooth((yy - (F.lipY - 10.0)) / 24.0)                      # lower cheek (beside / under the corners)
        sgn = np.sign(xx - F.cxm)
        kp, kr, ao = max(k, 0.0), max(-k, 0.0), max(a, 0.0)
        cy = CHEEK * (-2.5 * kp + 1.0 * kr + 2.0 * ao * low)               # px, + = down
        cx = CHEEK * (1.8 * kp - 2.0 * kr - 0.5 * ao * low)                # px, + = outward
        yt = yt - cy * wc
        xs = xs - sgn * cx * wc
    Tx = np.interp(xs, np.arange(len(F.T)), F.T)
    Bx = np.interp(xs, np.arange(len(F.B)), F.B)
    us = F.u(xs)
    sy = yt.copy()
    layer = np.zeros(yy.shape, np.int8)        # 0 photo (upper), 4 photo (lower, moved), 5 photo (upper, moved)
    J = np.zeros_like(yy)
    Btop, Jt = Bx, J
    if a > 0:
        # opening outline: a lens from corner to corner. Round lips (O/U): the lips stay together near the corners
        # and the opening is an ellipse in the middle third (umax), not a slot from corner to corner
        umax = OPEN_SPAN - ROUND_NARROW * r
        q = 0.75 - 0.25 * r
        lens = np.clip(1 - (np.abs(us) / umax) ** 2.0, 0, 1) ** q * win
        J = a * JAW * F.mw * lens
        # the upper lip lifts a little on open vowels (more on a funnelled O/U): the opening is not all jaw, and the
        # upper teeth show their full height. The skin above eases back to 0 at the nose.
        Ju = a * F.mw * (UPLIFT + ROUND_LIFT * r) * lens
        # round 4: the philtrum follows the lift more closely (exponent PHILTRUM < 1); still exactly 0 at the nose
        wu_ = smooth((yt - F.nose) / np.maximum(4.0, Bx - Ju - F.nose)) ** PHILTRUM
        upper = yt < Bx - Ju
        sy = np.where(upper, yt + Ju * wu_, sy)
        layer[upper & (Ju * wu_ > 0.3)] = 5
        Btop, Jt = Bx - Ju, J + Ju
        # below the lower lip the jaw is wider than the mouth: the drop spreads sideways toward the chin so the
        # jaw line moves as one piece instead of shearing at the mouth corners
        span = 1.0 + 1.3 * smooth((yt - F.lip_bot) / max(4.0, F.chin - F.lip_bot))
        Jy = a * JAW * F.mw * np.clip(1 - (us / span) ** 2, 0, 1) ** 0.75 * win
        # at the lip line the drop is the OPENING's (J: narrow for O/U, the lips stay together near the corners);
        # under the lower lip it becomes the jaw's (wide, the whole chin moves as one piece)
        Jy = J + (Jy - J) * smooth((yt - (F.lo + 2.0)) / max(4.0, F.lip_bot + 6.0 - F.lo))
        # the chin follows a third of the drop and the skin under it eases out over ~50 px: the jaw / neck contour
        # bends smoothly instead of kinking where the moving chin meets the still neck
        # (round 2, 2026-11-25: the chin followed a third and read as fixed; now half, so the jaw visibly drops)
        mid_w = 1 - (1 - CHIN_FOLLOW) * smooth((yt - (F.lip_bot + 4)) / max(4.0, F.chin - F.lip_bot - 4))
        wl = np.where(yt < F.lip_bot + 4, 1.0, np.where(yt < F.chin, mid_w, CHIN_FOLLOW * smooth((F.chin + 50 - yt) / 50.0)))
        below = yt >= Bx + J
        rim = 1.8 * np.clip(J / 2, 0, 1) * np.exp(-((yt - (Bx + J)) / 7.0) ** 2)   # skip the grey teeth/lip rows
        sy = np.where(below, yt - Jy * wl + rim, sy)
        layer[below] = 4
        cav = (yt >= Btop) & ~below
        layer[cav] = 1
    elif a < 0:
        c = -a
        gap = np.maximum(Bx - Tx, 0)
        Dm = c * (gap / 2.0 + 1.0) * win
        midl = (Tx + Bx) / 2.0
        wu = smooth((yt - F.nose) / max(4.0, F.up - F.nose))
        wl2 = np.where(yt < F.chin - 30, 1.0, smooth((F.chin - yt) / 30.0))
        upper_out = yt < Tx + Dm
        lower_out = yt >= Bx - Dm
        if c >= 0.999:
            upper_out = yt < midl
            lower_out = ~upper_out
        else:
            both = upper_out & lower_out
            upper_out = upper_out & ~(both & (yt >= midl))
            lower_out = lower_out & ~(both & (yt < midl))
        rim_u = 1.6 * smooth(c / 0.3) * np.exp(-((yt - (Tx + Dm)) / 6.0) ** 2)
        sy = np.where(upper_out, yt - Dm * np.where(yt >= Tx, 1.0, wu) - rim_u, sy)
        rim_c = 1.8 * smooth(c / 0.3) * np.exp(-((yt - (Bx - Dm)) / 7.0) ** 2)
        sy = np.where(lower_out, yt + Dm * np.where(yt <= Bx, 1.0, wl2) + rim_c, sy)
        layer[upper_out & (Dm > 0.05)] = 5
        layer[lower_out & (Dm > 0.05)] = 4
        teeth_rows = ~upper_out & ~lower_out
        remain = np.maximum(gap - 2 * Dm, 0)
        shade_close = np.where(teeth_rows, 1.0 - 0.55 * (1 - smooth(remain / 10.0)) * smooth(gap / 5.0), 1.0)
    if P["tuck"] > 0:
        # F/V: the lower lip rises behind the upper incisors (its top edge hides behind them), a little flatter
        lt_lens = np.clip(1 - (us / 1.05) ** 2, 0, 1) ** 0.5 * win
        Dt = P["tuck"] * (0.42 * np.maximum(Bx - Tx, 0) + 1.5) * lt_lens
        wl3 = np.where(yt < F.lip_bot + 2, 1.0, smooth((F.chin - 10 - yt) / max(4.0, F.chin - 10 - F.lip_bot - 2)))
        low = yt >= Bx
        sy = np.where(low, yt + Dt * wl3, sy)
        layer[low & (Dt > 0.05)] = 4
    out = np.stack([map_coordinates(F.clean[..., ch], np.array([sy, xs]), order=3, mode="nearest") for ch in range(3)], -1)
    if a < 0:
        dk = (1 - shade_close)[..., None]
        out = out * (1 - dk * np.array([0.85, 1.0, 1.0], np.float32))
        # the last few px of the gap: the inside in shadow (warm, dark), not a grey strip of teeth (the "seam")
        g = (0.9 * (1 - smooth(remain / 9.0)) * teeth_rows * np.clip(1 - np.abs(us), 0, 1) ** 0.3)[..., None]
        out = out * (1 - g) + MOUTH_GAP * g
        if -a > 0.6:   # lips pressed together: a soft line where they meet, strongest in the middle
            c = -a
            midl = (Tx + Bx) / 2.0
            across = np.clip(1 - np.abs(us), 0, 1) ** 0.6
            seam = np.exp(-((yt - midl) / (0.9 + 0.3 * P["press"])) ** 2) * across * smooth((c - 0.6) / 0.4) * win
            out = out * (1 - (0.30 + 0.08 * P["press"]) * seam[..., None])
    if P["tuck"] > 0:
        # contact shadow: the teeth edge resting on the lower lip
        sh = np.exp(-((yt - Bx - 0.6) / 1.1) ** 2) * (yt >= Bx - 0.5) * np.clip(1 - np.abs(us), 0, 1) ** 0.5 * win * P["tuck"]
        out = out * (1 - 0.28 * sh[..., None])
    if a > 0:
        out = _inside(F, P, out, layer, yt, xs, us, Btop, Jt)
        if r > 0:
            # puckered lips: the lips protrude, so their middle catches the light and the pulled-in corners sit
            # deeper in shadow; the cheek skin next to them stays as it is
            lipband = np.exp(-((yt - F.lipY) / (0.75 * lip_h)) ** 2) * (layer != 1) * (layer != 2) * (layer != 3) * (layer != 6)
            uo = np.abs(F.u(xx))
            corner_sh = smooth((uo - 0.35) / 0.45) * (1 - smooth((uo - 0.85) / 0.35))
            mid_lit = np.exp(-(uo / 0.35) ** 2)
            g = (1 - 0.12 * r * corner_sh + 0.04 * r * mid_lit) * lipband * win + (1 - lipband * win)
            out = out * g[..., None]
    return out, layer, sy


def _inside(F: Face, P: dict, out, layer, yt, xs, us, Bx, J):
    """Paint the open mouth: cavity -> tongue -> lower teeth (-> tongue between the teeth), plus upper incisal
    notches and the lower-lip contact shadow. Works on one (sub)sample grid; antialiasing happens across grids."""
    cav = layer == 1
    t = np.clip((yt - Bx) / np.maximum(J, 1e-3), 0, 1)
    thin = np.clip((J - 1.5) / 5.0, 0, 1)[..., None]
    au = np.abs(us)
    # depth (2026-11-25: "dark and flat"): the back of the mouth is darkest just under the upper teeth and in the
    # corners (the cheeks' inner walls turn away from the light); the floor of the mouth is warmer and a little
    # lit; the lower lip's wet inner surface shows as a band of deep lip colour along the bottom of the opening.
    col = CAV_TOP + (CAV_DEEP - CAV_TOP) * smooth((t - 0.15) / 0.85)[..., None]
    jopen = (Bx + J)                                   # lower edge of the opening (top of the lower lip)
    walls = 1.0 - 0.50 * smooth((au - 0.38) / 0.55)    # buccal corridors
    col = col * walls[..., None]
    # a faint warm glow at the back in the middle (soft palate / back of the tongue): reads as depth, not a hole
    glow = np.exp(-((t - 0.62) / 0.28) ** 2) * np.exp(-(au / 0.42) ** 2) * np.clip((J - 8.0) / 10.0, 0, 1)
    col = col + (CAV_BACK - col) * (0.38 * glow)[..., None]
    yb = jopen - yt                                    # px above the lower lip
    inner_h = 0.9 + 0.09 * J
    inner = np.exp(-(np.maximum(yb, 0) / np.maximum(inner_h, 0.5)) ** 2) * np.clip((J - 5.0) / 6.0, 0, 1) * (1 - smooth((au - 0.55) / 0.4))
    col = col * (1 - 0.75 * inner[..., None]) + INNER_LIP * walls[..., None] * 0.75 * inner[..., None]
    col = LIP_SHADOW * (1 - thin) + col * thin
    paint = np.where(cav[..., None], col, out)
    lid = layer.copy()
    # synthetic upper teeth (portrait with closed lips): hang from the upper lip into the cavity
    up_t = None
    if not F.has_teeth:
        up_t, up_col = _upper_teeth(F, cav, xs, yt, us, Bx, J, max(0.0, -P["k"]))
    tongue_kind = P["tongue"]
    tg = P["tg"]
    if tongue_kind and tg > 0:
        # tongue: a rounded mound lying behind the lower teeth, narrower than the opening. Its top surface is lit
        # from the front (a soft highlight band under the top edge), it rolls away into shadow at its sides and at
        # the back, with a faint central groove; the front, where it dips behind the lower lip, is in that lip's shadow.
        h = np.clip((0.30 if tongue_kind != "low" else 0.36) * tg, 0, 0.55)
        tw = 0.72
        across = np.clip(1 - (us / tw) ** 2, 0, 1)
        y_top = Bx + J * (1 - h * across ** 0.6)
        body = cav & (yt >= y_top) & (J > 3.0) & (au < tw)
        if tongue_kind == "tip":   # tip raised behind the upper incisors
            e = ((xs - F.cxm) / (0.22 * np.where(xs < F.cxm, F.hl, F.hr))) ** 2 + ((yt - (Bx + 0.40 * J)) / (0.60 * J + 1e-3)) ** 2
            body |= cav & (e <= 1.0) & (J > 3.0)
            tipdark = (e <= 1.0) & (yt < y_top)
        depth_t = np.clip((yt - y_top) / np.maximum(jopen - y_top, 1.0), 0, 1)       # 0 top surface .. 1 at the lip
        lit = np.exp(-((yt - y_top - 1.2 - 0.06 * J) / (1.0 + 0.05 * J)) ** 2) * np.clip(1 - au / 0.55, 0, 1) ** 0.8
        roll = 0.62 + 0.38 * across ** 0.5                                            # sides roll into shadow
        groove = 1 - 0.10 * np.exp(-((xs - F.cxm) / 1.1) ** 2) * (1 - depth_t)
        front = 1 - 0.30 * smooth((depth_t - 0.45) / 0.55)                            # under the lower lip's shadow
        tcol = TONGUE * (roll * groove * front)[..., None] + (TONGUE_LIT - TONGUE) * (lit * 0.9)[..., None]
        if tongue_kind == "tip":   # the raised tip is in the teeth's shadow: darker toward the top
            tcol = np.where(tipdark[..., None], tcol * (0.62 + 0.3 * t[..., None]), tcol)
        # the tongue's edge against the cavity is soft (1 px), not cut out
        paint = np.where(body[..., None], tcol * walls[..., None] ** 0.5, paint)
        lid[body] = 2
        # the lower lip's inner surface is in front of the tongue too
        ib = (body * inner)[..., None]
        paint = paint * (1 - 0.75 * ib) + INNER_LIP * 0.75 * ib
    # lower incisors behind the lower lip (spread / open vowels), or filling the gap (teeth together: S, CH)
    # (a 1-2 px sliver of white reads as a seam, never as teeth: they show from ~3 px, on openings >= 8 px)
    open_lt = P["lt"] * np.clip(0.36 * J, 0, 6.0) * (J >= 9.0)
    bite_lt = P["bite"] * np.maximum(J - 1.0, 0)
    # open vowels: only the central lower incisors' tips show (|u| < 0.5), the side teeth stay under the lip
    lt_h = np.maximum(open_lt * np.clip(1 - (us / 0.5) ** 2, 0, 1) ** 0.5, bite_lt * np.clip(1 - (us / 0.72) ** 2, 0, 1) ** 0.35)
    yrel = (Bx + J) - yt                    # px above the lower lip
    tc, corner = teeth_texture(F, xs, lt_h - yrel, lt_h, lower=True)
    if P["bite"] <= 0:
        # behind the lower lip and under the upper teeth: in the mouth's shade, warm, never a grey-white strip
        tc = tc * np.array([0.96, 0.87, 0.82], np.float32) * (0.85 + 0.15 * np.clip(1 - np.abs(us) / 0.5, 0, 1))[..., None]
    low_t = cav & (yrel <= lt_h - corner * 0.6) & (lt_h > 2.2)
    paint = np.where(low_t[..., None], tc, paint)
    lid[low_t] = 3
    if up_t is not None:
        paint = np.where(up_t[..., None], up_col, paint)
        lid[up_t] = 6
    if tongue_kind == "between" and tg > 0:
        e = ((xs - F.cxm) / (0.26 * np.where(xs < F.cxm, F.hl, F.hr))) ** 2 + ((yt - (Bx + 0.6 * J)) / (0.6 * J + 1e-3)) ** 2
        tip = cav & (e <= 1.0) & (J > 2.5)
        if up_t is not None:
            tip &= ~up_t          # behind the (closed-lips portrait's) upper teeth, only under their edge
        lit = np.clip(1 - e, 0, 1)[..., None] ** 0.5
        paint = np.where(tip[..., None], TONGUE * (0.72 + 0.3 * lit), paint)
        lid[tip] = 2
    # notches between the upper incisal edges, visible against the dark cavity
    if F.has_teeth:
        notch = np.zeros_like(yt)
        for g in F.gaps:
            notch = np.maximum(notch, np.exp(-((xs - g) / 0.75) ** 2) * smooth((yt - (Bx - 2.2)) / 2.2))
        notch *= (yt < Bx) & (yt >= Bx - 2.5)
        notch *= np.clip(J / 3.0, 0, 1)
        paint = paint * (1 - 0.45 * notch[..., None]) + CAV_TOP * 0.45 * notch[..., None]
        # the incisal edge is a little translucent (greyer) on wide openings
        edge = np.exp(-((yt - (Bx - 0.8)) / 0.8) ** 2) * (yt < Bx) * np.clip((J - 4) / 8, 0, 1)
        paint = paint * (1 - 0.10 * edge[..., None])
    # inner lower lip: contact shadow where the lip turns into the mouth
    ls = np.exp(-((yt - (Bx + J) - 0.6) / 1.0) ** 2) * (layer == 4) * np.clip(J / 3.0, 0, 1)
    paint = paint * (1 - 0.22 * ls[..., None])
    layer[:] = lid
    return paint


def render_aa(F: Face, P: dict, box):
    """Exact sample where nothing is discontinuous; mean of 3x3 subsamples on layer edges and warp seams."""
    c0, l0, s0 = render(F, P, box)
    offs = [((i + 0.5) / SS - 0.5, (j + 0.5) / SS - 0.5) for i in range(SS) for j in range(SS)]
    acc = np.zeros_like(c0)
    cov = np.zeros(l0.shape, np.float32)
    edge = np.zeros(l0.shape, bool)
    for ox, oy in offs:
        c, lyr, s = render(F, P, box, ox, oy)
        acc += c
        cov += np.isin(lyr, (1, 2, 3, 6))
        edge |= (lyr != l0) | (np.abs(s - s0) > 1.2)
    mean = acc / len(offs)
    out = np.where(edge[..., None], mean, c0)
    painted_cov = cov / len(offs)
    # grain on painted pixels so they carry the photo's noise (fixed seed: rebuilds are identical)
    painted = np.isin(l0, (1, 2, 3, 6))
    if painted.any():
        rng = np.random.default_rng(7)
        grain = gaussian_filter(rng.normal(0, 2.2, l0.shape), 0.6)
        out = out + (grain * painted)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), painted_cov


def finish_cell(cell: np.ndarray, cov: np.ndarray) -> Image.Image:
    """The shared unsharp mask (same as base.png) on the photo pixels only: on the painted inside (and 1-2 px around
    it) it drew a bright halo on the lip edge and a zipper on the lower teeth."""
    sharp = np.asarray(gm.sharpen(Image.fromarray(cell)), np.float32)
    keep = np.clip(gaussian_filter(np.minimum(1.0, cov * 3.0), 1.0) * 2.0, 0, 1)[..., None]
    return Image.fromarray(np.clip(sharp * (1 - keep) + cell * keep, 0, 255).astype(np.uint8))


def mouth_box(F: Face) -> list[int]:
    """the patch: wide enough that the jaw drop fades out well inside the feather (no kink in the jaw line)"""
    return [int(F.cxm - F.mw * 1.2), int(F.nose - 14), int(F.cxm + F.mw * 1.2), int(F.chin + 72)]


def symmetry(F: Face, P: dict, box) -> dict:
    """Opening / closure symmetry of a frame in the face's own frame: centre of the cavity in u (0 = on the lips'
    axis) and left/right extent in u (equal when symmetric)."""
    _, lyr, _ = render(F, P, box)
    x0, y0 = box[0], box[1]
    ys, xs = np.nonzero(np.isin(lyr, (1, 2, 3, 6)))
    if len(xs) < 4:
        return {}
    uu = F.u(xs.astype(np.float32) + x0)
    return {"u_left": round(float(uu.min()), 3), "u_right": round(float(uu.max()), 3),
            "u_centre": round(float((uu.min() + uu.max()) / 2), 3)}


def load_portrait(args) -> tuple[np.ndarray, dict, Path, bool]:
    """(rgb, points, source path, is_default). --photo: framed + landmarked into avatar_v2/portrait first."""
    default_dir = TM / "prepared"
    if args.photo:
        pdir = HERE / "portrait"
        pdir.mkdir(exist_ok=True)
        cmd = [sys.executable, str(TM / "prepare_portrait.py"), "--photo", str(args.photo), "--out", str(pdir)]
        subprocess.run(cmd, check=True)
        src = pdir / "face.jpg"
        face = json.loads((pdir / "face.json").read_text(encoding="utf-8"))
        return np.asarray(Image.open(src).convert("RGB"), np.uint8), face["points"], src, False
    src = default_dir / "face.jpg"
    face = json.loads((default_dir / "face.json").read_text(encoding="utf-8"))
    return np.asarray(Image.open(src).convert("RGB"), np.uint8), face["points"], src, True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--photo", type=Path, help="any front-facing portrait (closed or parted lips); default: prepared/face.jpg")
    ap.add_argument("--out", type=Path, default=HERE / "pack")
    ap.add_argument("--only", help="comma-separated visemes (preview)")
    ap.add_argument("--make-donor", type=Path, metavar="DIR",
                    help="cut teeth_donor.png from DIR/face.jpg + face.json (a portrait with visible upper teeth) and stop")
    a = ap.parse_args()
    if a.make_donor:
        make_donor(np.asarray(Image.open(a.make_donor / "face.jpg").convert("RGB"), np.uint8),
                   json.loads((a.make_donor / "face.json").read_text(encoding="utf-8"))["points"])
        print("wrote", DONOR)
        return 0
    rgb, pts, src, is_default = load_portrait(a)
    F = Face(rgb, pts)
    if not F.has_teeth:
        # closed lips in the photo: F/V needs the lips just parted so the upper teeth show resting on the lower lip
        VISEMES["FF"] = dict(a=0.15, k=0.1)
        global REST_A
        REST_A = 0.0
    box = mouth_box(F)
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    names: list[tuple[str, str, float]] = []
    for v in VISEMES:
        if a.only and v not in a.only.split(","):
            continue
        for lv in ((1.0,) if v in SINGLE else LEVELS):
            names.append((v if v in SINGLE else f"{v}_{int(round(lv * 100))}", v, lv))
    cols = 8
    rows = (len(names) + cols - 1) // cols
    atlas = Image.new("RGB", (bw * cols, bh * rows))
    out_dir = a.out
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    frames, sym, visemes, frame_a, frame_k = {}, {}, {}, {}, {}
    for i, (name, v, lv) in enumerate(names):
        P = params_at(v, lv)
        im = finish_cell(*render_aa(F, P, box))
        atlas.paste(im, ((i % cols) * bw, (i // cols) * bh))
        frames[name] = i
        frame_a[name] = round(float(P["a"]), 3)
        frame_k[name] = round(float(P["k"]), 3)
        visemes.setdefault(v, {"levels": [], "frames": [], "params": {k: VISEMES[v][k] for k in VISEMES[v]}})
        visemes[v]["levels"].append(lv)
        visemes[v]["frames"].append(name)
        sym[name] = symmetry(F, P, box) if P["a"] > 0 else {}
        if lv >= 0.999:
            im.save(out_dir / "frames" / f"{v}.png")
        print(f"{name:7s} a={P['a']:+.2f} k={P['k']:+.2f} {sym[name]}")
    atlas.save(out_dir / "atlas.jpg", quality=93, subsampling=0)
    # base the card draws under the mouth: the default portrait keeps path_b/mouths/base.png (eye/brow layers are
    # aligned to it); any other portrait ships its own sharpened base + landmarks with the pack
    if not is_default:
        gm.sharpen(Image.fromarray(rgb)).save(out_dir / "base.png")
        shutil.copyfile(src.with_name("face.json"), out_dir / "face.json")
    digest = hashlib.sha1((out_dir / "atlas.jpg").read_bytes()).hexdigest()[:12]
    manifest = {
        "version": digest,
        "canvas": [int(rgb.shape[1]), int(rgb.shape[0])],
        "box": [x0, y0, bw, bh],
        "feather": FEATHER,
        "atlas": "atlas.jpg",
        "cols": cols,
        "frames": frames,
        "frame_a": frame_a,     # aperture of every frame (-1 pressed .. 1 jaw fully open): QA / sync measurement
        "frame_k": frame_k,     # lip width of every frame (-1 round .. 1 spread): with frame_a, the card's in-between
                                # cells from path_b/mouths/grid2.jpg (same renderer, same box) instead of a crossfade
        "visemes": visemes,
        "base": None if is_default else "base.png",
        "midline_x": round(F.cxm, 1),
        "source": {"portrait": str(src.relative_to(TM.parent)).replace("\\", "/"),
                   "sha1": hashlib.sha1(src.read_bytes()).hexdigest()[:12], "teeth_in_photo": bool(F.has_teeth)},
        "symmetry": sym,
        "runtime": {"lead_s": 0.08, "crossfade_ms": 50, "hold_ms": 55, "inbetween": "grid"},
        "note": "build_reference_pack.py; viseme names match human_bot/llm/visemes_v2.py",
    }
    (out_dir / "pack.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    _sheet(atlas, names, bw, bh, cols, out_dir / "sheet.png")
    print("wrote", out_dir, atlas.size, "box", manifest["box"], "frames", len(names))
    return 0


def _sheet(atlas: Image.Image, names, bw, bh, cols, path: Path) -> None:
    sheet = atlas.copy()
    dr = ImageDraw.Draw(sheet)
    for i, (name, _, _) in enumerate(names):
        x, y = (i % cols) * bw, (i // cols) * bh
        dr.rectangle([x, y, x + 52, y + 13], fill=(0, 0, 0))
        dr.text((x + 3, y + 1), name, fill=(255, 255, 255))
    sheet.save(path)


if __name__ == "__main__":
    raise SystemExit(main())
