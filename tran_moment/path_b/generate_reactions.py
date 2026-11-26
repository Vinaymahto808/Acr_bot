"""Reaction frames for the puppet from prepared/face.jpg (CPU, numpy + PIL + scipy; no model, no GPU).

The mouth shapes (generate_mouths.py) cover speech. This adds the *other* moments a face has in a
conversation, each as an aligned PNG layer over the same 640x852 canvas (transparent outside its region):

  eyes   eyes_closed, eyes_half           blink frames with the real upper lid stretched over the eye
         gaze_left, gaze_right, gaze_up   the iris/sclera shifted inside the opening (listening / thinking)
  brows  brows_up                         raised (question, interest, "acha!")
         brows_in                         inner ends down and in (concern, "hmm, dikkat hai")
  mouth  smile.png (into mouths/)         corners lifted and widened, lips closed: the warm resting face
  poses  head_tilt_l, head_tilt_r, head_down, head_up, lean_in   whole-frame body/head variants (for
         clip-style crossfades and review; the live card does these with CSS transforms)

Writes path_b/reactions/*.png, path_b/reactions.json and path_b/reactions_sheet.png (contact sheet).

    python tran_moment/path_b/generate_reactions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_mouths import _alpha as mouth_alpha, _smooth  # noqa: E402

PREP = HERE.parent / "prepared"
OUT = HERE / "reactions"
MOUTHS = HERE / "mouths"


def warp(rgb: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Output pixel (y, x) samples source (y - dy, x - dx): positive dy moves content DOWN."""
    H, W = rgb.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    src = np.array([yy - dy, xx - dx], dtype=np.float32)
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        out[..., c] = map_coordinates(rgb[..., c].astype(np.float32), src, order=3, mode="nearest")   # bicubic: bilinear softened every speaking shape (2026-11-13)
    return out


def box_alpha(shape: tuple[int, int], x0: float, x1: float, y0: float, y1: float, fx: float, fy: float) -> np.ndarray:
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ax = _smooth((xx - x0) / fx) * _smooth((x1 - xx) / fx)
    ay = _smooth((yy - y0) / fy) * _smooth((y1 - yy) / fy)
    return np.clip(ax * ay, 0, 1).astype(np.float32)


def save_layer(rgb: np.ndarray, alpha01: np.ndarray, path: Path) -> None:
    a = (np.clip(alpha01, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    im.putalpha(Image.fromarray(a, "L"))
    from generate_mouths import sharpen  # noqa: PLC0415
    sharpen(im).save(path)   # same as the mouth shapes: no sharpness jump between layers


# ---------------------------------------------------------------- eyes
def eye_geom(p: dict, side: str) -> dict:
    o, i, t, b = p[f"eye_{side}_outer"], p[f"eye_{side}_inner"], p[f"eye_{side}_top"], p[f"eye_{side}_bottom"]
    x0, x1 = min(o[0], i[0]), max(o[0], i[0])
    return {"cx": (x0 + x1) / 2, "w": x1 - x0, "top": float(t[1]), "bottom": float(b[1]), "x0": x0, "x1": x1}


# ---------------------------------------------------------------- eyes v2 (2026-11-25): a lid model, not a warp
# The old blink stretched the lid band (lashes and liner included) down over the eye: the lashes doubled and a white
# line of sclera stayed under the lid (the warp faded to 0 at the corners). The old gaze sheared the iris into an
# egg. Now, per eye, from MediaPipe's dense eye contour:
#   blink  the lid skin unfolds from under the crease down to the lid margin M (half, low, closed): a lashless lid
#          colour (per-column median of the lid band, smoothed) with dome shading; the liner and the photo's own
#          lash strokes travel with the margin (up while it is half down, flipped down and foreshortened when
#          closed); the eyeball under a half-lowered lid stays where it is, in the lid's shadow.
#   gaze   the iris + pupil move as one rigid disc inside the opening; the sclera behind is re-laid row by row.
EYE_IDX = {"r": {"up": [33, 246, 161, 160, 159, 158, 157, 173, 133], "lo": [33, 7, 163, 144, 145, 153, 154, 155, 133], "iris": 468},
           "l": {"up": [263, 466, 388, 387, 386, 385, 384, 398, 362], "lo": [263, 249, 390, 373, 374, 380, 381, 382, 362], "iris": 473}}
EYES_JSON = HERE / "eyes.json"
LID_MODE = "unroll"   # "unroll": v4 lid from her own lid band (default); "v3": the toned fill
MODEL = HERE.parent / "motion" / "models" / "face_landmarker.task"


def eye_contours(img: Image.Image) -> dict:
    """{side: {up: [[x,y]..], lo: [..], iris: [x,y], rad}} from MediaPipe (on a 2x copy for sub-pixel points), cached
    in path_b/eyes.json together with the portrait's hash (generate_reactions.py runs without mediapipe too)."""
    import hashlib  # noqa: PLC0415
    digest = hashlib.sha1(img.tobytes()).hexdigest()[:12]
    if EYES_JSON.exists():
        cached = json.loads(EYES_JSON.read_text(encoding="utf-8"))
        if cached.get("portrait") == digest:
            return cached["eyes"]
    import mediapipe as mp  # noqa: PLC0415  (only when the portrait changed)
    from mediapipe.tasks import python as mpp  # noqa: PLC0415
    from mediapipe.tasks.python import vision  # noqa: PLC0415
    W, H = img.size
    big = np.asarray(img.resize((W * 2, H * 2), Image.LANCZOS))
    det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(MODEL)), num_faces=1))
    lm = [[q.x * W, q.y * H] for q in det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=big)).face_landmarks[0]]
    eyes = {}
    for s, e in EYE_IDX.items():
        c = lm[e["iris"]]
        rad = float(np.mean([np.hypot(lm[e["iris"] + k][0] - c[0], lm[e["iris"] + k][1] - c[1]) for k in range(1, 5)]))
        eyes[s] = {"up": [[round(lm[i][0], 2), round(lm[i][1], 2)] for i in e["up"]],
                   "lo": [[round(lm[i][0], 2), round(lm[i][1], 2)] for i in e["lo"]],
                   "iris": [round(c[0], 2), round(c[1], 2)], "rad": round(rad, 2)}
    EYES_JSON.write_text(json.dumps({"portrait": digest, "eyes": eyes, "note": "MediaPipe eye contours + iris (generate_reactions.py)"}, indent=1), encoding="utf-8")
    return eyes


def eye_model(e: dict, W: int) -> dict:
    xs = np.arange(W, dtype=np.float32)

    def curve(pts):
        p = np.array(sorted(pts))
        return np.interp(xs, p[:, 0], p[:, 1]).astype(np.float32)
    up = np.array(e["up"])
    xa, xb = float(up[:, 0].min()), float(up[:, 0].max())
    cx, hw = (xa + xb) / 2, (xb - xa) / 2
    outer = e["up"][0][0]                                    # the first contour point is the outer corner
    return {"xs": xs, "U": curve(e["up"]), "L": curve(e["lo"]), "xa": xa, "xb": xb, "cx": cx, "hw": hw,
            "u": np.clip((xs - cx) / hw, -1, 1), "inside": ((xs > xa) & (xs < xb)).astype(np.float32),
            "iris": e["iris"], "rad": e["rad"], "outer_dir": outer - cx}


def _lum(a):
    return a @ np.array([0.299, 0.587, 0.114], np.float32)


def _sm(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def lid_frame(out: np.ndarray, src: np.ndarray, g: dict, amt: float, brow_y: float) -> np.ndarray:
    """One eye with its upper lid `amt` of the way down (0 open .. 1 closed). `src` is the untouched portrait.
    v3 (2026-11-25 review: v2's lid was a flat grey-mauve fill with a dark outline, "painted"): the lid's texture is
    her real skin just above the crease (mirror-tiled at 1:1, dark high frequencies clamped so no line is copied),
    its colour her warm skin tone there with soft crease / dome / lash-line shading; the fill starts above the
    photo's crease line and is feathered 4-5 px into the skin; thin dark lines around the eye (the crease running
    to the temple, stray lash tips) are softened. The only dark line left is the lash line with its lashes."""
    from scipy.ndimage import gaussian_filter, gaussian_filter1d, grey_closing  # noqa: PLC0415
    H, W = src.shape[:2]
    xs, U, L, u = g["xs"], g["U"], g["L"], g["u"]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ext = 8.0
    xa, xb = g["xa"] - ext, g["xb"] + ext
    ue = np.clip((xs - g["cx"]) / (g["hw"] + ext), -1, 1)
    dome = np.sqrt(np.clip(1 - ue ** 2, 0, 1))
    lens = np.clip(1 - np.abs(u) ** 2.2, 0, 1) * g["inside"]
    C = L - 0.6 * lens
    M = U + amt * (C - U)
    crease = np.minimum(U, L) - (1.5 + 8.5 * dome)
    top = crease - 2.5 * dome - 0.5                  # the fill starts above the photo's crease line (it is covered)
    # skin source: the band just above `top`, below the brow, mirrored (ping-pong) at 1:1 scale -> real texture
    Hs = np.clip(0.75 * (top - (brow_y + 7.0)), 4.0, 10.0)
    Mx, Tx, Ux, Lx, Hsx = (np.interp(xx, xs, a) for a in (M, top, U, L, Hs))
    d = np.maximum(yy - Tx, 0)
    k = np.mod(d, 2 * Hsx)
    pp = np.where(k < Hsx, k, 2 * Hsx - k)
    sy = Tx - 1.0 - pp
    skin = np.stack([map_coordinates(src[..., c], [sy, xx], order=1, mode="nearest") for c in range(3)], -1)
    # colour: keep the texture (high frequencies), set the low frequencies to her warm skin tone + lid shading
    low = gaussian_filter(skin, (2.5, 2.5, 0))
    # warm tone per column: the skin above the crease, robust median, smoothed sideways
    col = np.zeros((W, 3), np.float32)
    for x in range(int(xa) - 2, int(xb) + 3):
        y1 = int(top[x]) - 1; y0 = int(top[x] - Hs[x]) - 1
        col[x] = np.median(src[y0:y1 + 1, x], axis=0)
    sel = slice(int(xa) - 2, int(xb) + 3)
    col[sel] = gaussian_filter1d(col[sel], 3.0, axis=0, mode="nearest")
    colx = np.stack([np.interp(xx, xs, col[:, c]) for c in range(3)], -1)
    t = np.clip((yy - Tx) / np.maximum(Mx - Tx, 1e-3), 0, 1)
    ax = np.abs(np.clip((xx - g["cx"]) / (g["hw"] + ext), -1, 1))
    # soft crease shadow at the top, a light dome across the middle, the lid turning away into the lash line
    shade = (1 - 0.07 * np.exp(-((yy - (Tx + 1.5)) / 2.2) ** 2) * (1 - ax ** 2)
             + 0.035 * np.exp(-((t - 0.5) / 0.28) ** 2) * (1 - ax)
             - 0.10 * _sm((t - 0.65) / 0.35) - 0.05 * ax ** 2)
    target = colx * (0.96 * shade)[..., None]          # a closed lid sits a little under the brow-bone highlight
    tex = np.clip(skin - low, -5.0, 4.0)            # pores and grain only: no dark line can be copied in
    lid = target + tex
    if LID_MODE == "unroll":
        # v4 (2026-11-25 round 2: the v3 lid read as a flat beige oval with an outline). The lid's colour is no longer
        # one median tone per column: it is her own skin just above the crease (lash-free, below the brow), reflected
        # down ONCE and scaled to the lid height, as the lid skin continues the skin above it when the eye closes. Its
        # own light gradient comes with it; the dome catches a little more light, the lash line and the corners sit
        # in shade. (Unrolling the visible lid band itself was tried: it is 4-6 px of lash roots -> grey streaks.)
        Hm = np.maximum(Hsx, 4.0)
        sy_l = Tx - 1.0 - np.clip((yy - Tx) / np.maximum(Mx - Tx, 1e-3), 0, 1) * (Hm - 1.0)
        refl = np.stack([map_coordinates(src[..., c], [sy_l, xx], order=1, mode="nearest") for c in range(3)], -1)
        base_l = gaussian_filter(refl, (1.6, 2.2, 0))              # its low frequencies: the real tone + gradient
        shade2 = (1 + 0.05 * np.exp(-((t - 0.42) / 0.30) ** 2) * (1 - ax ** 2)    # dome highlight
                  - 0.12 * _sm((t - 0.62) / 0.38)                                  # turning into the lash line
                  - 0.06 * ax ** 2)                                                # corners in the eye socket's shade
        # at the outer end the brow tail dips into the reflected band: a lid is never darker than her skin there
        lum_b, lum_c = _lum(base_l), _lum(colx)
        k_dk = np.clip((0.95 * lum_c - lum_b) / 12.0, 0, 1)[..., None]
        base_l = base_l * (1 - k_dk) + colx * 0.95 * k_dk
        lid = base_l * (0.97 * shade2)[..., None] + 0.6 * tex
    # the lid band's own make-up tint near the lash line (her mauve shadow), faint
    side = _sm((xx - xa) / 5.0) * _sm((xb - xx) / 5.0)
    a_top = _sm((yy - (Tx - 6.0)) / 6.0) if LID_MODE == "unroll" else _sm((yy - (Tx - 4.0)) / 4.0)   # v4: softer top edge, no outline
    a = (a_top * np.clip(Mx + 0.5 - yy, 0, 1) * side * ((xx > xa) & (xx < xb))).astype(np.float32)
    # the photo's crease line and stray lash tips above the eye (incl. where the crease runs past the corners
    # toward the temple): thin dark lines -> the skin around them, so no outline is left around the closed eye
    zx0, zx1 = int(xa) - 10, int(xb) + 11
    zy0 = int(brow_y + 7); zy1 = int(np.max(U[int(g["xa"]):int(g["xb"])])) + 1
    sub = src[zy0:zy1, zx0:zx1]
    cl = gaussian_filter(np.stack([grey_closing(sub[..., c], size=(5, 5)) for c in range(3)], -1), (0.8, 0.8, 0))
    dk = np.clip((_lum(cl) - _lum(sub) - 5.0) / 18.0, 0, 1)
    yl, xl = np.mgrid[zy0:zy1, zx0:zx1].astype(np.float32)
    corner_y = np.interp(xl, xs, U)
    # keep the lash line + flick: nothing within 3.5 px above the upper lid contour / its extension past the corners
    outer = np.sign(g["outer_dir"])
    ext_y = np.where((xl - g["cx"]) * outer > g["hw"], np.interp(g["cx"] + outer * g["hw"], xs, U) - 0.55 * (np.abs(xl - g["cx"]) - g["hw"]), corner_y)
    keep = _sm((ext_y - 3.5 - yl) / 2.0)
    fade = _sm((yl - zy0) / 3.0) * _sm((xl - zx0) / 4.0) * _sm((zx1 - xl) / 4.0)
    m = (dk * keep * fade * 0.9)[..., None]
    out = out.copy()
    out[zy0:zy1, zx0:zx1] = out[zy0:zy1, zx0:zx1] * (1 - m) + cl * m
    o = out * (1 - a[..., None]) + lid * a[..., None]
    insx = (xx > xa) & (xx < xb)
    if amt < 0.99:
        sh = np.exp(-((yy - Mx) / 1.8) ** 2) * (yy > Mx) * (yy < Lx) * insx
        o = o * (1 - 0.32 * sh[..., None])
    # lashes + liner on the lid edge only
    clean = np.stack([grey_closing(src[..., c], size=(7, 5)) for c in range(3)], -1)
    jy = Ux - yy
    lidcol = np.stack([np.interp(xx, xs, gaussian_filter1d(np.array([np.median(clean[int(crease[x]) + 2:max(int(min(U[x], L[x]) - 3), int(crease[x]) + 3), x], 0) if xa - 2 < x < xb + 2 else np.zeros(3) for x in range(W)]), 2.2, axis=0)[:, c]) for c in range(3)], -1)
    dark = np.clip((_lum(lidcol) - _lum(src) - 6.0) / 55.0, 0, 1) * _sm((jy - 1.5) / 1.5) * (jy < np.minimum(11.0, Ux - np.interp(xx, xs, crease) - 2.0))
    if amt >= 0.8:
        j = (yy - Mx) / 0.6
    else:
        j = (Mx - yy) / (1.0 - 0.1 * amt)
    la = map_coordinates(dark, [Ux - j, xx], order=1, mode="constant") * (j >= -0.3) * insx * side
    if amt < 0.8:   # lowering lid: the lashes read as a soft dense fringe, not a comb of single spikes
        la = np.clip(gaussian_filter(la, (0.5, 0.9)) * 1.25, 0, 1)
    thick = (0.85 if amt >= 0.8 else 1.1) + 0.55 * np.clip((xx - g["cx"]) * np.sign(g["outer_dir"]) / g["hw"], 0, 1)
    liner = np.exp(-((yy - (Mx - 0.3)) / thick) ** 2) * insx * side * (0.55 + 0.45 * np.interp(xx, xs, lens))
    A = np.clip(np.maximum(la, liner * 0.9), 0, 1)
    ink = np.array([34, 22, 22], np.float32)
    return o * (1 - A[..., None]) + ink * A[..., None]



def gaze_frame(out: np.ndarray, src: np.ndarray, g: dict, dx: float, dy: float) -> np.ndarray:
    """One eye looking (dx, dy) px away: the iris disc moved rigidly, clipped by the lids."""
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415
    H, W = src.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    Ux, Lx = np.interp(xx, g["xs"], g["U"]), np.interp(xx, g["xs"], g["L"])
    opening = np.clip(yy - Ux + 0.5, 0, 1) * np.clip(Lx - yy + 0.5, 0, 1) * ((xx > g["xa"]) & (xx < g["xb"]))
    (icx, icy), r = g["iris"], g["rad"] + 0.6
    d0 = np.hypot(xx - icx, yy - icy)
    scl = src.copy()
    for y in range(int(icy - r - 2), int(icy + r + 3)):
        hh = r * r - (y - icy) ** 2
        if hh <= 0:
            continue
        w = np.sqrt(hh)
        xl, xr = int(np.floor(icx - w - 2)), int(np.ceil(icx + w + 2))
        cl, cr = src[y, xl - 2:xl + 1].mean(0), src[y, xr:xr + 3].mean(0)
        if y > icy + 0.55 * r:      # bottom of the disc: lit sclera, not the lower lid's shadow
            ref = np.median(src[int(icy) - 2:int(icy) + 3, np.r_[xl - 6:xl - 2, xr + 3:xr + 7]].reshape(-1, 3), 0)
            k = min(1.0, (y - icy - 0.55 * r) / (0.45 * r)) * 0.6
            cl, cr = cl * (1 - k) + ref * k, cr * (1 - k) + ref * k
        tt = (np.arange(xl, xr + 1) - xl) / max(1, xr - xl)
        scl[y, xl:xr + 1] = cl[None] * (1 - tt[:, None]) + cr[None] * tt[:, None]
    scl = gaussian_filter(scl, (0.6, 0.6, 0))
    vis = (opening > 0.95) & (d0 < r)
    core = vis & (d0 > 0.45 * r) & (d0 < 0.8 * r)
    icol = np.median(src[core], axis=0) if core.any() else np.array([90, 60, 45], np.float32)
    full = src.copy()
    hid = ~vis & (d0 < r + 1)           # the iris cap hidden under the lids: its own colour, darker at the limbus
    full[hid] = icol[None] * (1 - 0.35 * _sm((d0[hid] - 0.8 * r) / (0.2 * r)))[:, None]
    ia = np.clip(r - np.hypot(xx - (icx + dx), yy - (icy + dy)) + 0.5, 0, 1)
    moved = np.stack([map_coordinates(full[..., c], [yy - dy, xx - dx], order=1, mode="nearest") for c in range(3)], -1)
    old = np.clip(r + 1.5 - d0, 0, 1)
    eye = src * (1 - old[..., None]) + scl * old[..., None]
    eye = eye * (1 - ia[..., None]) + moved * ia[..., None]
    eye = eye * (1 - 0.18 * (ia * np.exp(-((yy - Ux) / 2.0) ** 2) * (yy > Ux))[..., None])
    a = opening[..., None]
    return out * (1 - a) + eye * a


def lid_field(shape, g: dict, amt: float) -> tuple[np.ndarray, np.ndarray]:
    """Upper lid skin stretched down over the opening: closes `amt` of the eye height, almond-shaped."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    eye_h = max(6.0, g["bottom"] - g["top"])
    k = eye_h * 1.3                       # how much lid skin above the eye is used
    cover = eye_h * amt                   # how much of the opening the lid covers
    y_start, y_end = g["top"] - k, g["top"] + cover
    stretch = 1 - k / (k + cover)         # the lid band [y_start, top] is stretched over [y_start, y_end]
    ell = np.clip(1 - ((xx - g["cx"]) / (g["w"] / 2 * 1.05)) ** 2, 0, 1) ** 0.5  # almond: nothing at the corners
    band = (yy >= y_start) & (yy <= y_end)
    dy = np.where(band, (yy - y_start) * stretch, 0.0)
    # below the lid edge the rest of the opening is compressed: displacement fades from `cover` to 0
    below = (yy > y_end) & (yy <= g["bottom"] + 4)
    rem = max(2.0, (g["bottom"] + 4) - y_end)
    dy = np.where(below, cover * stretch / max(stretch, 1e-6) * 0 + cover * np.clip(1 - (yy - y_end) / rem, 0, 1), dy)
    dy = (dy * ell).astype(np.float32)
    return np.zeros_like(dy), dy


def gaze_field(shape, g: dict, dx_px: float, dy_px: float) -> tuple[np.ndarray, np.ndarray]:
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy = (g["top"] + g["bottom"]) / 2
    ry = max(4.0, (g["bottom"] - g["top"]) / 2 * 0.9)
    rx = g["w"] / 2 * 0.92
    r = ((xx - g["cx"]) / rx) ** 2 + ((yy - cy) / ry) ** 2
    m = np.clip(1 - r, 0, 1) ** 0.6
    return (dx_px * m).astype(np.float32), (dy_px * m).astype(np.float32)


def eyes_alpha(shape, gl: dict, gr: dict) -> np.ndarray:
    pad = 14
    a = np.zeros(shape, np.float32)
    for g in (gl, gr):
        a = np.maximum(a, box_alpha(shape, g["x0"] - pad, g["x1"] + pad, g["top"] - 26, g["bottom"] + 10, 9, 7))
    return a


# ---------------------------------------------------------------- brows
def brow_field(shape, p: dict, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = np.zeros(shape, np.float32); dy = np.zeros(shape, np.float32); alpha = np.zeros(shape, np.float32)
    for side in ("l", "r"):
        g = eye_geom(p, side)
        by = float(p[f"brow_{side}"][1])
        x0, x1 = g["x0"] - 16, g["x1"] + 16
        band = box_alpha(shape, x0, x1, by - 34, by + 30, 10, 14)
        vert = np.exp(-((yy - by) / 16.0) ** 2)
        if mode == "up":
            dy -= 10.5 * band * vert           # whole brow up (2026-11-03: 7 -> 10.5 px, the raise was too subtle on the card)
        else:  # "in": inner end down and toward the nose
            inner_x = g["x1"] if side == "l" else g["x0"]
            near = np.clip(1 - np.abs(xx - inner_x) / (g["w"] * 0.8), 0, 1) ** 1.2
            dy += 7.0 * band * vert * near
            dx += (8.0 if side == "l" else -8.0) * band * vert * near
        alpha = np.maximum(alpha, box_alpha(shape, x0 - 6, x1 + 6, by - 42, by + 34, 12, 12))
    return dx, dy, alpha


def brow_lift(rgb: np.ndarray, p: dict, lift: float) -> np.ndarray:
    """Raised brows as a RIGID lift (2026-11-25): the brow hairs are taken off the skin (grey closing, wider than the
    brow is tall) and put back `lift` px higher. The old warp stretched the band and left the real brow showing
    under the raised one (a second arch)."""
    from scipy.ndimage import gaussian_filter, grey_closing  # noqa: PLC0415
    H, W = rgb.shape[:2]
    out = rgb.copy()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for side in ("l", "r"):
        g = eye_geom(p, side)
        by = float(p[f"brow_{side}"][1])
        x0, x1, y0, y1 = int(g["x0"] - 24), int(g["x1"] + 24), int(by - 30), int(by + 26)
        sub = rgb[y0:y1, x0:x1]
        skin = np.stack([grey_closing(sub[..., c], size=(19, 9)) for c in range(3)], -1)
        skin = gaussian_filter(skin, (2.0, 2.0, 0))
        dark = np.clip((_lum(skin) - _lum(sub) - 6.0) / 45.0, 0, 1)
        band = box_alpha((y1 - y0, x1 - x0), 4, x1 - x0 - 4, 3, y1 - y0 - 3, 5, 5)
        # lifted brow: the photo's pixels and their darkness, `lift` px up (bilinear, sub-pixel safe)
        yl, xl = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
        hair = np.stack([map_coordinates(sub[..., c], [yl + lift, xl], order=1, mode="nearest") for c in range(3)], -1)
        da = map_coordinates(dark, [yl + lift, xl], order=1, mode="constant") * band
        frame = skin * (1 - da[..., None]) + hair * da[..., None]
        out[y0:y1, x0:x1] = sub * (1 - band[..., None]) + frame * band[..., None]
    return out


# ---------------------------------------------------------------- smile (mouth layer)
def smile_field(shape, p: dict, amt: float) -> tuple[np.ndarray, np.ndarray]:
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lx, ly = p["mouth_left"]; rx, ry = p["mouth_right"]
    mw = rx - lx
    dx = np.zeros(shape, np.float32); dy = np.zeros(shape, np.float32)
    for (cx, cy, sgn) in ((lx, ly, -1.0), (rx, ry, 1.0)):
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (mw * 0.42)
        w = np.clip(1 - r, 0, 1) ** 1.4
        dy -= 6.0 * amt * w                    # corner up
        dx += sgn * 4.0 * amt * w              # corner out
    # cheeks lift a touch with it
    for (cx, cy) in ((lx - mw * 0.25, ly - mw * 0.35), (rx + mw * 0.25, ry - mw * 0.35)):
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (mw * 0.5)
        dy -= 2.5 * amt * np.clip(1 - r, 0, 1) ** 1.5
    return dx, dy


# ---------------------------------------------------------------- poses (whole frame)
def pose(img: Image.Image, p: dict, kind: str, bg: tuple) -> Image.Image:
    W, H = img.size
    cx, chin = float(p["chin"][0]), float(p["chin"][1])
    if kind in ("head_tilt_l", "head_tilt_r"):
        ang = 3.0 if kind == "head_tilt_l" else -3.0   # rigid roll; 5 deg read as a crooked head
        return img.rotate(ang, resample=Image.BICUBIC, center=(cx, chin + 90), fillcolor=bg)
    # head_down / head_up: RIGID shifts (2026-11-25). They were vertical squashes / stretches of the face, i.e. a
    # warp that changes the face's proportions ("shakal tedha"); a pose may only move the head as one piece.
    if kind == "head_down":
        return img.transform((W, H), Image.AFFINE, (1, 0, 0, 0, 1, -8), resample=Image.BICUBIC, fillcolor=bg)
    if kind == "head_up":
        return img.transform((W, H), Image.AFFINE, (1, 0, 0, 0, 1, 6), resample=Image.BICUBIC, fillcolor=bg)
    if kind == "lean_in":     # toward the camera: scale about the chest
        s = 1.07; ox, oy = W / 2, chin + 140
        return img.transform((W, H), Image.AFFINE, (1 / s, 0, ox - ox / s, 0, 1 / s, oy - oy / s), resample=Image.BICUBIC, fillcolor=bg)
    return img


def main() -> int:
    jpg, meta = PREP / "face.jpg", PREP / "face.json"
    if not jpg.exists() or not meta.exists():
        print("run tran_moment/prepare_portrait.py first", file=sys.stderr)
        return 1
    img = Image.open(jpg).convert("RGB")
    rgb = np.asarray(img).astype(np.float32)
    H, W = rgb.shape[:2]
    p = json.loads(meta.read_text(encoding="utf-8"))["points"]
    OUT.mkdir(exist_ok=True)
    edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = tuple(int(v) for v in np.median(edge, axis=0))
    layers: dict[str, dict] = {}
    gl, gr = eye_geom(p, "l"), eye_geom(p, "r")
    ea = eyes_alpha((H, W), gl, gr)
    ebox = [int(min(gl["x0"], gr["x0"]) - 20), int(min(gl["top"], gr["top"]) - 30), int(max(gl["x1"], gr["x1"]) + 20), int(max(gl["bottom"], gr["bottom"]) + 14)]

    # eyes (v2: lid model + rigid iris, see lid_frame / gaze_frame): blink frames half -> low -> closed
    models = [eye_model(e, W) for e in eye_contours(img).values()]
    def brow_y(g):   # the brow over this eye (nearest brow landmark in x)
        return float(min((p["brow_l"], p["brow_r"]), key=lambda b: abs(b[0] - g["cx"]))[1])
    for name, amt in (("eyes_half", 0.5), ("eyes_low", 0.85), ("eyes_closed", 1.0)):
        o = rgb.copy()
        for g in models:
            o = lid_frame(o, rgb, g, amt, brow_y(g))
        save_layer(o, ea, OUT / f"{name}.png")
        layers[name] = {"file": f"{name}.png", "kind": "eyes", "box": ebox, "closed": amt}
    if "--eyes-only" in sys.argv:   # round 2: rebuild the blink frames only (everything else stays as it is)
        print("wrote eyes_half / eyes_low / eyes_closed")
        return 0
    # eyes: gaze (a glance aside while listening, a short look up while thinking)
    for name, (gx, gy) in (("gaze_left", (-5.0, 0.0)), ("gaze_right", (5.0, 0.0)), ("gaze_up", (0.0, -2.5))):
        o = rgb.copy()
        for g in models:
            o = gaze_frame(o, rgb, g, gx, gy)
        save_layer(o, ea, OUT / f"{name}.png")
        layers[name] = {"file": f"{name}.png", "kind": "eyes", "box": ebox, "gaze": [gx, gy]}
    # brows
    for name, mode in (("brows_up", "up"), ("brows_in", "in")):
        dx, dy, a = brow_field((H, W), p, mode)
        frame = brow_lift(rgb, p, 6.0) if mode == "up" else warp(rgb, dx, dy)
        save_layer(frame, a, OUT / f"{name}.png")
        ys, xs = np.where(a > 0.02)
        layers[name] = {"file": f"{name}.png", "kind": "brows", "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}
    # smile: a resting mouth shape, saved with the mouth set so the puppet can use it as "rest while warm"
    MOUTHS.mkdir(exist_ok=True)
    dx, dy = smile_field((H, W), p, 1.0)
    ma = mouth_alpha((H, W), p).astype(np.float32) / 255.0
    # same dark-seam cleanup as the speech shapes (generate_mouths._clean_teeth_seam): the warm resting mouth
    # must not bring the black line between the lips back (2026-11-13 feedback)
    from generate_mouths import _clean_teeth_seam  # noqa: PLC0415
    clean = _clean_teeth_seam(np.asarray(img), p).astype(np.float32)
    save_layer(warp(clean, dx, dy), ma, MOUTHS / "smile.png")
    layers["smile"] = {"file": "../mouths/smile.png", "kind": "mouth", "note": "closed lips, corners lifted"}
    # poses (whole frame)
    for name in ("head_tilt_l", "head_tilt_r", "head_down", "head_up", "lean_in"):
        pose(img, p, name, bg).save(OUT / f"{name}.png")
        layers[name] = {"file": f"{name}.png", "kind": "pose"}

    (HERE / "reactions.json").write_text(json.dumps({"canvas": {"width": W, "height": H}, "layers": layers,
        "note": "Warps of prepared/face.jpg (no model). eyes/brows layers composite over base.png (or over a mouth shape) in the card's puppet; poses are whole frames."}, indent=1), encoding="utf-8")

    # contact sheet for review
    names = ["base"] + list(layers)
    thumbs = []
    for n in names:
        if n == "base":
            im = img.copy()
        elif layers[n]["kind"] == "pose":
            im = Image.open(OUT / layers[n]["file"]).convert("RGB")
        else:
            im = img.copy().convert("RGBA")
            lay = Image.open((MOUTHS / "smile.png") if n == "smile" else OUT / layers[n]["file"]).convert("RGBA")
            im.alpha_composite(lay); im = im.convert("RGB")
        crop = im.crop((120, 120, 520, 560)) if layers.get(n, {}).get("kind") in ("eyes", "brows", "mouth") else im
        crop = crop.resize((200, int(200 * crop.height / crop.width)))
        canvas = Image.new("RGB", (200, 290), (20, 20, 24)); canvas.paste(crop, (0, 0))
        ImageDraw.Draw(canvas).text((6, 272), n, fill=(255, 255, 255))
        thumbs.append(canvas)
    cols = 6
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 204, rows * 294), (10, 10, 12))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * 204, (i // cols) * 294))
    sheet.save(HERE / "reactions_sheet.png")
    print(f"wrote {len(layers)} reaction layers to {OUT}, reactions.json, mouths/smile.png and reactions_sheet.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
