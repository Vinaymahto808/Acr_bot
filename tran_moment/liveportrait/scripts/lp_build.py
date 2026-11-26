"""Build the LivePortrait ("_lp") asset set for the card's puppet from Neha's own portrait (CPU, offline, ~25 min).

    tran_moment/liveportrait/.venv/Scripts/python.exe -I tran_moment/liveportrait/scripts/lp_build.py \
        --repo tran_moment/liveportrait/repo --acr . [--only grid|expr|eyes]

Writes NEXT TO the current set (nothing of it is replaced), served by the existing /avatar_puppet/ route:
    path_b/mouths/mouths_lp.json + grid_lp.jpg      lip atlas: same (open x width) grid, same box as mouths.json
    path_b/mouths/expressions_lp.json + path_b/expressions/lp_<name>/fNN.webp   smile_soft, smile_warm, brows_up
    path_b/mouths/reactions_lp.json + path_b/reactions/lp_<eyes>.png            blink half/low/closed, gaze l/r/up
The card uses them with ?avatar=lp (embed.html puppetInit).

Paste-back: every generated 512x512 frame is compared with LivePortrait's own NEUTRAL render of the same crop. Only
where the face actually changed (|frame - neutral|, smoothed and dilated) is the frame used, with a low-frequency
colour correction (source - neutral, Gaussian) so its tone equals the photo; everywhere else the photo's own pixels
stay, bit for bit. A feathered region box limits each layer (mouth/cheeks/jaw for the lip grid, eyes for blinks,
brows for brows), so hair, earrings and background never change. Every frame is cached (cache/), so the paste-back
can be retuned without rendering again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import (binary_closing, binary_erosion, binary_fill_holes, gaussian_filter, label,
                           map_coordinates, maximum_filter, minimum_filter)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lp_core import LP, face_crop_box, load_rgb  # noqa: E402

OPENS = [-1.0, -0.8, -0.6, -0.45, -0.3, -0.15, 0.0, 0.07, 0.14, 0.22, 0.3, 0.38, 0.47, 0.56, 0.66, 0.76, 0.88, 1.0]
WIDTHS = [-0.95, -0.62, -0.3, 0.0, 0.3, 0.62, 0.92]
OPEN_MAX = 72.0     # LivePortrait "lip close <-> open" at the grid's open 1.0 (probe: 70 = a full natural "ah")
GRIN_MAX = 5.0      # "grin" at width +1 (E / I: corners out); 9 showed a crowded, discoloured row of teeth
PURSE_MAX = 13.0    # "pursing" at width -1 (O / U: rounded, forward)
PRESS = -14.0       # "pursing" negative: M / B / P pressed lips


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def grid_params(o: float, k: float) -> dict:
    if o <= 0:
        p = float(smooth((-o - 0.6) / 0.4))
        return {"purse": PRESS * p} if p > 0.01 else {}
    kk = k if k >= 0 else k * (0.25 + 0.75 * float(smooth((o - 0.05) / 0.35)))   # no pout on a barely parted mouth
    op = OPEN_MAX * o * (1.0 - 0.5 * abs(kk))
    out = {"open_": round(op, 3)}
    if kk > 0:
        out["grin"] = round(GRIN_MAX * kk, 3)
    elif kk < 0:
        out["purse"] = round(PURSE_MAX * -kk, 3)
    return out


def sharpen(rgb: np.ndarray) -> np.ndarray:
    """the shared halo-free unsharp mask of the face layers (path_b/generate_mouths.sharpen)"""
    src = Image.fromarray(rgb.astype(np.uint8))
    sh = np.asarray(src.filter(ImageFilter.UnsharpMask(radius=1.3, percent=65, threshold=2)), np.int16)
    s = np.asarray(src, np.int16)
    return np.clip(sh, minimum_filter(s, size=(3, 3, 1)) - 3, maximum_filter(s, size=(3, 3, 1)) + 3).astype(np.float32)


class Builder:
    def __init__(self, lp: LP, base: np.ndarray, box: list[int], cache: Path):
        self.lp, self.base, self.box, self.cache = lp, base.astype(np.float32), box, cache
        cache.mkdir(parents=True, exist_ok=True)
        self.side = box[2] - box[0]
        self.renders, self.secs = 0, 0.0
        self.n_full = self.to_canvas(self.gen({}))
        # LivePortrait's colour bias (its lips come out darker / more purple than the photo): one affine colour map
        # neutral-render -> photo, least squares over the face (eyes to chin), applied to every frame before pasting
        fx0, fy0, fx1, fy1 = box[0] + self.side // 5, box[1] + self.side // 4, box[2] - self.side // 5, box[3] - self.side // 8
        n_ = self.n_full[fy0:fy1, fx0:fx1].reshape(-1, 3); b_ = self.base[fy0:fy1, fx0:fx1].reshape(-1, 3)
        X = np.hstack([n_, np.ones((len(n_), 1), np.float32)])
        self.cmap = np.linalg.lstsq(X, b_, rcond=None)[0].astype(np.float32)          # 4x3
        print("colour map", np.round(self.cmap, 3).tolist(), flush=True)

    def recolour(self, a: np.ndarray) -> np.ndarray:
        return a @ self.cmap[:3] + self.cmap[3]

    def gen(self, params: dict) -> np.ndarray:
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        f = self.cache / f"{key}.png"
        if f.exists():
            return np.asarray(Image.open(f).convert("RGB"))
        t0 = time.time()
        fr = self.lp.render(**params)
        self.secs += time.time() - t0; self.renders += 1
        Image.fromarray(fr).save(f)
        return fr

    def to_canvas(self, fr512: np.ndarray) -> np.ndarray:
        """512x512 crop render -> full canvas array (NaN-free: outside the crop = base)"""
        x0, y0, x1, y1 = self.box
        out = self.base.copy()
        im = np.asarray(Image.fromarray(fr512).resize((self.side, self.side), Image.LANCZOS), np.float32)
        H, W = out.shape[:2]
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(W, x1), min(H, y1)
        out[dy0:dy1, dx0:dx1] = im[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
        return out

    def teeth(self, rgb: np.ndarray, region: list[int], o: float, k: float) -> np.ndarray:
        """Clean LivePortrait's teeth in one lip-atlas cell (round 5b, 2026-11-25: they read as brown / beige, gappy,
        uneven). LP's tooth GEOMETRY stays (it sits right inside its own lips; pasting the donor teeth onto LP's thin,
        irregular openings was tried and misplaced them). Its own tooth pixels are cleaned: (1) re-toned to a warm
        off-white keeping their shading, (2) the dark gaps between teeth closed by a horizontal grey closing of the
        teeth themselves (subtle separations stay), (3) brightness evened out, darker toward the corners and right
        under the upper lip (gum-line shade). Lips, interior, tongue and jaw are untouched."""
        from scipy.ndimage import grey_closing  # noqa: PLC0415
        F = self.F
        if o <= 0.05:
            return rgb
        x0, y0, x1, y1 = region
        cxm, cy = F.cxm, F.lipY + 4.0
        rx0, rx1 = int(cxm - 60 - x0), int(cxm + 60 - x0)
        ry0, ry1 = int(cy - 26 - y0), int(cy + 34 - y0)
        sub = rgb[ry0:ry1, rx0:rx1].astype(np.float32)
        h, w = sub.shape[:2]
        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        Y += ry0 + y0; X += rx0 + x0
        u = F.u(X)
        R, G, Bc = sub[..., 0], sub[..., 1], sub[..., 2]
        lum = sub @ np.array([0.299, 0.587, 0.114], np.float32)
        # measured on LP cells: lips R-G 110-130, LP's (pinkish) teeth R-G 55-80, its dark slits R-G ~80 but dim
        lip = ((R - G) > 95) & (R > 150)
        ell = (np.abs(u) < 0.92) & (((Y - cy) / 24.0) ** 2 + ((X - cxm) / 56.0) ** 2 < 1.0)
        # inside the mouth only: a lip pixel above AND below in the same column (skin outside the lips never counts)
        above = np.maximum.accumulate(lip, axis=0)
        below = np.maximum.accumulate(lip[::-1], axis=0)[::-1]
        teeth = ell & ~lip & above & below & (lum > 118) & ((R - G) < 88)
        teeth = binary_closing(teeth, np.ones((1, 3), bool)) & above & below
        # a lip-gloss highlight passes the colour test too: tooth regions under 15 px are dropped
        lab, nlab = label(teeth)
        if nlab:
            sizes = np.bincount(lab.ravel()); keep = sizes >= 15; keep[0] = False
            teeth = keep[lab]
        if teeth.sum() < 8:
            self.teeth_log.append((o, k, 0)); return rgb
        # the teeth band incl. the dark slits between LP's teeth (closed horizontally and a little vertically)
        # the band, bridging LP's dark slits AND pink gum / inner-lip pixels inside the row (they read as missing
        # teeth, live frame 2026-11-25): horizontal gaps up to 8 px between teeth are filled
        band0 = binary_closing(teeth, np.ones((3, 7), bool)) & ell & above & below
        bridged = binary_closing(band0, np.ones((1, 13), bool)) & ell & above & below   # gaps up to 12 px (an 8 px bridge left a pink block, 2026-11-25)
        band = (band0 & ~lip) | (bridged & ~band0)
        D = F.donor
        dcol_h = (D["a"] > 0.5).sum(axis=0).astype(np.float32)
        out = sub.copy()
        A = np.zeros((h, w), np.float32)
        for j in range(w):
            rows = np.where(band[:, j])[0]
            if len(rows) < 2:
                continue
            # the largest contiguous run in this column
            runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
            run = max(runs, key=len)
            t, b = int(run[0]), int(run[-1])
            if b - t < 1:
                continue
            uj = float(u[0, j])
            if abs(uj) > 0.95:
                continue
            cj = (np.clip(uj, -1, 1) + 1) / 2 * (D["a"].shape[1] - 1)
            hj = float(np.interp(cj, np.arange(len(dcol_h)), dcol_h))
            # upper teeth hang from the lip above; a band resting on the lower lip is the lower row (flipped, darker)
            lower = (t > 2 and not lip[max(0, t - 2), j]) and (b + 2 < h and lip[min(h - 1, b + 2), j])
            ys_ = np.arange(t, b + 1, dtype=np.float32)
            frac = (ys_ - t + 0.5) / (b - t + 1)
            r = (1 - frac) * hj if lower else frac * hj
            px = np.stack([np.interp(r, np.arange(D["rgb"].shape[0]), D["rgb"][:, int(round(cj)), c]) for c in range(3)], -1)
            dy = (ys_ - t)
            shade = (1.0 - 0.30 * np.clip(abs(uj) / 0.85, 0, 1) ** 1.4 - (0.0 if lower else 0.16 * np.exp(-(dy / 1.4) ** 2))
                     + (0.0 if lower else 0.04 * np.exp(-((dy - 3.0) / 1.5) ** 2))) * (0.86 if lower else 0.99)
            shade = np.broadcast_to(np.asarray(shade, np.float32), dy.shape)
            out[t:b + 1, j] = px * shade[:, None] * np.array([1.05, 0.99, 0.93], np.float32)   # warm: next to pink lips a neutral white reads grey-green
            A[t:b + 1, j] = 1.0
        # 1-px soft edge, and the lips overlap the teeth (never drawn over a lip pixel)
        # soft ends: the row fades out over ~3 px toward the mouth corners instead of stopping as a block
        A = gaussian_filter(A, (0.6, 1.6)) * ((~lip) | (bridged & ~band0)) * above * below
        out = sub * (1 - A[..., None]) + out * A[..., None]
        rgb = rgb.copy(); rgb[ry0:ry1, rx0:rx1] = out
        self.teeth_log.append((o, k, int(band.sum())))
        return rgb

    def nose_factor(self, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        """0 on the nose, 1 elsewhere (round 5c, owner: "the nose must not move in any reaction"). LivePortrait redraws
        the nostrils and tip with the mouth; the photo's own nose is kept in every layer. Nose = the bridge from between
        the inner eye corners down to the tip, plus the alae / nostrils, from face.json; the edge is feathered outward
        into the cheeks. Below the nostril base the generated change ramps in over ~10 px, so the philtrum still
        follows the upper lip but is exactly 0 where it meets the nose."""
        P = self.pts
        cx = float(P["nose_tip"][0]); nb = float(P["nose_bottom"][1])
        top = 0.5 * (P["eye_l_inner"][1] + P["eye_r_inner"][1]) - 8.0
        ty = float(P["nose_tip"][1])
        hw = 11.0 + 12.0 * np.clip((yy - top) / max(1.0, ty - 8.0 - top), 0, 1)
        bridge = (yy >= top) & (yy <= ty) & (np.abs(xx - cx) <= hw)
        alae = ((xx - cx) / 32.0) ** 2 + ((yy - (nb - 9.0)) / 12.5) ** 2 <= 1.0
        nose = (bridge | alae).astype(np.float32)
        # 1 on the nose; the skin hugging the alae / bridge stays too, fading out over ~10-12 px into the cheeks (a
        # 4 px edge left LP's cheek motion right against the nostrils, which read as the nose changing shape)
        keep = np.clip(gaussian_filter(nose, 7.5) * 2.2, 0, 1)
        philtrum = np.where(np.abs(xx - cx) < 28.0, smooth((yy - (nb + 1.0)) / 10.0), 1.0)
        philtrum = np.where(yy < nb - 14.0, 1.0, philtrum)               # only the strip right under the nostril base
        return (1.0 - keep) * np.where(nose > 0, 0.0, philtrum)

    def compose(self, params: dict, region: list[int], feather: int = 10, floor_y: float | None = None, teeth=None,
                ceil_y: float | None = None):
        """(rgb, alpha) of `region` [x0,y0,x1,y1]: the generated change pasted onto the photo, colour-matched"""
        G = self.to_canvas(self.gen(params))
        x0, y0, x1, y1 = region
        g, n, b = self.recolour(G[y0:y1, x0:x1]), self.recolour(self.n_full[y0:y1, x0:x1]), self.base[y0:y1, x0:x1]
        # where the face changed (not the renderer's reconstruction noise)
        d = gaussian_filter(np.abs(g - n).max(-1), 1.5)
        m = smooth((d - 2.5) / 9.0)
        m = gaussian_filter(maximum_filter(m, size=11), 5.0)     # soft edge: no ring where the change fades out
        # colour / tone: the renderer's own offset from the photo (low frequencies), measured ONLY where nothing moved
        # and filled into the moving area from around it (normalised convolution). Measured in place it carried the
        # closed lips' colour offset onto the teeth that open there: beige / pink teeth, magenta lips (first build).
        wgt = np.clip(1.0 - m * 1.6, 0, 1) ** 2
        num = gaussian_filter((b - n) * wgt[..., None], (9, 9, 0)); den = gaussian_filter(wgt, 9)[..., None]
        corr = num / np.maximum(den, 1e-3)
        corr = np.where(den > 0.02, corr, gaussian_filter(b - n, (9, 9, 0)))
        gc = sharpen(np.clip(g + corr, 0, 255))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        f2 = feather + 2
        win = (smooth((xx - x0 - f2) / feather) * smooth((x1 - 1 - f2 - xx) / feather)
               * smooth((yy - y0 - f2) / feather) * smooth((y1 - 1 - f2 - yy) / feather))
        if floor_y is not None:      # nothing above this row (the brows' own channel lives there)
            win = win * smooth((yy - floor_y) / 4.0)
        if ceil_y is not None:       # nothing below this row (a blink moves the upper lid, not the skin under the eye)
            win = win * smooth((ceil_y - yy) / 4.0)
        a = np.clip(m * win * self.nose_factor(yy, xx), 0, 1)
        rgb = b * (1 - a[..., None]) + gc * a[..., None]
        if teeth is not None:
            rgb = self.teeth(rgb, region, *teeth)
        return np.clip(rgb, 0, 255), a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--acr", type=Path, required=True)
    ap.add_argument("--only", default="all")
    a = ap.parse_args()
    tm = a.acr / "tran_moment"; pb = tm / "path_b"
    base = load_rgb(pb / "mouths" / "base.png")
    box = face_crop_box(tm / "prepared" / "face.json")
    lp = LP(a.repo)
    crop = np.asarray(Image.fromarray(base).crop(box))
    lp.prepare(crop)
    B = Builder(lp, base, box, tm / "liveportrait" / "cache")
    # the photo's mouth geometry + the real donor teeth (the avatar_v2 renderer's Face)
    sys.path.insert(0, str(tm / "avatar_v2")); sys.path.insert(0, str(pb))
    import build_reference_pack as rp  # noqa: PLC0415
    B.F = rp.Face(np.asarray(Image.open(tm / "prepared" / "face.jpg").convert("RGB")),
                  json.loads((tm / "prepared" / "face.json").read_text(encoding="utf-8"))["points"])
    B.teeth_log = []
    B.pts = json.loads((tm / "prepared" / "face.json").read_text(encoding="utf-8"))["points"]
    t_all = time.time()
    version = hashlib.sha1()

    if a.only in ("all", "grid"):
        m = json.loads((pb / "mouths.json").read_text(encoding="utf-8"))
        gx, gy, gw, gh = m["grid"]["box"]
        region = [gx, gy, gx + gw, gy + gh]
        atlas = Image.new("RGB", (gw * len(OPENS), gh * len(WIDTHS)))
        for r, k in enumerate(WIDTHS):
            for c, o in enumerate(OPENS):
                rgb, _ = B.compose(grid_params(o, k), region, feather=m["grid"].get("feather", 10), teeth=(o, k))
                atlas.paste(Image.fromarray(rgb.astype(np.uint8)), (c * gw, r * gh))
            print(f"grid row {r + 1}/{len(WIDTHS)}  renders {B.renders}  {B.secs:.0f}s", flush=True)
        atlas.save(pb / "mouths" / "grid_lp.jpg", quality=95, subsampling=0)
        version.update((pb / "mouths" / "grid_lp.jpg").read_bytes())
        lpm = dict(m)
        lpm["grid"] = {**m["grid"], "file": "grid_lp.jpg", "opens": OPENS, "widths": WIDTHS,
                       "note": "LivePortrait (tran_moment/liveportrait/scripts/lp_build.py): neural renders of the same photo, pasted back where they change"}
        lpm["grid"].pop("tuck_w", None)
        lpm["version"] = "lp" + version.hexdigest()[:10]
        (pb / "mouths" / "mouths_lp.json").write_text(json.dumps(lpm, indent=1), encoding="utf-8")

    if a.only in ("all", "expr"):
        ex = json.loads((pb / "expressions.json").read_text(encoding="utf-8"))
        layers = {}
        # LivePortrait's smile parts the lips and shows a dark gap between the front teeth (reads as a missing tooth,
        # build 2026-11-25): the owner's gentle smile is closed-mouth, so the photo-warp smiles stay (copied below)
        specs = {"brows_up": ("brows", 6, lambda t: {"eyebrow": round(9.0 * t, 4)})}
        for name, (region_kind, n, fn) in specs.items():
            L = ex["layers"][name]
            x0, y0 = L["box"]; w, h = L["size"]
            region = [x0, y0, x0 + w, y0 + h]
            out = pb / "expressions" / f"lp_{name}"
            out.mkdir(parents=True, exist_ok=True)
            frames = []
            for i in range(1, n + 1):
                t = float(smooth(i / n))
                rgb, al = B.compose(fn(t), region, feather=8)
                im = Image.fromarray(np.dstack([rgb, al * 255]).astype(np.uint8), "RGBA")
                p = out / f"f{i:02d}.webp"; im.save(p, lossless=True, quality=100, method=4)
                version.update(p.read_bytes()); frames.append(f"lp_{name}/f{i:02d}.webp")
            layers[name] = {**L, "frames": frames, "note": "LivePortrait"}
            print(f"expr {name} done, renders {B.renders} {B.secs:.0f}s", flush=True)
        # the worried inner-brow raise has no LivePortrait control (its eyebrow slider only raises or frowns): the
        # current photo-warp sequence is kept for concern
        for keep in ("brows_concern", "smile_soft", "smile_warm"):
            layers[keep] = ex["layers"][keep]
        (pb / "mouths" / "expressions_lp.json").write_text(json.dumps(
            {"version": "lp" + version.hexdigest()[:10], "layers": layers, "moods": ex.get("moods", {}),
             "note": "LivePortrait set (lp_build.py): brows_up from LivePortrait; brows_concern, smile_soft, smile_warm = the photo-warp sequences"}, indent=1), encoding="utf-8")

    if a.only in ("all", "eyes"):
        rx = json.loads((pb / "reactions.json").read_text(encoding="utf-8"))
        region = [205, 186, 435, 262]
        H, W = base.shape[:2]
        layers = {}
        # blinks only: LivePortrait's eyeball moves smeared the sclera and iris (probe 2026-11-25), so the gaze layers
        # stay the current rigid-iris ones. floor_y 198: the brows' channel above is never touched (nor the bindi)
        for name, params in (("eyes_half", {"eye": 0.18}), ("eyes_low", {"eye": 0.09}), ("eyes_closed", {"eye": 0.0})):
            rgb, al = B.compose(params, region, feather=6, floor_y=198.0, ceil_y=246.0)
            full = np.zeros((H, W, 4), np.uint8)
            x0, y0, x1, y1 = region
            full[y0:y1, x0:x1, :3] = rgb.astype(np.uint8); full[y0:y1, x0:x1, 3] = (al * 255).astype(np.uint8)
            p = pb / "reactions" / f"lp_{name}.png"; Image.fromarray(full, "RGBA").save(p)
            version.update(p.read_bytes())
            old = rx["layers"].get(name, {"kind": "eyes"})
            layers[name] = {**old, "file": f"lp_{name}.png", "note": "LivePortrait"}
        for name in ("gaze_left", "gaze_right", "gaze_up"):
            if name in rx["layers"]:
                layers[name] = rx["layers"][name]
        print(f"eyes done, renders {B.renders} {B.secs:.0f}s", flush=True)
        (pb / "mouths" / "reactions_lp.json").write_text(json.dumps(
            {"canvas": rx.get("canvas"), "version": "lp" + version.hexdigest()[:10], "layers": layers,
             "note": "LivePortrait blink / gaze layers (lp_build.py); no brow warp layers, no poses"}, indent=1), encoding="utf-8")

    if B.teeth_log:
        print("teeth:", B.teeth_log, flush=True)
    print(f"total {time.time() - t_all:.0f}s, new renders {B.renders} ({B.secs / max(1, B.renders):.1f}s each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
