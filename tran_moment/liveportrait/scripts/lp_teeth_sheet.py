"""3x teeth zoom: atlas cells (E, I, medium ah, slightly parted, loud ah, O) composited on base, old LP atlas vs new"""
import json, sys
import numpy as np
from PIL import Image, ImageDraw
acr, old_atlas, out = sys.argv[1], sys.argv[2], sys.argv[3]
pb = f"{acr}/tran_moment/path_b"
g = json.load(open(f"{pb}/mouths/mouths_lp.json"))["grid"]; bx, by, bw, bh = g["box"]; f = g["feather"]
base = Image.open(f"{pb}/mouths/base.png").convert("RGB")
def sm(t): t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)
xs, ys = np.arange(bw), np.arange(bh)
mask = np.minimum(sm(xs / f), sm((bw - xs) / f))[None, :] * np.minimum(sm(ys / f), sm((bh - ys) / f))[:, None]
def frame(at, o, w):
    ci = g["opens"].index(o); ri = g["widths"].index(w)
    cell = np.asarray(at.crop((ci * bw, ri * bh, ci * bw + bw, ri * bh + bh)), np.float32)
    a = np.asarray(base, np.float32).copy(); r = a[by:by + bh, bx:bx + bw]
    a[by:by + bh, bx:bx + bw] = r * (1 - mask[..., None]) + cell * mask[..., None]
    return Image.fromarray(a.astype(np.uint8))
cells = [("E", 0.47, 0.92), ("I", 0.38, 0.92), ("medium ah", 0.56, 0.0), ("slightly parted", 0.22, 0.0), ("E soft", 0.3, 0.62), ("loud ah", 1.0, 0.0), ("O", 0.66, -0.95)]
old, new = Image.open(old_atlas).convert("RGB"), Image.open(f"{pb}/mouths/grid_lp.jpg").convert("RGB")
crop = (262, 322, 374, 372); S = 3; cw, ch = 112 * S, 50 * S
sh = Image.new("RGB", (cw * 2, len(cells) * (ch + 14) + 14), "white"); d = ImageDraw.Draw(sh)
d.text((3, 1), "LP before (round 5)", fill="black"); d.text((cw + 3, 1), "LP + donor teeth (round 5b)", fill="black")
for i, (l, o, w) in enumerate(cells):
    y = 14 + i * (ch + 14)
    for j, at in enumerate((old, new)):
        sh.paste(frame(at, o, w).crop(crop).resize((cw, ch), Image.LANCZOS), (j * cw, y + 14)); d.text((j * cw + 3, y + 1), f"{l} (open {o}, width {w})", fill="black")
sh.save(out); print(out)
