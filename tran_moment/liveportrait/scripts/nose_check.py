"""Nose stillness at the ATLAS level: every lip-grid cell, expression frame and reaction layer composited on base.png
the way the card does, vs base. Metric = the coordinator's: grey mean |diff| over the nose box x 270-370, y 240-300
(and its max over the box). python nose_check.py ACR [lp]"""
import json, sys
import numpy as np
from PIL import Image
acr = sys.argv[1]; lp = len(sys.argv) > 2
pb = f"{acr}/tran_moment/path_b"
base = np.asarray(Image.open(f"{pb}/mouths/base.png").convert("RGB"), np.float32)
grey = lambda a: a @ np.array([0.299, 0.587, 0.114], np.float32)
bg = grey(base)
BX = (270, 240, 370, 300)
def score(img):
    d = np.abs(grey(img) - bg)[BX[1]:BX[3], BX[0]:BX[2]]
    return float(d.mean()), float(d.max())
res = {}
m = json.load(open(f"{pb}/mouths/mouths_lp.json" if lp else f"{pb}/mouths.json"))["grid"]
bx, by, bw, bh = m["box"]; f = m["feather"]
at = np.asarray(Image.open(f"{pb}/mouths/{m['file']}").convert("RGB"), np.float32)
sm = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
xs, ys = np.arange(bw), np.arange(bh)
mask = (np.minimum(sm(xs / f), sm((bw - xs) / f))[None, :] * np.minimum(sm(ys / f), sm((bh - ys) / f))[:, None])[..., None]
worst = (0, None)
for r in range(at.shape[0] // bh):
    for c in range(at.shape[1] // bw):
        cell = at[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
        img = base.copy(); reg = img[by:by + bh, bx:bx + bw]; img[by:by + bh, bx:bx + bw] = reg * (1 - mask) + cell * mask
        s = score(img)
        if s[0] > worst[0]: worst = (s[0], (m["opens"][c] if c < len(m["opens"]) else c, m["widths"][r] if r < len(m["widths"]) else r, round(s[1], 1)))
res["grid worst mean (open, width, max)"] = (round(worst[0], 2), worst[1])
ex = json.load(open(f"{pb}/mouths/expressions_lp.json" if lp else f"{pb}/expressions.json"))
for name, L in ex["layers"].items():
    fr = Image.open(f"{pb}/expressions/{L['frames'][-1]}").convert("RGBA")
    img = Image.fromarray(base.astype(np.uint8)).convert("RGBA"); img.alpha_composite(fr, tuple(L["box"][:2]))
    res[f"expr {name}"] = tuple(round(v, 2) for v in score(np.asarray(img.convert("RGB"), np.float32)))
rx = json.load(open(f"{pb}/mouths/reactions_lp.json" if lp else f"{pb}/reactions.json"))
for name, L in rx["layers"].items():
    if L.get("kind") == "pose": continue
    p = f"{pb}/reactions/{L['file'].split('/')[-1]}" if not L["file"].startswith("../") else f"{pb}/mouths/{L['file'].split('/')[-1]}"
    img = Image.fromarray(base.astype(np.uint8)).convert("RGBA"); img.alpha_composite(Image.open(p).convert("RGBA"))
    res[f"layer {name}"] = tuple(round(v, 2) for v in score(np.asarray(img.convert("RGB"), np.float32)))
print("LP set" if lp else "DEFAULT set")
for k, v in res.items(): print(f"  {k:40s} mean/max {v}")
