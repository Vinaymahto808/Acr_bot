"""3x zoom of LP atlas cells with the detected mouth opening (green) - teeth segmentation debug"""
import json, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_fill_holes, label, binary_dilation
acr, out = sys.argv[1], sys.argv[2]
pb = f"{acr}/tran_moment/path_b"
m = json.load(open(f"{pb}/mouths/mouths_lp.json"))["grid"]; bx, by, bw, bh = m["box"]
at = Image.open(f"{pb}/mouths/grid_lp.jpg").convert("RGB")
cells = [(0.07, 0.0), (0.22, 0.0), (0.38, 0.0), (0.47, 0.0), (0.56, 0.0), (0.76, 0.0), (1.0, 0.0), (0.47, 0.92), (0.38, 0.92), (0.66, -0.95), (0.3, 0.62), (0.22, 0.92)]
cx, cy = 317 - bx, 352 - by; S = 3; cw, ch = 150 * S, 70 * S
sh = Image.new("RGB", (cw * 2, (ch + 14) * ((len(cells) + 1) // 2)), "white"); d = ImageDraw.Draw(sh)
for i, (o, k) in enumerate(cells):
    ci = m["opens"].index(o); ri = m["widths"].index(k)
    cell = np.asarray(at.crop((ci * bw, ri * bh, ci * bw + bw, ri * bh + bh)), np.float32)
    sub = cell[cy - 30:cy + 40, cx - 75:cx + 75]
    R, G = sub[..., 0], sub[..., 1]
    lip = binary_closing(((R - G) > 58) & (R > 110), np.ones((3, 3), bool))
    inter = binary_fill_holes(lip) & ~lip
    lab, n = label(inter)
    if n: inter = lab == 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    edge = binary_dilation(inter) & ~inter
    vis = sub.copy(); vis[edge] = [0, 255, 0]
    im = Image.fromarray(vis.astype(np.uint8)).resize((cw, ch), Image.NEAREST)
    x, y = (i % 2) * cw, (i // 2) * (ch + 14)
    sh.paste(im, (x, y + 14)); d.text((x + 3, y + 1), f"open {o} width {k}  opening px {int(inter.sum())}", fill="black")
sh.save(out); print(out)
