"""Probe: a handful of LivePortrait edits of Neha's crop, timed, saved as a labelled sheet.
    .venv/Scripts/python.exe -I scripts/lp_probe.py --repo REPO --acr ACR --out DIR"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from lp_core import LP, face_crop_box, load_rgb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--acr", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
tm = a.acr / "tran_moment"
img = load_rgb(tm / "path_b" / "mouths" / "base.png")
box = face_crop_box(tm / "prepared" / "face.json")
print("crop box", box, "image", img.shape)
crop = np.asarray(Image.fromarray(img).crop(box))
lp = LP(a.repo)
print(f"load {lp.load_s:.1f}s")
t0 = time.time(); lp.prepare(crop); print(f"prepare {time.time() - t0:.1f}s", "source eye/lip ratio", lp.source_ratios())
cases = [("neutral", {}), ("lip .25", {"lip": 0.25}), ("lip .45", {"lip": 0.45}), ("open 60", {"open_": 60}),
         ("grin 8", {"grin": 8}), ("purse -12", {"purse": -12}), ("pout .05", {"pout": 0.05}), ("smile .5", {"smile": 0.5}),
         ("smile 1.0", {"smile": 1.0}), ("brow -15", {"eyebrow": -15}), ("eye 0.0", {"eye": 0.0}), ("gaze x 12", {"gaze_x": 12}),
         ("yaw +4", {"yaw": 4})]
cells = []
for name, kw in cases:
    t0 = time.time(); fr = lp.render(**kw); dt = time.time() - t0
    print(f"{name:10s} {dt:.1f}s"); Image.fromarray(fr).save(a.out / f"{name.replace(' ', '_')}.png")
    cells.append((f"{name} {dt:.1f}s", fr))
src = np.asarray(Image.fromarray(crop).resize((512, 512), Image.LANCZOS))
cells.insert(0, ("SOURCE crop", src))
W = 256; cols = 5; rows = (len(cells) + cols - 1) // cols
sh = Image.new("RGB", (cols * W, rows * (W + 14)), "white"); d = ImageDraw.Draw(sh)
for i, (l, c) in enumerate(cells):
    x, y = (i % cols) * W, (i // cols) * (W + 14)
    sh.paste(Image.fromarray(c).resize((W, W), Image.LANCZOS), (x, y + 14)); d.text((x + 3, y + 1), l, fill="black")
sh.save(a.out / "probe_sheet.png")
print("sheet", a.out / "probe_sheet.png")
