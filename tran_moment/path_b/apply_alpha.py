"""Give the puppet's full-frame images the source photo's own transparency (no background behind her).

prepare_portrait.py frames an RGB image (a transparent PNG is flattened onto a colour first, because MediaPipe
needs RGB). This step runs the ORIGINAL photo's colour and alpha through the same framing and the same pose
warps, so the edges are exact and carry the real colours (no white or black halo):

    prepared/face.png            RGBA portrait (face.jpg stays: launcher bubble, header thumbnail)
    path_b/mouths/base.png       RGBA base the card's puppet draws first
    path_b/reactions/<pose>.png  head_tilt_l/r, head_down, head_up, lean_in with alpha

Mouth, eye and brow layers are patches inside the face (opaque there), so they need nothing.

    python tran_moment/prepare_portrait.py --photo <flat.png>        # landmarks + face.jpg/face.json
    python tran_moment/path_b/generate_mouths.py && python tran_moment/path_b/generate_reactions.py
    python tran_moment/path_b/apply_alpha.py --source <transparent.png> --flat <flat.png>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
TM = HERE.parent
sys.path[:0] = [str(TM), str(HERE)]
import generate_mouths as gm  # noqa: E402
import generate_reactions as gr  # noqa: E402
import prepare_portrait as pp  # noqa: E402

POSES = ("head_tilt_l", "head_tilt_r", "head_down", "head_up", "lean_in")


def rgba(rgb: Image.Image, alpha: Image.Image) -> Image.Image:
    out = rgb.convert("RGB").copy()
    out.putalpha(alpha.convert("L"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="the original photo with its transparent background")
    ap.add_argument("--flat", type=Path, required=True, help="the flattened copy prepare_portrait.py was run on")
    a = ap.parse_args()
    src = ImageOps.exif_transpose(Image.open(a.source)).convert("RGBA")
    flat = ImageOps.exif_transpose(Image.open(a.flat)).convert("RGB")
    if src.size != flat.size:
        print("source and flat must be the same photo (same size)", file=sys.stderr)
        return 1
    p_src = pp.landmarks(flat)                     # the same points prepare_portrait.py framed with
    colour = pp.frame(src.convert("RGB"), p_src)   # real colours (also at semi-transparent edge pixels)
    alpha = pp.frame(src.getchannel("A").convert("RGB"), p_src).getchannel("R")
    base = rgba(colour, alpha)
    face_jpg = Image.open(TM / "prepared" / "face.jpg").convert("RGB")
    # sanity: inside the opaque area the framed colours must match face.jpg (same transform)
    m = np.asarray(alpha) > 250
    diff = np.abs(np.asarray(colour, np.int16) - np.asarray(face_jpg, np.int16))[m].mean() if m.any() else 99
    if diff > 6:
        print(f"framing does not match face.jpg (mean diff {diff:.1f}); re-run prepare_portrait.py on --flat", file=sys.stderr)
        return 1
    p = json.loads((TM / "prepared" / "face.json").read_text(encoding="utf-8"))["points"]
    base.save(TM / "prepared" / "face.png")
    # the launcher bubble / header thumbnail: the portrait on black, like the card (2026-11-13)
    black = Image.new("RGB", base.size, (0, 0, 0)); black.paste(base, mask=base.getchannel("A"))
    black.save(TM / "prepared" / "face_black.jpg", quality=92)
    # the puppet's base: same dark-seam cleanup as the mouth shapes (generate_mouths._clean_teeth_seam)
    gm.sharpen(rgba(Image.fromarray(gm._clean_teeth_seam(np.asarray(colour.convert("RGB")), p)), alpha)).save(HERE / "mouths" / "base.png")
    for name in POSES:
        f = HERE / "reactions" / f"{name}.png"
        if f.exists():
            rgba(gr.pose(colour, p, name, (0, 0, 0)), gr.pose(alpha.convert("RGB"), p, name, (0, 0, 0)).getchannel("R")).save(f)
    cover = (np.asarray(alpha) > 127).mean()
    print(f"alpha applied: base + {len(POSES)} poses; person covers {cover:.0%} of the frame; colour match {diff:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
